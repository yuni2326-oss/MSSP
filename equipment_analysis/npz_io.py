"""npz raw 세션 리더 — 2026-07-20 18:01 DAQ 변경 이후 저장형식(가산 전용 모듈).

프로덕션 `dataio`는 CSV(`*_raw.csv` + `*_fft_chN.csv`)만 읽는다. 2026-07-20 18:01:59부터
취득이 `*_raw.npz` 단일파일로 바뀌어(7채널 × 256k, fs 51.2kHz) 기존 파이프라인에 잡히지 않으므로,
캐비테이션 실험 분석경로를 위해 별도 리더를 둔다. **기존 dataio는 건드리지 않는다**(프로덕션 불변).

npz 스키마: data(7,256000) float32, sample_rate() float64, unit() str, timestamp() str.

⚠️ **unit 필드는 전 채널 공통 단일 스칼라("m/s^2")인데 ch6은 공기 마이크로폰으로 실제 단위가 Pa다.**
   즉 파일 메타데이터가 ch6을 잘못 라벨한다. 가속도 채널용 배율/교정을 ch6에 적용하면 안 되고,
   ch6은 음압(Pa)으로 따로 다뤄야 한다. `channel_units()`가 이 예외를 명시적으로 돌려준다.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

_STAMP_RE = re.compile(r"^(\d{8}_\d{6})_raw\.npz$")

# ch6 = 공기 마이크로폰(2026-07-20 추가). 파일의 unit 필드는 이 채널을 m/s^2로 잘못 표기한다.
MIC_CHANNEL = 6
MIC_UNIT = "Pa"
ACCEL_UNIT = "m/s^2"

# 계측구성 경계(문서화용; baseline 분리에 필수)
CH0_RELOCATED_FROM = "20260721_100038"   # ch0 토출밸브 → 임펠러 옆
NPZ_FORMAT_FROM = "20260720_180159"      # CSV → npz 전환(= IEPE 수정 시점)


@dataclass
class NpzSession:
    stamp: str
    path: Path


def discover_npz_sessions(data_root: str | Path, date_from: str = "", date_to: str = "") -> list[NpzSession]:
    """`YYYYMMDD_HHMMSS_raw.npz` 세션 목록(스탬프 정렬). date_from/to는 스탬프 문자열 비교."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"데이터 디렉토리가 없습니다: {root}")
    out: list[NpzSession] = []
    for p in sorted(root.iterdir()):
        m = _STAMP_RE.match(p.name)
        if not m:
            continue
        stamp = m.group(1)
        if date_from and stamp < date_from:
            continue
        if date_to and stamp > date_to:
            continue
        out.append(NpzSession(stamp=stamp, path=p))
    return out


# 펌프 비운전(드라이브 통전·모터 정지) QC. 실측 사례: 20260727_082305·182253은 ch0 RMS≈0.87
# (정상 중앙값 6.9의 ~13%, 정상 최소 5.9)이고 축 1×가 사라졌다. 이런 세션이 baseline·Tier-1
# 자연변동 풀에 섞이면 대역 log-ratio가 ~20 dB로 튀어 p95를 오염시킨다(robust 통계는 견디나
# 꼬리 통계는 붕괴). 정상 최소(5.9)와 펌프정지(0.87) 사이가 넓어 ch0 RMS 단독 임계로 안전히 분리된다.
PUMP_RMS_FLOOR = 3.0   # ch0 광대역 RMS 하한(m/s²); 미만이면 '펌프 비운전'으로 제외


def is_pump_running(chans: dict[str, np.ndarray], rms_floor: float = PUMP_RMS_FLOOR) -> bool:
    """ch0 광대역 RMS로 펌프 운전 여부 판정(시간영역, FFT 불필요)."""
    x0 = chans.get("ch0")
    if x0 is None:
        return True
    return float(np.sqrt(np.mean(np.square(x0)))) >= rms_floor


def discover_running_sessions(data_root: str | Path, date_from: str = "", date_to: str = "",
                              rms_floor: float = PUMP_RMS_FLOOR) -> list[NpzSession]:
    """`discover_npz_sessions` + 펌프 비운전 세션 제외(각 세션 로드해 ch0 RMS 확인).
    baseline/Tier-1/열 분석의 정상-세션 풀은 이 함수로 만들어야 pump-off 오염을 막는다."""
    out: list[NpzSession] = []
    for s in discover_npz_sessions(data_root, date_from, date_to):
        chans, _fs, _meta = load_npz_raw(s.path)
        if is_pump_running(chans, rms_floor):
            out.append(s)
    return out


def channel_units(n_channels: int) -> list[str]:
    """채널별 실제 물리단위. 파일의 단일 unit 필드를 신뢰하지 말고 이 함수를 쓸 것(ch6=Pa 예외)."""
    return [MIC_UNIT if i == MIC_CHANNEL else ACCEL_UNIT for i in range(n_channels)]


def load_npz_raw(path: str | Path) -> tuple[dict[str, np.ndarray], float, dict]:
    """npz 세션 → ({"ch0":arr, ...}, fs, meta). 배열은 float64로 승격.
    meta: {stamp, unit_field(파일 그대로), units(채널별 실제단위), n_channels, n_samples}."""
    z = np.load(Path(path), allow_pickle=True)
    data = np.asarray(z["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data 형상이 (n_ch, n_samp)가 아닙니다: {data.shape}")
    fs = float(z["sample_rate"])
    n_ch, n_samp = data.shape
    chans = {f"ch{i}": data[i] for i in range(n_ch)}
    meta = {
        "stamp": str(z["timestamp"]) if "timestamp" in z else Path(path).name[:15],
        "unit_field": str(z["unit"]) if "unit" in z else "",
        "units": channel_units(n_ch),            # ch6=Pa 예외 반영
        "n_channels": n_ch,
        "n_samples": n_samp,
    }
    return chans, fs, meta


def amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """저장 CSV fft와 동일 규약(Hann, |rfft|/N*4)의 단측 진폭 스펙트럼 — CSV 시대와 비교 가능하게."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mag = np.abs(np.fft.rfft(x * np.hanning(n))) / n * 4.0
    return np.fft.rfftfreq(n, d=1.0 / fs), mag


def electrical_mask(freq: np.ndarray, carrier: float = 7820.0, n_harm: int = 3,
                    width: float = 300.0, sidebands: tuple[float, ...] = (8180.0,),
                    sideband_width: float = 40.0) -> np.ndarray:
    """전기 기원 성분(VFD PWM 콤 + 명시 사이드밴드) 제외 마스크(True=유지).

    콤은 k차 반폭 k·width로 확산되고, **carrier+BPF 사이드밴드(≈8180Hz)는 콤 규칙으로
    제거되지 않는다** — ch0(임펠러 옆)에서는 이 사이드밴드가 스펙트럼 최대선이므로
    광대역/HF 특징을 오염시킨다. 기본값으로 함께 제외한다."""
    freq = np.asarray(freq, dtype=float)
    keep = np.ones_like(freq, dtype=bool)
    for k in range(1, n_harm + 1):
        keep &= np.abs(freq - carrier * k) > width * k
    for fsb in sidebands:
        keep &= np.abs(freq - fsb) > sideband_width
    return keep


def band_rms(freq: np.ndarray, mag: np.ndarray, band: tuple[float, float],
             keep: np.ndarray | None = None) -> float:
    """진폭 스펙트럼에서 대역 RMS(전기성분 마스크 적용 가능)."""
    lo, hi = band
    sel = (freq >= lo) & (freq < hi)
    if keep is not None:
        sel &= keep
    return float(np.sqrt(np.sum(np.asarray(mag)[sel] ** 2) / 2.0))
