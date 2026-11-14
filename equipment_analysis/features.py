# pump_analysis/features.py
"""진동 특징 추출(시간/주파수 영역) 및 회전수 검출."""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from scipy.signal import butter, sosfiltfilt, hilbert
from . import dataio


# 표준 단열 깊은홈 볼베어링 제원(공개 카탈로그): model -> (볼수 n, 볼지름 d[mm], 피치경 D[mm]).
# 현장 확인 형번(6204 구동단 / 6203 반구동단) + 인접 6205 포함.
BEARING_CATALOG: dict[str, tuple[float, float, float]] = {
    "6203": (8, 6.747, 28.5),
    "6204": (8, 7.940, 33.5),
    "6205": (9, 7.940, 39.0),
}


def _resolve_bearing(spec: dict) -> tuple[float, float, float, float]:
    """베어링 스펙 dict → (n, d, D, theta_deg). {"model":"6204"} 카탈로그 조회 또는
    명시적 {"n","d","D"[,"theta"]} 지원."""
    theta = float(spec.get("theta", 0.0))
    model = spec.get("model")
    if model is not None:
        if model not in BEARING_CATALOG:
            raise KeyError(f"알 수 없는 베어링 형번: {model} (카탈로그: {list(BEARING_CATALOG)})")
        n, d, D = BEARING_CATALOG[model]
        return float(n), float(d), float(D), theta
    return float(spec["n"]), float(spec["d"]), float(spec["D"]), theta


def envelope_spectrum(sig: np.ndarray, fs: float,
                      band: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """포락선 분석: band(Hz)로 밴드패스 → 힐베르트 포락선 → DC 제거 → 편측 FFT.
    베어링 충격의 반복 간격(포락선)을 주파수 영역에 드러낸다. (freq, mag) 반환."""
    sig = np.asarray(sig, dtype=float)
    nyq = fs / 2.0
    lo, hi = band
    hi = min(hi, nyq * 0.99)               # fs가 낮아 hi가 나이퀴스트 초과 시 클램프
    if not (0 < lo < hi):
        raise ValueError(f"밴드가 유효하지 않습니다: band={band}, fs={fs}")
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    bp = sosfiltfilt(sos, sig)
    env = np.abs(hilbert(bp))
    env = env - env.mean()                 # DC(평균 진폭) 제거
    n = len(env)
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(env * win)) / n * 2.0
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, mag


def envelope_features(sig: np.ndarray, fs: float, defect_freqs: dict[str, float],
                      band: tuple[float, float] = (2000.0, 10000.0),
                      n_harm: int = 3, tol_frac: float = 0.01,
                      shaft_fr: float = 0.0, floor_hw: float = 40.0) -> dict[str, float]:
    """포락선 스펙트럼에서 각 결함주파수의 **SNR**(peak/국소 floor)을 특징화(fable F2 수정).
    이전 진폭합은 정상펌프의 정수 축차수 포락에너지를 harvest함(6203/6204 결함차수가 정수차수
    ±2%내). 수정: (1)tol_frac 0.02→0.01, (2)shaft_fr 주어지면 정수 축차수(m·shaft_fr)와 분해능
    내로 겹치는 하모닉은 신뢰불가로 배제, (3)각 하모닉 SNR=peak/median(±floor_hw 국소 floor, peak창
    제외), env_{name}=유효 하모닉 중 최대 SNR. 슬립으로 실결함선은 정수차수서 1~2% 벗어남
    (Randall&Antoni2011)이라 '정확히 정수차수=결함 아님' 판별이 강력. shaft_fr=0=배제 안 함."""
    freq, mag = envelope_spectrum(sig, fs, band)
    df = float(freq[1] - freq[0]) if len(freq) > 1 else 1.0
    out: dict[str, float] = {}
    for name, f0 in defect_freqs.items():
        best = 0.0
        if f0 and f0 > 0:
            for k in range(1, n_harm + 1):
                target = f0 * k
                if shaft_fr > 0:                       # 정수 축차수 충돌 하모닉 배제(분해능 기준)
                    m = round(target / shaft_fr)
                    if m >= 1 and abs(target - m * shaft_fr) <= max(2.0 * df, 0.005 * target):
                        continue
                tol = max(target * tol_frac, 2.0 * df)
                win = np.abs(freq - target) <= tol
                if not win.any():
                    continue
                pk = float(mag[win].max())
                fl = (np.abs(freq - target) <= floor_hw) & (~win)
                floor = float(np.median(mag[fl])) if fl.any() else float(np.median(mag))
                if floor > 0:
                    best = max(best, pk / floor)
        out[f"env_{name}"] = best
    return out


def bearing_defect_freqs(fr: float, n: float, d: float, D: float,
                         theta_deg: float = 0.0) -> dict[str, float]:
    """축 회전주파수 fr(Hz)과 베어링 제원(볼수 n, 볼지름 d, 피치경 D, 접촉각 θ)으로
    결함 주파수(Hz) 4종을 반환. d와 D는 같은 단위면 됨(비율만 사용)."""
    ratio = (d / D) * math.cos(math.radians(theta_deg))
    return {
        "bpfo": (n / 2.0) * fr * (1.0 - ratio),
        "bpfi": (n / 2.0) * fr * (1.0 + ratio),
        "bsf":  (D / (2.0 * d)) * fr * (1.0 - ratio ** 2),
        "ftf":  (fr / 2.0) * (1.0 - ratio),
    }

def time_features(sig: np.ndarray) -> dict[str, float]:
    sig = np.asarray(sig, dtype=float)
    rms = float(np.sqrt(np.mean(sig**2)))
    peak = float(np.max(np.abs(sig)))
    return {
        "rms": rms,
        "peak": peak,
        "p2p": float(np.max(sig) - np.min(sig)),
        "crest": float(peak / rms) if rms > 0 else 0.0,
        "kurtosis": float(stats.kurtosis(sig, fisher=True)),  # excess
        "std": float(np.std(sig)),
        "mean": float(np.mean(sig)),
    }

def detect_rpm(freq: np.ndarray, mag: np.ndarray,
               search_hz: tuple[float, float] = (20.0, 120.0),
               n_harm: int = 6) -> float:
    """축 회전수(rpm) 추정. 단순 argmax는 60Hz 상용전원·2×·인버터출력에 락되므로(fable F1),
    하모닉 comb 점수 Σ_k (1/k)·max|mag| near k·f0 를 최대화하는 f0를 고른다(하모닉족을 가진
    진짜 기본파 선택). 1x 빈 포물선 보간으로 세분. w_k=1/k 로 기본파 가중이 서브하모닉 오검출 방지."""
    freq = np.asarray(freq, dtype=float); mag = np.asarray(mag, dtype=float)
    lo, hi = search_hz
    band = (freq >= lo) & (freq <= hi)
    if not band.any() or len(freq) < 2:
        return 0.0
    df = float(np.median(np.diff(freq)))
    fmax = float(freq[-1])

    def _amp(fc: float, tol: float) -> float:
        l = int(np.searchsorted(freq, fc - tol)); r = int(np.searchsorted(freq, fc + tol))
        return float(mag[l:r].max()) if r > l else 0.0

    best_f, best_s = 0.0, -1.0
    for i in np.where(band)[0]:
        f0 = float(freq[i]); s = 0.0
        for k in range(1, n_harm + 1):
            fk = f0 * k
            if fk > fmax:
                break
            s += _amp(fk, max(1.5 * df, 0.005 * k * f0)) / k
        if s > best_s:
            best_s, best_f = s, f0
    tol1 = max(1.5 * df, 0.005 * best_f)                   # comb가 고른 f0 근처 실제 1x 피크로 snap
    l = int(np.searchsorted(freq, best_f - tol1)); r = int(np.searchsorted(freq, best_f + tol1))
    j = (l + int(np.argmax(mag[l:r]))) if r > l else int(np.argmin(np.abs(freq - best_f)))
    best_f = float(freq[j])
    if 0 < j < len(freq) - 1:                              # 1x 빈 포물선 보간
        den = mag[j - 1] - 2 * mag[j] + mag[j + 1]
        if den != 0:
            best_f = freq[j] + 0.5 * (mag[j - 1] - mag[j + 1]) / den * df
    return best_f * 60.0


def order_features(freq: np.ndarray, mag: np.ndarray, rpm: float,
                   n_harm: int = 10) -> dict[str, float]:
    """rpm 차수 도메인 무차원 특징(fable P1, 속도-robust). 정렬불량(2x/1x)·느슨함(하모닉합)·
    서브싱크(0.5x)·블레이드패스(6x/1x)를 1x로 정규화. 6깃이라 BPF=6×이므로 ord_sum은 k=6·12
    제외(BPF 오염 방지, F3). 1x가 잡음층 이하면 비율 미정의→빈 dict(NaN 행으로 안전 제외)."""
    freq = np.asarray(freq, dtype=float); mag = np.asarray(mag, dtype=float)
    if rpm <= 0 or len(freq) < 2:
        return {}
    f1 = rpm / 60.0; df = float(np.median(np.diff(freq))); fmax = float(freq[-1])

    def amp(o: float) -> float:
        fk = f1 * o
        if fk > fmax:
            return 0.0
        tol = max(1.5 * df, 0.005 * o * f1)
        l = int(np.searchsorted(freq, fk - tol)); r = int(np.searchsorted(freq, fk + tol))
        return float(mag[l:r].max()) if r > l else 0.0

    a1 = amp(1.0)
    out = {"ord_1x": a1}
    if a1 <= 0:
        return out
    out["ord_2x_ratio"] = amp(2.0) / a1
    out["ord_0p5x_ratio"] = amp(0.5) / a1
    out["ord_sum_ratio"] = sum(amp(k) for k in range(2, n_harm + 1) if k not in (6, 12)) / a1
    out["ord_6x"] = amp(6.0)
    out["ord_12x"] = amp(12.0)
    out["ord_bpf_ratio"] = amp(6.0) / a1       # BPF=6× (6깃) 대 1x
    return out


def spectral_flatness(freq: np.ndarray, mag: np.ndarray,
                      band: tuple[float, float], eps: float = 1e-12) -> float:
    """스펙트럼 평탄도(Wiener entropy) = geomean(P)/mean(P), P=mag². 1=백색(광대역), →0=톤.
    마스크(0)된 bin(PWM 콤 제거부)은 제외해 flatness가 0으로 붕괴하는 것 방지(fable 지적)."""
    freq = np.asarray(freq, dtype=float); mag = np.asarray(mag, dtype=float)
    lo, hi = band
    sel = (freq >= lo) & (freq <= hi) & (mag > 0)          # 마스크 bin 제외
    if int(sel.sum()) < 4:
        return 0.0
    p = mag[sel] ** 2
    am = float(np.mean(p))
    return float(np.exp(np.mean(np.log(p + eps))) / am) if am > 0 else 0.0


def band_rms_spectral(freq: np.ndarray, mag: np.ndarray,
                      band: tuple[float, float]) -> float:
    """스펙트럼에서 대역 RMS = sqrt(Σ mag²). HF 유압/캐비테이션 대역 에너지(양측 변화 지표)."""
    freq = np.asarray(freq, dtype=float); mag = np.asarray(mag, dtype=float)
    lo, hi = band
    sel = (freq >= lo) & (freq <= hi)
    return float(np.sqrt(np.sum(mag[sel] ** 2))) if sel.any() else 0.0


def band_kurtosis(sig: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """대역통과 신호의 초과첨도 = 임펄스성 지표(초기 베어링·rub). 포락선 라인의 보완적 stage-1
    검출자(Antoni). band는 구조공진 반송 대역(예 2–7kHz). fs 낮으면 나이퀴스트로 클램프."""
    sig = np.asarray(sig, dtype=float)
    nyq = fs / 2.0
    lo, hi = band
    hi = min(hi, nyq * 0.99)
    if not (0 < lo < hi):
        return 0.0
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return float(stats.kurtosis(sosfiltfilt(sos, sig), fisher=True))

def spectrum_features(freq: np.ndarray, mag: np.ndarray, rpm: float,
                      n_harm: int = 5) -> dict[str, float]:
    freq = np.asarray(freq, dtype=float); mag = np.asarray(mag, dtype=float)
    out: dict[str, float] = {"peak_hz": rpm / 60.0 if rpm else 0.0}
    f1x = rpm / 60.0
    df = float(np.median(np.diff(freq))) if len(freq) > 1 else 0.2
    for k in range(1, n_harm + 1):
        target = f1x * k
        tol = max(df * 1.5, 0.005 * k * f1x)      # 차수별 허용오차(rpm오차 ±0.5%·k) — 상수tol은 k≥5서 놓침(fable)
        sel = np.abs(freq - target) <= tol
        out[f"harm_{k}x"] = float(mag[sel].max()) if (f1x > 0 and sel.any()) else 0.0
    out["band_energy"] = float(np.sum(mag**2))
    return out

def notch_pwm_comb(sig: np.ndarray, fs: float, carrier: float,
                   n_harm: int = 3, width: float = 400.0) -> np.ndarray:
    """VFD PWM 스위칭 캐리어 하모닉 콤(k·carrier, k=1..n_harm)을 raw 신호에서 band-stop 제거.
    각 하모닉 반폭 = width·k (콤은 기본파 불확실도 ±Δ가 k차서 ±kΔ로 퍼짐). 제로위상 sosfiltfilt.
    나이퀴스트 초과 하모닉은 건너뜀. PWM 전기노이즈가 RMS/kurtosis/포락선을 오염시키는 것 방지."""
    sig = np.asarray(sig, dtype=float)
    nyq = fs / 2.0
    out = sig
    for k in range(1, int(n_harm) + 1):
        fc = carrier * k
        hw = width * k
        lo, hi = (fc - hw) / nyq, (fc + hw) / nyq
        if not (0 < lo < hi < 1.0):
            continue
        sos = butter(2, [lo, hi], btype="bandstop", output="sos")
        out = sosfiltfilt(sos, out)
    return out


def _mask_comb(freq: np.ndarray, mag: np.ndarray, carrier: float,
               n_harm: int, width: float) -> np.ndarray:
    """스펙트럼 mag에서 PWM 콤 하모닉 대역을 0으로(band_energy 등에서 전기노이즈 제외)."""
    mag = np.asarray(mag, dtype=float).copy()
    for k in range(1, int(n_harm) + 1):
        fc = carrier * k
        hw = width * k
        mag[(freq >= fc - hw) & (freq <= fc + hw)] = 0.0
    return mag


def resonance_features(freq: np.ndarray, mag: np.ndarray, anchors,
                       search_frac: float = 0.06) -> dict[str, float]:
    """구조 공진 앵커별로 실제 스펙트럼 피크 주파수를 추적. 손상(국소 강성저하)→공진 하향이동을
    포착 — band_energy 등 진폭특징이 못 잡는 '주파수 이동'을 잡는다(candidate3 검출갭 보완).
    앵커 f0마다 [f0(1−search_frac), f0(1+search_frac)]에서 최대 피크 주파수를 찾아:
      res{f0}_f     = 추적 피크 주파수(Hz)
      res{f0}_shift = 분율 이동 = peak/f0 − 1  (음수=하향=강성저하 의심)
    대역 내 에너지가 없으면 (f0, 0.0)(정보 없음 → 이상 아님). 앵커는 OMA/PINN 검증 공진."""
    freq = np.asarray(freq, dtype=float); mag = np.asarray(mag, dtype=float)
    out: dict[str, float] = {}
    for f0 in anchors:
        f0 = float(f0)
        sel = (freq >= f0 * (1 - search_frac)) & (freq <= f0 * (1 + search_frac))
        if sel.any() and float(mag[sel].max()) > 0:
            fpk = float(freq[sel][int(np.argmax(mag[sel]))])
        else:
            fpk = f0
        key = int(round(f0))
        out[f"res{key}_f"] = fpk
        out[f"res{key}_shift"] = fpk / f0 - 1.0 if f0 > 0 else 0.0
    return out


def session_features(session: dataio.Session,
                     channels: list[str] | None = None,
                     rpm_range: tuple[float, float] = (20.0, 120.0),
                     bearings: dict[str, dict] | None = None,
                     env_band: tuple[float, float] = (2000.0, 10000.0),
                     calibration: dict | None = None,
                     pwm_notch: dict | None = None,
                     resonances: dict | None = None,
                     speed_gate: bool = False) -> dict[str, float]:
    t, chans, fs = dataio.load_raw(session.raw_path)
    # 보정(calibration): 특정 시각 이전 세션 raw/fft를 배율로 스케일(단위설정 변경 정합).
    # scale은 스칼라(전 채널 동일) 또는 {ch: 배율} dict(채널별, 예: 4464 g→m/s² vs 4492 V→m/s²).
    # 진폭선형 특징(rms/peak/harm/env)은 배율배, band_energy는 배율²배, crest/kurtosis는 불변.
    _cal_scale = None
    if calibration and calibration.get("before") and session.stamp < calibration["before"]:
        _cal_scale = calibration.get("scale", 1.0)

    def _chscale(ch: str) -> float:
        if _cal_scale is None:
            return 1.0
        if isinstance(_cal_scale, dict):
            return float(_cal_scale.get(ch, 1.0))
        return float(_cal_scale)

    if _cal_scale is not None:
        chans = {c: (v * _chscale(c) if _chscale(c) != 1.0 else v) for c, v in chans.items()}
    # PWM 스위칭 콤 제거(설정 시): time/envelope 특징을 VFD 전기노이즈로부터 정화. rpm(저주파)엔 무영향.
    _pn = None
    if pwm_notch and fs > 0:
        _pn = (float(pwm_notch["carrier"]), int(pwm_notch.get("n_harm", 3)),
               float(pwm_notch.get("width", 400.0)))
        chans = {c: notch_pwm_comb(v, fs, *_pn) for c, v in chans.items()}
    keys = channels or list(chans.keys())
    # 회전수: 가용한 첫 fft 채널에서 검출. rpm_range(Hz)는 장비 config에서 주입(펌프 1x 밴드).
    rpm = 0.0
    for ch_idx in sorted(session.fft_paths):
        fr, mg = dataio.load_fft(session.fft_paths[ch_idx])
        rpm = detect_rpm(fr, mg, rpm_range)
        if rpm > 0:
            break
    # 속도 안정 게이트(fable P0, speed_gate=True 시): 기록 전·후반 rpm이 흔들리면(램프 중 캡처)
    # 차수/포락/공진 등 rpm-추적 특징은 신뢰불가 → 세션 끝에서 NaN(오탐 방지). speed_drift도 이상 지표.
    speed_drift = 0.0
    stable = True
    if speed_gate and fs > 0 and rpm > 0 and chans:
        ref = chans.get("ch3") or chans.get("ch1") or next(iter(chans.values()))
        h = len(ref) // 2

        def _half_rpm(x: np.ndarray) -> float:
            if len(x) < 16:
                return 0.0
            m = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x) * 2.0
            return detect_rpm(np.fft.rfftfreq(len(x), 1.0 / fs), m, rpm_range)
        r1, r2 = _half_rpm(ref[:h]), _half_rpm(ref[h:2 * h])
        if r1 > 0 and r2 > 0:
            speed_drift = abs(r1 - r2) / 60.0
        stable = speed_drift <= 0.1
    row: dict[str, float] = {"stamp": session.stamp, "rpm": rpm, "fs": fs}
    if speed_gate:
        row["speed_drift"] = speed_drift
    for ch in keys:
        for name, val in time_features(chans[ch]).items():
            row[f"{ch}_{name}"] = val
        if fs > 0:                                   # 대역첨도(임펄스성): 구조공진 반송대역 2–7kHz
            row[f"{ch}_bandkurt"] = band_kurtosis(chans[ch], fs, (2000.0, 7000.0))
        idx = int(ch.replace("ch", "")) if ch.startswith("ch") else None
        if idx is not None and idx in session.fft_paths:
            fr, mg = dataio.load_fft(session.fft_paths[idx])
            if _chscale(ch) != 1.0:
                mg = mg * _chscale(ch)              # 스펙트럼 크기도 동일 배율(harm/band_energy 정합)
            if _pn is not None:
                mg = _mask_comb(fr, mg, *_pn)       # band_energy 등에서 PWM 콤 제외
            for name, val in spectrum_features(fr, mg, rpm).items():
                row[f"{ch}_{name}"] = val
            for name, val in order_features(fr, mg, rpm).items():   # 차수 무차원 비율(정렬·느슨·BPF)
                row[f"{ch}_{name}"] = val
            row[f"{ch}_spec_flatness"] = spectral_flatness(fr, mg, (2000.0, 15000.0))  # 톤 vs 광대역
            row[f"{ch}_hf_rms"] = band_rms_spectral(fr, mg, (9000.0, 15000.0))         # HF 유압/캐비 대역
            if resonances and ch in resonances:      # 구조 공진 주파수 추적(손상→하향이동)
                for name, val in resonance_features(fr, mg, resonances[ch]).items():
                    row[f"{ch}_{name}"] = val
    # 베어링 포락선 특징(설정된 채널만): raw 파형에서 결함주파수 대역 진폭 추출.
    if bearings and rpm > 0 and fs > 0:
        shaft_fr = rpm / 60.0
        for ch, spec in bearings.items():
            if ch not in chans:
                continue
            n, d, D, theta = _resolve_bearing(spec)
            dfreqs = bearing_defect_freqs(shaft_fr, n, d, D, theta)
            for name, val in envelope_features(chans[ch], fs, dfreqs, band=env_band,
                                               shaft_fr=shaft_fr).items():   # 정수차수 배제
                row[f"{ch}_{name}"] = val
    if not stable:                       # 속도 불안정 → rpm-추적 특징 무효화(NaN, 오탐 방지)
        for k in list(row):
            sub = k.split("_", 1)[1] if "_" in k and k.startswith("ch") else k
            if sub.startswith(("harm_", "ord_", "env_", "res")):
                row[k] = float("nan")
    return row
