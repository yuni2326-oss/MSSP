"""물리 합성 신호를 실측 세션 raw 채널에 주입 → 신규 세션 디렉토리(비파괴, 원본 불변).
주입 채널의 fft는 주입된 raw로 재계산(저장 규약 = Hann 윈도우·|rfft|/N*4; 실데이터 역설계로 확인,
corr=1.0)해 스펙트럼 특징이 주입을 반영. 나머지 채널 fft는 복사(rpm 검출·타채널 스펙트럼 유지)."""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from equipment_analysis import dataio


def _recompute_fft(signal: np.ndarray, fs: float) -> pd.DataFrame:
    """저장 fft 규약으로 진폭 스펙트럼 재계산: Hann 윈도우 후 |rfft|/N*4 (단측 진폭×윈도우 보정)."""
    n = len(signal)
    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(signal * win)) / n * 4.0
    frq = np.fft.rfftfreq(n, d=1.0 / fs) if fs > 0 else np.arange(len(mag), dtype=float)
    return pd.DataFrame({"frequency_hz": frq, "magnitude": mag})


def make_injected_session(src_dir, src_stamp, out_dir, channel, ringdown, out_stamp,
                          *, recompute_fft: bool = True) -> Path:
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = dataio.read_csv_fallback(src / f"{src_stamp}_raw.csv")
    if channel not in raw.columns:
        raise ValueError(f"채널 없음: {channel}")
    ringdown = np.asarray(ringdown, dtype=float)
    if len(ringdown) != len(raw):
        raise ValueError(f"링다운 길이 {len(ringdown)} != raw 길이 {len(raw)}")
    raw[channel] = raw[channel].to_numpy(dtype=float) + ringdown
    raw_out = out / f"{out_stamp}_raw.csv"
    raw.to_csv(raw_out, index=False)

    inj_idx = int(channel[2:]) if channel.startswith("ch") and channel[2:].isdigit() else None
    fs = 0.0
    if "time_s" in raw.columns and len(raw) > 1:
        t = raw["time_s"].to_numpy(dtype=float)
        dt = np.median(np.diff(t))
        fs = (1.0 / dt) if dt > 0 else 0.0
    for f in sorted(src.glob(f"{src_stamp}_fft_ch*.csv")):
        idx = int(f.name.split("_fft_ch")[1].split(".")[0])
        dst = out / f"{out_stamp}_fft_ch{idx}.csv"
        if recompute_fft and idx == inj_idx:
            _recompute_fft(raw[channel].to_numpy(dtype=float), fs).to_csv(dst, index=False)
        else:
            shutil.copy2(f, dst)
    return raw_out
