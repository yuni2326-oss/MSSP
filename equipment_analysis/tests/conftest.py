import csv, math, random
from pathlib import Path
import pytest

FS = 51200          # 샘플레이트 Hz
N_SAMPLES = 5120    # 테스트는 0.1초(5120샘플)로 축소 (속도)
DF = 0.2            # FFT 주파수 스텝
N_BINS = 2560       # 0~512Hz 까지만 (테스트 축소)

def _write_raw(path: Path, amp1x=1.0, noise=0.01, kurtosis_spikes=0):
    """time_s,ch0..ch5 raw CSV 생성. ch0=58.2Hz 정현파, ch4/ch5=저레벨."""
    rng = random.Random(42)
    f1x = 58.2
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "ch0", "ch1", "ch2", "ch3", "ch4", "ch5"])
        for i in range(N_SAMPLES):
            t = i / FS
            base = amp1x * math.sin(2 * math.pi * f1x * t)
            spike = 0.0
            if kurtosis_spikes and i % (N_SAMPLES // kurtosis_spikes) == 0:
                spike = 8.0  # 임펄스 → 첨도 상승
            ch0 = base + spike + rng.gauss(0, noise)
            row = [f"{t:.8f}",
                   f"{ch0:.6f}",
                   f"{0.5*base + rng.gauss(0, noise):.6f}",
                   f"{0.3*base + rng.gauss(0, noise):.6f}",
                   f"{0.2*base + rng.gauss(0, noise):.6f}",
                   f"{rng.gauss(0, 0.0005):.6f}",   # ch4 저레벨
                   f"{rng.gauss(0, 0.0005):.6f}"]   # ch5 저레벨
            w.writerow(row)

def _write_fft(path: Path, peak_hz=58.2, peak_mag=0.2):
    """frequency_hz,magnitude FFT CSV 생성. peak_hz에 피크."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "magnitude"])
        for k in range(N_BINS):
            fr = k * DF
            mag = 0.001
            if abs(fr - peak_hz) < DF:        # 1x
                mag = peak_mag
            elif abs(fr - 2 * peak_hz) < DF:  # 2x
                mag = peak_mag * 0.3
            w.writerow([f"{fr:.4f}", f"{mag:.6f}"])

@pytest.fixture
def synth_data_root(tmp_path):
    """3개 세션(정상2 + 첨도이상1) 합성 디렉토리 반환."""
    root = tmp_path / "pump data"
    root.mkdir()
    stamps = ["20260616_050000", "20260616_053000", "20260616_060000"]
    for idx, stamp in enumerate(stamps):
        spikes = 0 if idx < 2 else 20  # 3번째 세션에 임펄스 → 첨도 이상
        _write_raw(root / f"{stamp}_raw.csv", kurtosis_spikes=spikes)
        for ch in range(6):
            _write_fft(root / f"{stamp}_fft_ch{ch}.csv")
    return root
