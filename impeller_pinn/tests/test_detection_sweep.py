import numpy as np
import pytest

from impeller_pinn.detection_sweep import damaged_mode_freqs, snr_scale, _band_rms

FS = 51200.0


def test_damaged_mode_freqs_band_limited_under_nyquist():
    freqs = damaged_mode_freqs(0.1, FS)
    assert all(f < 0.98 * FS / 2 for f in freqs)
    assert len(freqs) >= 2   # mode 1,2 should survive band-limiting at this severity/fs


def test_damaged_mode_freqs_drop_with_severity():
    f_low = damaged_mode_freqs(0.1, FS)[0]
    f_high = damaged_mode_freqs(0.6, FS)[0]
    assert f_high < f_low


def test_snr_scale_hits_target_db():
    rng = np.random.default_rng(0)
    n = FS  # 1 second
    t = np.arange(int(n)) / FS
    real = rng.normal(0.0, 0.4, int(n))
    freqs = [1000.0, 6000.0]
    ring = np.sin(2 * np.pi * freqs[0] * t) + 0.5 * np.sin(2 * np.pi * freqs[1] * t)
    for target_db in (-10.0, -5.0, 0.0, 5.0, 10.0):
        scaled = snr_scale(ring, real, FS, freqs, target_db)
        rms_real = _band_rms(real, FS, freqs)
        rms_ring = _band_rms(scaled, FS, freqs)
        achieved_db = 20 * np.log10(rms_ring / rms_real)
        assert achieved_db == pytest.approx(target_db, abs=0.5)


def test_snr_scale_monotonic_in_target_db():
    rng = np.random.default_rng(1)
    n = int(FS)
    t = np.arange(n) / FS
    real = rng.normal(0.0, 0.4, n)
    freqs = [1000.0]
    ring = np.sin(2 * np.pi * freqs[0] * t)
    scaled_lo = snr_scale(ring, real, FS, freqs, -10.0)
    scaled_hi = snr_scale(ring, real, FS, freqs, 10.0)
    assert np.abs(scaled_hi).max() > np.abs(scaled_lo).max()
