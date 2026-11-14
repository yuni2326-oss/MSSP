import numpy as np, pytest
from impeller_pinn.synth import impact_ringdown, shift_freqs

def test_ringdown_has_peaks_at_modal_freqs():
    fs, dur = 51200, 0.2
    sig = impact_ringdown(fs, dur, freqs=[1000.0, 3000.0], zetas=[0.01, 0.01],
                          amps=[1.0, 0.5], t0=0.0, noise=0.0)
    f = np.fft.rfftfreq(len(sig), 1/fs); X = np.abs(np.fft.rfft(sig))
    peak = f[np.argmax(X)]
    assert peak == pytest.approx(1000.0, abs=30)          # 최대 피크 = 1kHz 모드
    assert X[np.argmin(np.abs(f-3000))] > 10*np.median(X)  # 3kHz도 뚜렷

def test_ringdown_decays_and_is_localized():
    fs, dur = 51200, 0.5
    sig = impact_ringdown(fs, dur, [2000.0], [0.02], [1.0], t0=0.0)
    n = len(sig)
    assert np.std(sig[:n//10]) > 5 * np.std(sig[-n//10:])  # 앞>뒤(감쇠)

def test_shift_freqs():
    assert shift_freqs([1000.0, 2000.0], 0.9) == [900.0, 1800.0]
