import numpy as np
import pytest

torch = pytest.importorskip("torch")   # 가짜 모드형을 torch 함수로 만든다 — torch 없는 환경에서는 건너뛴다

from impeller_pinn import modal


def _fake_mode(n, f):
    # shape(x̃)=상수(간단): φ(x_s)=φ(x_i)=1 → amp은 가속도가중 ω에 비례.
    return {"n": n, "f_pinn": float(f), "shape": lambda x: torch.ones_like(x)}


def test_modal_impact_response_contains_mode_freqs():
    # 모달중첩 링다운이 각 모드 주파수 성분을 스펙트럼에 담아야 함(감쇠사인 합).
    fs, dur = 51200.0, 0.2
    modes = [_fake_mode(1, 900.0), _fake_mode(2, 6400.0)]
    sig = modal.modal_impact_response(modes, fs, dur, x_sensor=0.6, x_impact=0.9,
                                      zeta=0.01, t0=0.0)
    assert len(sig) == int(round(fs * dur))
    mag = np.abs(np.fft.rfft(sig))
    frq = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    # 각 모드 주파수 ±50Hz 창의 피크가 전역 잡음보다 뚜렷
    for f in (900.0, 6400.0):
        band = (frq > f - 50) & (frq < f + 50)
        assert mag[band].max() > 5 * np.median(mag)


def test_modal_accel_weighting_favors_high_mode():
    # 가속도계 가중(amp∝ω)은 고차 모드 성분을 저차보다 강조해야 함(충격 고주파 특성).
    fs, dur = 51200.0, 0.2
    modes = [_fake_mode(1, 1000.0), _fake_mode(2, 10000.0)]
    sig = modal.modal_impact_response(modes, fs, dur, zeta=0.005, accel=True)
    mag = np.abs(np.fft.rfft(sig)); frq = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    p_lo = mag[(frq > 950) & (frq < 1050)].max()
    p_hi = mag[(frq > 9950) & (frq < 10050)].max()
    assert p_hi > p_lo


def test_impact_train_periodic_and_impulsive():
    # 충격열: 주기 간격으로 배치되고, 스파스 임펄스라 kurtosis가 가우시안(3)보다 크게 뜸.
    fs, dur, period = 51200.0, 1.0, 1.0 / 58.0
    single = modal.modal_impact_response([_fake_mode(1, 3000.0)], fs, 0.02, zeta=0.05)
    train = modal.impact_train(single, fs, period, dur)
    assert len(train) == int(round(fs * dur))
    # 포락선(|해석신호| 근사=abs) 스펙트럼에 회전주파수 성분
    env = np.abs(train); env -= env.mean()
    mag = np.abs(np.fft.rfft(env)); frq = np.fft.rfftfreq(len(env), d=1.0 / fs)
    band = (frq > 58 - 5) & (frq < 58 + 5)
    assert mag[band].max() > 5 * np.median(mag)               # f_rot 주기성
    from scipy.stats import kurtosis
    assert kurtosis(train, fisher=False) > 4.0                # 충격성(가우시안 3 초과)


def test_scale_to_rms():
    sig = np.random.default_rng(0).normal(0, 3.0, 10000)
    out = modal.scale_to_rms(sig, 1.5)
    assert np.sqrt(np.mean(out ** 2)) == __import__("pytest").approx(1.5, rel=1e-6)
