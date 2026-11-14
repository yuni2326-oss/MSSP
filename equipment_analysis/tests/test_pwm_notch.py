"""VFD PWM 콤 제거 필터 테스트."""
import numpy as np
import pandas as pd

from equipment_analysis import features, dataio


def test_notch_removes_pwm_comb():
    # 100Hz 신호 + 강한 8k 캐리어 + 16k 2차 하모닉 → notch 후 콤 제거·기저 톤 보존.
    fs, n = 51200.0, 20000
    t = np.arange(n) / fs
    base = 1.0 * np.sin(2 * np.pi * 100 * t)
    comb = 3.0 * np.sin(2 * np.pi * 8000 * t) + 1.5 * np.sin(2 * np.pi * 16000 * t)
    sig = base + comb
    out = features.notch_pwm_comb(sig, fs, carrier=8000.0, n_harm=2, width=400.0)
    rms_base = np.sqrt(np.mean(base ** 2))
    assert np.sqrt(np.mean(sig ** 2)) > 2.0                       # 콤이 지배
    assert abs(np.sqrt(np.mean(out ** 2)) - rms_base) / rms_base < 0.15   # 콤 제거→기저만
    mag = np.abs(np.fft.rfft(out * np.hanning(n))) / n * 2
    frq = np.fft.rfftfreq(n, 1 / fs)
    pk = lambda f0: float(mag[(frq > f0 - 300) & (frq < f0 + 300)].max())
    assert pk(8000) < 0.05 and pk(16000) < 0.05                  # 콤 소멸
    assert pk(100) > 0.3                                          # 기저 톤 보존


def test_notch_skips_above_nyquist():
    # 나이퀴스트 초과 하모닉은 건너뛰고 에러 없이 동작.
    fs, n = 51200.0, 8000
    sig = np.random.default_rng(0).normal(0, 1, n)
    out = features.notch_pwm_comb(sig, fs, carrier=8000.0, n_harm=5, width=400.0)  # 4·8k=32k>Nyq
    assert out.shape == sig.shape and np.all(np.isfinite(out))


def test_session_features_pwm_notch(tmp_path):
    # session_features(pwm_notch=...) 가 콤 오염 채널의 rms를 크게 낮추고, None이면 불변.
    fs, n = 51200.0, 15000
    t = np.arange(n) / fs
    ch = 1.0 * np.sin(2 * np.pi * 100 * t) + 3.0 * np.sin(2 * np.pi * 8000 * t)
    st = "20260715_120000"
    pd.DataFrame({"time_s": t, **{f"ch{i}": ch for i in range(6)}}).to_csv(
        tmp_path / f"{st}_raw.csv", index=False)
    sess = {s.stamp: s for s in dataio.discover_sessions(str(tmp_path))}[st]
    none = features.session_features(sess)
    notched = features.session_features(
        sess, pwm_notch={"carrier": 8000.0, "n_harm": 1, "width": 400.0})
    assert notched["ch0_rms"] < 0.6 * none["ch0_rms"]            # 캐리어 제거→rms 하강
    assert none["ch0_rms"] > 2.0                                 # None=기존 동작(콤 포함)
