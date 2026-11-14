"""결함탐지 확장(fable P0/P1): rpm comb 추정·차수비율·스펙트럼형상·대역첨도."""
import numpy as np
import pandas as pd

from equipment_analysis import features, dataio


def _spec(peaks, fmax=25600.0, df=0.5, floor=1e-4):
    """peaks={freq:amp} → (freq, mag) 스펙트럼."""
    freq = np.arange(0, fmax, df)
    mag = np.full_like(freq, floor)
    for f, a in peaks.items():
        mag[int(np.argmin(np.abs(freq - f)))] = a
    return freq, mag


def test_detect_rpm_harmonic_comb_beats_strong_2x():
    # 2x(80Hz)가 최강 단일선이라 argmax는 80에 락. 그러나 완전한 하모닉 시리즈(40·80·120·160·200)를
    # 가진 진짜 기본파 40이 comb 점수를 이긴다(1x candidate가 전 하모닉을 합산).
    freq, mag = _spec({40: 0.2, 80: 0.4, 120: 0.15, 160: 0.2, 200: 0.1})
    rpm = features.detect_rpm(freq, mag, search_hz=(20, 120))
    assert abs(rpm - 40.0 * 60) <= 15  # comb가 40 선택(80 아님)


def test_detect_rpm_single_tone_still_works():
    freq, mag = _spec({58.2: 0.2})
    assert features.detect_rpm(freq, mag) == np.float64(58.2 * 60).round(0) or \
        abs(features.detect_rpm(freq, mag) - 58.2 * 60) <= 15


def test_order_features_ratios():
    # 1x=1.0, 2x=0.6, 0.5x=0.2, 6x=0.4, 3x=0.1 → 비율·BPF·합(6x·12x 제외) 검증.
    rpm = 60.0 * 60  # f1=60Hz
    freq, mag = _spec({60: 1.0, 120: 0.6, 30: 0.2, 180: 0.1, 360: 0.4})
    out = features.order_features(freq, mag, rpm)
    assert abs(out["ord_2x_ratio"] - 0.6) < 0.02
    assert abs(out["ord_0p5x_ratio"] - 0.2) < 0.02
    assert abs(out["ord_bpf_ratio"] - 0.4) < 0.02              # 6x=360Hz
    # ord_sum_ratio = (2x+3x)/1x = 0.7 (6x=360은 제외)
    assert abs(out["ord_sum_ratio"] - 0.7) < 0.03
    assert out["ord_1x"] == 1.0


def test_order_features_zero_rpm_empty():
    freq, mag = _spec({60: 1.0})
    assert features.order_features(freq, mag, 0.0) == {}


def test_spectral_flatness_mask_aware():
    # 마스크된(0) bin이 있어도 flatness가 0으로 붕괴하지 않음. 정현(톤)=낮음, 백색=높음.
    freq = np.arange(0, 16000, 1.0)
    white = np.abs(np.random.default_rng(0).normal(1, 0.01, len(freq)))
    white[(freq > 7600) & (freq < 8400)] = 0.0             # PWM 마스크 모사
    fl_white = features.spectral_flatness(freq, white, (2000, 15000))
    tonal = np.full_like(freq, 1e-4); tonal[3000] = 10.0
    tonal[(freq > 7600) & (freq < 8400)] = 0.0
    fl_tonal = features.spectral_flatness(freq, tonal, (2000, 15000))
    assert fl_white > 0.5                                   # 백색=평탄(마스크에도 0 아님)
    assert fl_tonal < 0.1                                   # 톤=비평탄
    assert fl_white > 5 * fl_tonal


def test_band_rms_spectral():
    freq, mag = _spec({12000: 2.0}, df=1.0, floor=0.0)
    r = features.band_rms_spectral(freq, mag, (9000, 15000))
    assert abs(r - 2.0) < 1e-6                              # 대역 내 유일 피크
    assert features.band_rms_spectral(freq, mag, (16000, 20000)) < 1e-6


def test_band_kurtosis_impulsive():
    fs, n = 51200.0, 51200
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, n)                              # 가우시안(대역첨도≈0)
    imp = base.copy(); imp[::5000] += 30                    # 임펄스 주입(첨도↑)
    k_base = features.band_kurtosis(base, fs, (2000, 7000))
    k_imp = features.band_kurtosis(imp, fs, (2000, 7000))
    assert k_imp > k_base + 2


def test_session_features_new_faultdetect_wiring(tmp_path):
    # session_features가 신규 결함특징(차수비율·spec_flatness·hf_rms·bandkurt)을 생성.
    fs, n = 51200.0, 51200
    t = np.arange(n) / fs
    f1 = 58.0
    sig = np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * 2 * f1 * t)   # 1x+2x
    st = "20260716_130000"
    pd.DataFrame({"time_s": t, **{f"ch{i}": sig for i in range(3)}}).to_csv(
        tmp_path / f"{st}_raw.csv", index=False)
    fr = np.fft.rfftfreq(n, 1 / fs)
    mg = np.abs(np.fft.rfft(sig * np.hanning(n))) / n * 2
    pd.DataFrame({"freq": fr, "mag": mg}).to_csv(tmp_path / f"{st}_fft_ch0.csv", index=False)
    sess = {s.stamp: s for s in dataio.discover_sessions(str(tmp_path))}[st]
    row = features.session_features(sess)
    assert row["rpm"] > 0
    assert "ch0_ord_2x_ratio" in row and row["ch0_ord_2x_ratio"] > 0.3   # 2x 존재
    assert "ch0_spec_flatness" in row and "ch0_hf_rms" in row
    assert "ch0_bandkurt" in row
    assert "ch1_bandkurt" in row                                          # fft 없는 채널도 시간특징은 생성
