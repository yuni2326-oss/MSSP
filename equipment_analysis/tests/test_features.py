# tests/test_features.py
import numpy as np
import pytest
from equipment_analysis import features, dataio

def test_time_features_sine():
    fs = 51200; t = np.arange(0, 0.1, 1/fs)
    sig = 2.0 * np.sin(2*np.pi*58.2*t)        # 진폭 2.0
    f = features.time_features(sig)
    assert f["rms"] == pytest.approx(2.0/np.sqrt(2), rel=0.02)   # A/√2
    assert f["peak"] == pytest.approx(2.0, rel=0.02)
    assert f["crest"] == pytest.approx(np.sqrt(2), rel=0.03)     # 정현파 crest=√2
    assert f["kurtosis"] == pytest.approx(-1.5, abs=0.1)         # 정현파 excess kurtosis (Step 1b 보정)

def test_time_features_gaussian_kurtosis():
    rng = np.random.default_rng(0)
    f = features.time_features(rng.normal(0, 1, 50000))
    assert abs(f["kurtosis"]) < 0.2          # 정규분포 excess kurtosis≈0

def test_time_features_impulse_high_kurtosis():
    sig = np.zeros(10000); sig[::500] = 10.0  # 임펄스 → 첨도 급증
    f = features.time_features(sig)
    assert f["kurtosis"] > 5.0

def test_detect_rpm():
    freq = np.arange(0, 512, 0.2)
    mag = np.full_like(freq, 0.001); mag[np.argmin(abs(freq-58.2))] = 0.2
    rpm = features.detect_rpm(freq, mag)
    assert rpm == pytest.approx(58.2*60, abs=15)   # ≈3492

def test_session_features(synth_data_root):
    sessions = dataio.discover_sessions(synth_data_root)
    row = features.session_features(sessions[0])
    assert row["stamp"] == "20260616_050000"
    assert "ch0_rms" in row and "ch0_kurtosis" in row
    assert row["rpm"] == pytest.approx(3492, abs=60)


def test_bearing_defect_freqs_6204():
    # 6204(n=8, d=7.94mm, D=33.5mm, θ=0) 공개 카탈로그 배수와 대조. fr=1Hz → 결과=배수.
    f = features.bearing_defect_freqs(fr=1.0, n=8, d=7.94, D=33.5)
    assert f["bpfo"] == pytest.approx(3.052, abs=0.01)
    assert f["bpfi"] == pytest.approx(4.948, abs=0.01)
    assert f["bsf"] == pytest.approx(1.991, abs=0.01)
    assert f["ftf"] == pytest.approx(0.3815, abs=0.01)
    # 물리 불변식: BPFO + BPFI = n·fr, 그리고 결함주파수는 fr에 선형 비례.
    assert f["bpfo"] + f["bpfi"] == pytest.approx(8.0, abs=1e-6)
    g = features.bearing_defect_freqs(fr=57.5, n=8, d=7.94, D=33.5)
    assert g["bpfo"] == pytest.approx(f["bpfo"] * 57.5, rel=1e-9)


def test_envelope_spectrum_recovers_modulation():
    # 고주파 반송파(6kHz, 공진대역)를 저주파(100Hz)로 진폭변조 → 포락선 FFT가 100Hz를 복원.
    fs = 51200
    t = np.arange(0, 0.5, 1 / fs)
    fc, fm = 6000.0, 100.0
    am = (1.0 + 0.8 * np.cos(2 * np.pi * fm * t)) * np.cos(2 * np.pi * fc * t)
    freq, spec = features.envelope_spectrum(am, fs, band=(2000.0, 10000.0))
    valid = freq > 10.0                       # DC 근처 제외
    peak_hz = float(freq[valid][np.argmax(spec[valid])])
    assert peak_hz == pytest.approx(fm, abs=3.0)
    # 반송파 성분(6kHz)은 복조로 제거되어 포락선 스펙트럼에 거의 없어야 함.
    carrier_amp = float(spec[np.argmin(np.abs(freq - fc))])
    assert carrier_amp < spec[valid].max() * 0.1


def _bearing_impact_signal(fs, dur, defect_hz, fc=6000.0, decay=900.0, amp=1.0, noise=0.05, seed=0):
    """defect_hz 간격 충격열이 공진(fc)을 여기하는 감쇠 사인으로 나타나는 신호 + 잡음."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    x = rng.normal(0.0, noise, n)
    period = 1.0 / defect_hz
    tk = 0.0
    while tk < dur:
        i0 = int(tk * fs)
        tau = t[i0:] - tk
        x[i0:] += amp * np.exp(-decay * tau) * np.sin(2 * np.pi * fc * tau)
        tk += period
    return x


def test_envelope_features_selective_to_bpfo():
    fs, dur, fr = 51200, 0.5, 30.0
    dfreqs = features.bearing_defect_freqs(fr, n=8, d=7.94, D=33.5)  # 6204
    band = (2000.0, 10000.0)
    sig = _bearing_impact_signal(fs, dur, dfreqs["bpfo"])            # 외륜 결함 주입
    feats = features.envelope_features(sig, fs, dfreqs, band=band)   # SNR(peak/floor) 의미
    # 외륜 결함 → env_bpfo SNR이 4개 결함특징 중 최대이고 다른 것들보다 뚜렷이 큼.
    assert feats["env_bpfo"] == max(feats.values())
    assert feats["env_bpfo"] > 3 * feats["env_bpfi"]
    assert feats["env_bpfo"] > 1.5 * feats["env_bsf"]   # BSF k=3(179Hz)이 2×BPFO(183) 누출↑(짧은 기록 교차오염); bpfo 여전히 우세
    # 정상(잡음만)의 env_bpfo SNR은 낮고(≈수), 결함 SNR이 크게 상승.
    rng = np.random.default_rng(1)
    baseline = features.envelope_features(rng.normal(0, 0.05, int(fs * dur)), fs, dfreqs, band=band)
    assert feats["env_bpfo"] > 3 * baseline["env_bpfo"]


def test_session_features_calibration_scales_pre_cutoff(tmp_path):
    from equipment_analysis.tests.conftest import _write_raw, _write_fft
    d = tmp_path / "pump data"; d.mkdir()
    pre, post = "20260101_000000", "20260710_000000"     # 컷오프 전/후, 동일 raw
    for st in (pre, post):
        _write_raw(d / f"{st}_raw.csv", amp1x=1.0)
        for ch in range(6):
            _write_fft(d / f"{st}_fft_ch{ch}.csv")
    cal = {"before": "20260707_112346", "scale": 10.0}
    sessions = {s.stamp: s for s in dataio.discover_sessions(str(d))}
    a = features.session_features(sessions[pre], calibration=cal)    # 이전 → ×10
    b = features.session_features(sessions[post], calibration=cal)   # 이후 → 그대로
    # 진폭선형 특징은 정확히 scale배, 스케일불변량(crest/kurtosis)은 동일.
    assert a["ch0_rms"] == pytest.approx(10.0 * b["ch0_rms"], rel=1e-6)
    assert a["ch0_peak"] == pytest.approx(10.0 * b["ch0_peak"], rel=1e-6)
    assert a["ch0_harm_1x"] == pytest.approx(10.0 * b["ch0_harm_1x"], rel=1e-6)
    assert a["ch0_crest"] == pytest.approx(b["ch0_crest"], rel=1e-6)
    assert a["ch0_kurtosis"] == pytest.approx(b["ch0_kurtosis"], rel=1e-6)
    # 컷오프 이후 세션은 미적용(=raw 그대로). calibration 없으면 pre도 동일.
    c = features.session_features(sessions[pre])                     # calibration 미지정
    assert c["ch0_rms"] == pytest.approx(b["ch0_rms"], rel=1e-6)


def test_session_features_calibration_per_channel(tmp_path):
    from equipment_analysis.tests.conftest import _write_raw, _write_fft
    d = tmp_path / "pump data"; d.mkdir()
    pre, post = "20260101_000000", "20260710_000000"
    for st in (pre, post):
        _write_raw(d / f"{st}_raw.csv", amp1x=1.0)
        for ch in range(6):
            _write_fft(d / f"{st}_fft_ch{ch}.csv")
    # 채널별 계수: ch0=g→m/s²(9.80665), ch4=V→m/s²(98.04). ch1은 미지정→그대로.
    cal = {"before": "20260707_112346", "scale": {"ch0": 9.80665, "ch4": 98.04}}
    sess = {s.stamp: s for s in dataio.discover_sessions(str(d))}
    a = features.session_features(sess[pre], calibration=cal)     # 이전 세션 → 채널별 스케일
    b = features.session_features(sess[post], calibration=cal)    # 이후 세션 → 그대로
    assert a["ch0_rms"] == pytest.approx(9.80665 * b["ch0_rms"], rel=1e-6)
    assert a["ch4_rms"] == pytest.approx(98.04 * b["ch4_rms"], rel=1e-6)
    assert a["ch1_rms"] == pytest.approx(b["ch1_rms"], rel=1e-6)   # 미지정 채널은 1.0
    assert a["ch0_harm_1x"] == pytest.approx(9.80665 * b["ch0_harm_1x"], rel=1e-6)


def test_session_features_envelope_wiring(tmp_path):
    import pandas as pd
    from equipment_analysis.tests.conftest import _write_fft
    fs, dur, fr = 51200, 2.0, 30.0                                # 2s: 포락 분해능 df=0.5Hz로 BPFO를 3×서 분리
    n = int(fs * dur)
    t = np.arange(n) / fs
    dfreqs = features.bearing_defect_freqs(fr, 8, 7.94, 33.5)     # 6204
    imp = _bearing_impact_signal(fs, dur, dfreqs["bpfo"], seed=3)  # ch2에 외륜 결함
    d = tmp_path / "pump data"; d.mkdir()
    stamp = "20260101_000000"
    cols = {"time_s": t}
    for c in range(6):
        cols[f"ch{c}"] = imp if c == 2 else np.random.default_rng(c).normal(0, 0.05, n)
    pd.DataFrame(cols).to_csv(d / f"{stamp}_raw.csv", index=False)
    for c in range(6):
        _write_fft(d / f"{stamp}_fft_ch{c}.csv", peak_hz=30.0)     # 회전수 30Hz→1800rpm
    sess = dataio.discover_sessions(str(d))[0]

    # bearings 미지정 → env 컬럼 없음(하위호환).
    plain = features.session_features(sess, rpm_range=(20.0, 120.0))
    assert plain["rpm"] == pytest.approx(1800, abs=60)
    assert not any(k.endswith("_env_bpfo") for k in plain)

    # bearings 지정 → 해당 채널에만 env 컬럼, 값은 직접 호출과 일치.
    row = features.session_features(sess, rpm_range=(20.0, 120.0),
                                    bearings={"ch2": {"model": "6204"}})
    assert "ch2_env_bpfo" in row and "ch0_env_bpfo" not in row
    sf = row["rpm"] / 60.0                                        # session이 쓴 검출 shaft_fr
    dfreqs_det = features.bearing_defect_freqs(sf, 8, 7.94, 33.5)  # 동일 결함주파수·shaft_fr
    expected = features.envelope_features(imp, fs, dfreqs_det, shaft_fr=sf)
    assert row["ch2_env_bpfo"] == pytest.approx(expected["env_bpfo"], rel=1e-9)
    assert row["ch2_env_bpfo"] > 3 * row["ch2_env_bpfi"]           # 외륜 결함 선택성 유지


def _write_two_peak_fft(path, peaks):
    """peaks={hz: mag}로 FFT CSV 생성(나머지 0.001)."""
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["frequency_hz", "magnitude"])
        for k in range(2560):
            fr = k * 0.2
            mag = 0.001
            for hz, m in peaks.items():
                if abs(fr - hz) < 0.2:
                    mag = m
            w.writerow([f"{fr:.4f}", f"{mag:.6f}"])


def _two_peak_session_dir(tmp_path):
    """30Hz(작은 피크)+100Hz(큰 피크) 세션 1개 디렉토리."""
    from equipment_analysis.tests.conftest import _write_raw
    d = tmp_path / "pump data"; d.mkdir()
    stamp = "20260101_000000"
    _write_raw(d / f"{stamp}_raw.csv")
    for ch in range(6):
        _write_two_peak_fft(d / f"{stamp}_fft_ch{ch}.csv", {30.0: 0.3, 100.0: 0.5})
    return d


def test_session_features_honors_rpm_range(tmp_path):
    # 넓은 범위 → 최대피크 100Hz 검출(6000 RPM), 좁은 범위 → 100Hz 제외돼 30Hz(1800 RPM).
    d = _two_peak_session_dir(tmp_path)
    sess = dataio.discover_sessions(str(d))[0]
    wide = features.session_features(sess, rpm_range=(20.0, 120.0))
    narrow = features.session_features(sess, rpm_range=(20.0, 50.0))
    assert wide["rpm"] == pytest.approx(6000, abs=60)     # 100Hz 피크
    assert narrow["rpm"] == pytest.approx(1800, abs=60)   # 100Hz 제외 → 30Hz


def test_build_feature_table_honors_rpm_range(tmp_path):
    from equipment_analysis import cache
    d = _two_peak_session_dir(tmp_path)
    wide = cache.build_feature_table(str(d), tmp_path / "cw.csv", rpm_range=(20.0, 120.0))
    narrow = cache.build_feature_table(str(d), tmp_path / "cn.csv", rpm_range=(20.0, 50.0))
    assert wide.iloc[0]["rpm"] == pytest.approx(6000, abs=60)
    assert narrow.iloc[0]["rpm"] == pytest.approx(1800, abs=60)
