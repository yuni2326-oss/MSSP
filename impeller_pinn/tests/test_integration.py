import numpy as np, pytest
from impeller_pinn.beam_modes import cantilever_frequencies
from impeller_pinn.synth import impact_ringdown
from impeller_pinn.inject import make_injected_session
from equipment_analysis import dataio, features, anomaly
from equipment_analysis.tests.conftest import _write_raw, _write_fft

def _src(tmp_path):
    d = tmp_path/"src"; d.mkdir(); st="20260707_120000"
    _write_raw(d/f"{st}_raw.csv")
    for c in range(6): _write_fft(d/f"{st}_fft_ch{c}.csv")
    return d, st

def test_injected_fault_raises_impulsive_features(tmp_path):
    src, st = _src(tmp_path)
    n = len(dataio.read_csv_fallback(src/f"{st}_raw.csv"))
    fs = 51200.0
    freqs = cantilever_frequencies(0.030, 0.0012, 193e9, 8000)   # 베인 모드
    # 국소 충격(짧은 링다운) → 임펄스성 → ch1 kurtosis/crest 상승
    ring = impact_ringdown(fs, n/fs, freqs, [0.02]*len(freqs), [5.0,2.5,1.2,0.6], t0=0.0)
    make_injected_session(src, st, tmp_path/"out", "ch1", ring, "20260707_120001")
    healthy = features.session_features(dataio.discover_sessions(str(src))[0])
    damaged = features.session_features(dataio.discover_sessions(str(tmp_path/"out"))[0])
    assert damaged["ch1_kurtosis"] > healthy["ch1_kurtosis"] + 1.0   # 충격성 상승
    assert damaged["ch1_rms"] > healthy["ch1_rms"]                    # 에너지 상승
    assert damaged["ch0_rms"] == pytest.approx(healthy["ch0_rms"], rel=1e-9)  # 타채널 불변

def test_robust_z_flags_injected_series(tmp_path):
    # 정상 여러 세션 + 마지막에 주입(손상) → robust_z가 ch1_kurtosis 이탈로 표시
    src, st = _src(tmp_path)
    n = len(dataio.read_csv_fallback(src/f"{st}_raw.csv")); fs=51200.0
    vals = []
    for i in range(12):
        h = features.session_features(dataio.discover_sessions(str(src))[0])
        vals.append(h["ch1_kurtosis"])
    ring = impact_ringdown(fs, n/fs, [2000.0,6000.0], [0.02,0.02], [6.0,3.0], t0=0.0)
    make_injected_session(src, st, tmp_path/"out", "ch1", ring, "20260707_120099")
    d = features.session_features(dataio.discover_sessions(str(tmp_path/"out"))[0])
    series = np.array(vals + [d["ch1_kurtosis"]])
    mask = anomaly.flag_anomalies(series, baseline_n=10, sensitivity=3.0, rel_floor=0.05)
    assert mask[-1]           # 주입 세션 이탈 검출
    assert not mask[:10].any()
