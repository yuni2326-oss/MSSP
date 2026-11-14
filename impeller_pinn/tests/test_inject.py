import numpy as np, pytest
from pathlib import Path
from impeller_pinn.inject import make_injected_session
from impeller_pinn.synth import impact_ringdown
from equipment_analysis import dataio, features
from equipment_analysis.tests.conftest import _write_raw, _write_fft

def _src(tmp_path):
    d = tmp_path/"src"; d.mkdir()
    st = "20260707_120000"
    _write_raw(d/f"{st}_raw.csv")                 # conftest: fs=51200, N=5120, ch0..ch5
    for c in range(6): _write_fft(d/f"{st}_fft_ch{c}.csv")
    return d, st

def test_inject_adds_only_target_channel(tmp_path):
    src, st = _src(tmp_path)
    raw0 = dataio.read_csv_fallback(src/f"{st}_raw.csv")
    n = len(raw0)
    ring = impact_ringdown(51200, n/51200, [2000.0], [0.02], [3.0], t0=0.0)
    out = make_injected_session(src, st, tmp_path/"out", "ch1", ring, "20260707_120001")
    inj = dataio.read_csv_fallback(out)
    assert np.allclose(inj["ch1"].to_numpy(float), raw0["ch1"].to_numpy(float) + ring)
    assert np.allclose(inj["ch0"].to_numpy(float), raw0["ch0"].to_numpy(float))  # 타채널 불변
    # 주입 세션이 session_features로 처리됨(fft 복사로 rpm 검출 유지)
    sess = {s.stamp: s for s in dataio.discover_sessions(str(tmp_path/"out"))}
    assert "20260707_120001" in sess
    row = features.session_features(sess["20260707_120001"])
    assert "ch1_kurtosis" in row and "ch1_rms" in row

def test_inject_length_mismatch_raises(tmp_path):
    src, st = _src(tmp_path)
    with pytest.raises(ValueError):
        make_injected_session(src, st, tmp_path/"o", "ch1", np.zeros(10), "20260707_120002")


def test_inject_recomputes_target_fft_spectrum(tmp_path):
    # 주입채널 fft가 주입 raw로 재계산되어 주입 주파수 피크가 생기고, 타채널 fft는 복사(동일).
    import filecmp
    src, st = _src(tmp_path)
    raw0 = dataio.read_csv_fallback(src/f"{st}_raw.csv"); n = len(raw0)
    fs, fpk = 51200.0, 3000.0
    ring = impact_ringdown(fs, n/fs, [fpk], [0.005], [10.0], t0=0.0)
    make_injected_session(src, st, tmp_path/"out", "ch1", ring, "20260707_120003")
    fr, mg = dataio.load_fft(tmp_path/"out"/"20260707_120003_fft_ch1.csv")
    band = (fr > fpk - 150) & (fr < fpk + 150)
    assert mg[band].max() > 5 * np.median(mg)                       # 주입 주파수 피크 출현
    assert filecmp.cmp(src/f"{st}_fft_ch0.csv",
                       tmp_path/"out"/"20260707_120003_fft_ch0.csv")  # 타채널 복사 불변
