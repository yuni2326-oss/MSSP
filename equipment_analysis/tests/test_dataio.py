# tests/test_dataio.py
import numpy as np
import pytest
from pathlib import Path
from equipment_analysis import dataio

def test_discover_sessions(synth_data_root):
    sessions = dataio.discover_sessions(synth_data_root)
    assert len(sessions) == 3
    assert sessions[0].stamp == "20260616_050000"
    assert sessions[0].raw_path.name == "20260616_050000_raw.csv"
    assert set(sessions[0].fft_paths.keys()) == {0, 1, 2, 3, 4, 5}

def test_discover_sessions_date_filter(synth_data_root):
    s = dataio.discover_sessions(synth_data_root, date_from="20260616_053000")
    assert [x.stamp for x in s] == ["20260616_053000", "20260616_060000"]

def test_safe_resolve_rejects_traversal(synth_data_root):
    with pytest.raises(ValueError):
        dataio.safe_resolve(synth_data_root, "../../../etc/passwd")

def test_safe_resolve_ok(synth_data_root):
    p = dataio.safe_resolve(synth_data_root, "20260616_050000_raw.csv")
    assert p.exists()

def test_load_raw(synth_data_root):
    raw = synth_data_root / "20260616_050000_raw.csv"
    t, ch, fs = dataio.load_raw(raw)
    assert len(t) == 5120
    assert set(ch.keys()) == {"ch0", "ch1", "ch2", "ch3", "ch4", "ch5"}
    assert abs(fs - 51200) < 50          # 샘플레이트 검출
    assert ch["ch0"].std() > ch["ch4"].std()  # ch4 저레벨

def test_load_fft(synth_data_root):
    fft = synth_data_root / "20260616_050000_fft_ch0.csv"
    freq, mag = dataio.load_fft(fft)
    assert freq[1] - freq[0] == pytest.approx(0.2, abs=1e-6)
    peak_freq = freq[int(np.argmax(mag))]
    assert abs(peak_freq - 58.2) < 0.3   # 1x 피크

def test_list_data_folders_finds_session_dirs(tmp_path):
    from equipment_analysis import dataio
    base = tmp_path
    (base / "GY_DATA" / "pump data").mkdir(parents=True)
    (base / "GY_DATA" / "pump data" / "20260616_050000_raw.csv").write_text("time_s,ch0\n0,0\n")
    (base / "empty").mkdir()
    (base / "empty" / "notes.txt").write_text("x")
    assert dataio.list_data_folders(base) == ["GY_DATA/pump data"]
