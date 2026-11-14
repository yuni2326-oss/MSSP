import pytest
from equipment_analysis import train_model, model, dataio
from equipment_analysis.tests.conftest import _write_raw, _write_fft


def _make_dir(base, n=13, dup=2):
    """n개 세션(앞 dup개는 amp 동일=수집에러 중복, 나머지는 distinct)."""
    d = base / "pump data"; d.mkdir(parents=True)
    for i in range(n):
        st = f"202601{i + 1:02d}_000000"
        amp = 1.0 if i < dup else 1.0 + 0.13 * i     # 앞 dup개 동일 → 중복 그룹
        _write_raw(d / f"{st}_raw.csv", amp1x=amp)
        for ch in range(6):
            _write_fft(d / f"{st}_fft_ch{ch}.csv")
    return d


def _reg(base):
    return {"base": str(base), "defaults": {},
            "folders": {"pump data": {"equipment_type": "pump", "rpm_lo": 20, "rpm_hi": 120}}}


def test_train_folder_dedups_and_saves(tmp_path):
    _make_dir(tmp_path / "data")
    res = train_model.train_folder("pump data", _reg(tmp_path / "data"),
                                   cache_path=tmp_path / "f.csv", model_path=tmp_path / "m.joblib")
    assert res["n_duplicates"] == 2               # 앞 2세션 동일
    assert res["n_train"] == 11                   # 13 - 2 중복
    assert (tmp_path / "m.joblib").exists()
    art = model.load_artifact(tmp_path / "m.joblib")   # 저장된 모델 로드 가능
    assert art["n_train"] == 11


def test_train_folder_no_dedup_includes_all(tmp_path):
    _make_dir(tmp_path / "data")
    res = train_model.train_folder("pump data", _reg(tmp_path / "data"), dedup=False,
                                   cache_path=tmp_path / "f.csv", model_path=tmp_path / "m.joblib")
    assert res["n_duplicates"] == 0
    assert res["n_train"] == 13


def test_quarantine_sessions_moves_and_hides(tmp_path):
    d = _make_dir(tmp_path / "data")
    stamps = ["20260101_000000", "20260102_000000"]
    moved = train_model.quarantine_sessions(d, stamps, tmp_path / "quar")
    assert moved == 2 * 7                          # 세션당 raw + fft6
    remaining = {s.stamp for s in dataio.discover_sessions(str(d))}
    assert stamps[0] not in remaining and stamps[1] not in remaining
    assert len(list((tmp_path / "quar").glob("*.bak"))) == 14
