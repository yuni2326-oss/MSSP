# tests/test_cache.py
from pathlib import Path
from equipment_analysis import cache, dataio

def test_build_and_load(synth_data_root, tmp_path):
    cp = tmp_path / "features.csv"
    df = cache.build_feature_table(synth_data_root, cp)
    assert len(df) == 3
    assert "ch0_rms" in df.columns and "stamp" in df.columns
    assert cp.exists()
    df2 = cache.load_feature_table(cp)
    assert list(df2["stamp"]) == ["20260616_050000", "20260616_053000", "20260616_060000"]

def test_incremental(synth_data_root, tmp_path, monkeypatch):
    cp = tmp_path / "features.csv"
    cache.build_feature_table(synth_data_root, cp)
    calls = {"n": 0}
    import equipment_analysis.cache as c
    orig = c.features.session_features
    def counting(session, channels=None):
        calls["n"] += 1
        return orig(session, channels)
    monkeypatch.setattr(c.features, "session_features", counting)
    cache.build_feature_table(synth_data_root, cp)   # 재실행 → 신규 없음
    assert calls["n"] == 0

def test_build_feature_table_computes_envelope(synth_data_root, tmp_path):
    df = cache.build_feature_table(synth_data_root, tmp_path / "fe.csv",
                                   bearings={"ch0": {"model": "6204"}})
    assert "ch0_env_bpfo" in df.columns and "ch0_env_bpfi" in df.columns
    assert "ch1_env_bpfo" not in df.columns          # 설정 안 한 채널엔 없음
    # bearings 미지정(기본) → env 컬럼 없음(하위호환)
    df0 = cache.build_feature_table(synth_data_root, tmp_path / "f0.csv")
    assert not any(c.endswith("_env_bpfo") for c in df0.columns)


def test_build_feature_table_applies_calibration(synth_data_root, tmp_path):
    import pytest
    cal = {"before": "20270101_000000", "scale": 10.0}   # 전 세션(2026)이 컷오프 이전
    base = cache.build_feature_table(synth_data_root, tmp_path / "b.csv")
    scaled = cache.build_feature_table(synth_data_root, tmp_path / "s.csv", calibration=cal)
    assert scaled.iloc[0]["ch0_rms"] == pytest.approx(10.0 * base.iloc[0]["ch0_rms"], rel=1e-6)
    assert scaled.iloc[0]["ch0_crest"] == pytest.approx(base.iloc[0]["ch0_crest"], rel=1e-6)


def test_duplicate_stamps_identifies_feature_identical_rows():
    import pandas as pd
    # s4,s5는 특징 완전동일(수집에러 중복), 나머지는 distinct. stamp/error는 해시 제외.
    df = pd.DataFrame({
        "stamp": ["s1", "s2", "s3", "s4", "s5"],
        "error": [None, None, None, None, None],
        "ch0_rms":  [1.0, 2.0, 3.0, 5.0, 5.0],
        "ch0_peak": [1.0, 2.0, 3.0, 9.0, 9.0],
    })
    assert set(cache.duplicate_stamps(df)) == {"s4", "s5"}            # 그룹 전체
    assert set(cache.duplicate_stamps(df, keep_first=True)) == {"s5"}  # 대표1개 유지→나머지만


def test_duplicate_stamps_empty_or_none():
    import pandas as pd
    assert cache.duplicate_stamps(pd.DataFrame()) == []
    df = pd.DataFrame({"stamp": ["a", "b"], "ch0_rms": [1.0, 2.0]})   # 중복 없음
    assert cache.duplicate_stamps(df) == []


def test_load_missing(tmp_path):
    df = cache.load_feature_table(tmp_path / "nope.csv")
    assert df.empty
