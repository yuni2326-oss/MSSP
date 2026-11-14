# tests/test_model.py
import numpy as np
import pandas as pd
import pytest
from equipment_analysis import model


def synth_features(n=60, seed=0, inject=None):
    """합성 특징 테이블: stamp + 3채널×(rms,kurtosis,crest) + 상수 rpm/fs.
    rms는 양의 레벨(~0.3, 세 채널 모두 활성)이라 죽은 채널 필터에 걸리지 않는다."""
    rng = np.random.default_rng(seed)
    cols = {}
    for c in range(3):
        cols[f"ch{c}_rms"] = np.abs(rng.normal(0.3, 0.03, n))   # 활성 채널 수준(양수)
        cols[f"ch{c}_kurtosis"] = rng.normal(0, 1, n)
        cols[f"ch{c}_crest"] = rng.normal(0, 1, n)
    df = pd.DataFrame(cols)
    df.insert(0, "stamp", [f"s{i:03d}" for i in range(n)])
    df["rpm"] = 3492.0
    df["fs"] = 51200.0
    if inject:                       # {row_index: {col: value}}
        for i, kv in inject.items():
            for c, v in kv.items():
                df.loc[i, c] = v
    return df


def test_feature_matrix_excludes_nonfeatures_and_constants():
    X, names, stamps = model.feature_matrix(synth_features(n=20))
    assert "stamp" not in names and "rpm" not in names and "fs" not in names
    assert len(names) == 9 and X.shape == (20, 9) and len(stamps) == 20


def test_feature_matrix_excludes_dead_channels():
    # ch2가 노이즈 플로어(rms~0.003)면 ch2 특징 전체가 제외돼야 한다.
    rng = np.random.default_rng(0); n = 30
    df = pd.DataFrame({"stamp": [f"s{i:03d}" for i in range(n)]})
    for ch, lvl in [("ch0", 0.30), ("ch1", 0.40), ("ch2", 0.003)]:
        df[f"{ch}_rms"] = np.abs(rng.normal(lvl, lvl * 0.1, n))
        df[f"{ch}_kurtosis"] = rng.normal(0, 1, n)
    _, names, _ = model.feature_matrix(df)
    assert not any(c.startswith("ch2_") for c in names)        # 죽은 채널 제외
    assert any(c.startswith("ch0_") for c in names)
    assert any(c.startswith("ch1_") for c in names)


def test_train_stores_activity_floor():
    art = model.train(synth_features(n=40))
    assert art["activity_floor"] == 0.05


def test_feature_matrix_dropna():
    df = synth_features(n=10)
    df.loc[3, "ch0_rms"] = np.nan
    X, names, stamps = model.feature_matrix(df)
    assert X.shape[0] == 9 and "s003" not in stamps


def test_train_artifact_keys():
    art = model.train(synth_features(n=60))
    for k in ("scaler", "model", "feature_names", "n_train", "score_p95", "score_mean", "score_std"):
        assert k in art
    assert art["n_train"] == 60 and len(art["feature_names"]) == 9


def test_train_too_few_samples_raises():
    with pytest.raises(ValueError):
        model.train(synth_features(n=5))


def test_save_load_roundtrip(tmp_path):
    art = model.train(synth_features(n=40))
    p = tmp_path / "m.joblib"
    model.save_artifact(art, p)
    assert p.exists()
    loaded = model.load_artifact(p)
    assert loaded["feature_names"] == art["feature_names"]


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        model.load_artifact(tmp_path / "nope.joblib")


def test_score_columns_and_normal():
    art = model.train(synth_features(n=60, seed=1))
    res = model.score(art, synth_features(n=60, seed=1))
    assert list(res.columns) == ["stamp", "anomaly_score", "is_anomaly", "top_features"]
    assert res["is_anomaly"].mean() < 0.2          # 정상 데이터 대부분 정상


def test_score_detects_injected_anomaly():
    art = model.train(synth_features(n=80, seed=2))
    # 마지막 행에 ch0_kurtosis 대폭 상승 주입
    test_df = synth_features(n=80, seed=2, inject={79: {"ch0_kurtosis": 14.0}})
    res = model.score(art, test_df).set_index("stamp")
    assert bool(res.loc["s079", "is_anomaly"]) is True
    assert "ch0_kurtosis" in res.loc["s079", "top_features"]


def test_score_missing_feature_raises():
    art = model.train(synth_features(n=40))
    bad = synth_features(n=10).drop(columns=["ch0_kurtosis"])
    with pytest.raises(ValueError):
        model.score(art, bad)


def test_score_top_k_param():
    art = model.train(synth_features(n=50, seed=3))
    res = model.score(art, synth_features(n=50, seed=3), top_k=1)
    # top_features는 단일 특징만 포함(쉼표 없음)
    assert all("," not in tf for tf in res["top_features"])
