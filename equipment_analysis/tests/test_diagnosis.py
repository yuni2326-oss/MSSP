import numpy as np, pandas as pd
import pytest
from equipment_analysis import diagnosis
from equipment_analysis.equipment import EquipmentConfig

def test_build_evidence_flags_spiking_feature():
    # baseline 10세션 평탄 + 마지막에 급등 → 이탈 검출
    vals = [1.0] * 12 + [9.0]
    stamps = [f"2026061{i:02d}_000000" for i in range(len(vals))]
    table = pd.DataFrame({"stamp": stamps, "rpm": [3492.0]*len(vals),
                          "ch0_rms": vals})
    ev = diagnosis.build_evidence(table, EquipmentConfig(baseline_n=10, sensitivity=3.0))
    assert ev["n_sessions"] == len(vals)
    assert ev["flagged_features"] and ev["flagged_features"][0]["feature"] == "ch0_rms"

def test_build_evidence_empty_table():
    ev = diagnosis.build_evidence(pd.DataFrame(), EquipmentConfig())
    assert ev["n_sessions"] == 0 and ev["flagged_features"] == []


def test_build_evidence_honors_baseline_from_and_floor():
    # 앞 10(옛 위치/스케일, 값 100, 무시대상) + baseline_from 이후 정상(값 10) + 마지막 소폭상승(11).
    stamps = [f"20260101_{i:06d}" for i in range(10)] + [f"20260707_{i:06d}" for i in range(12)]
    vals = [100.0] * 10 + [10.0] * 11 + [11.0]
    table = pd.DataFrame({"stamp": stamps, "rpm": [3492.0] * len(vals), "ch0_rms": vals})
    cfg = EquipmentConfig(baseline_n=10, sensitivity=3.0,
                          baseline_from="20260707_000000", z_rel_floor=0.05)
    ev = diagnosis.build_evidence(table, cfg)
    ff = {f["feature"]: f for f in ev["flagged_features"]}
    assert "ch0_rms" in ff
    # baseline=새 구간(med=10, floor=0.5) 기준 → 최근(11) z=(11-10)/0.5=2.0(옛 med=100 아님).
    assert ff["ch0_rms"]["z"] == pytest.approx(2.0, rel=1e-2)


def test_build_model_evidence_summarizes_anomalies(tmp_path):
    from equipment_analysis import model
    rng = np.random.default_rng(0)
    n = 30
    stamps = [f"20260601_{i:06d}" for i in range(n)]
    table = pd.DataFrame({
        "stamp": stamps, "rpm": [3492.0] * n,
        "ch0_rms": rng.normal(1.0, 0.1, n),
        "ch1_rms": rng.normal(2.0, 0.2, n),
        "ch0_peak": rng.normal(3.0, 0.3, n),
    })
    art = model.train(table, contamination=0.1)
    mp = tmp_path / "m.joblib"; model.save_artifact(art, mp)
    ev = diagnosis.build_model_evidence(table, mp)
    assert ev is not None
    assert ev["n_scored"] == n and ev["n_anomaly"] >= 1
    assert ev["top"] and set(ev["top"][0]) >= {"stamp", "score", "top_features"}
    # top은 이상점수 내림차순
    scores = [t["score"] for t in ev["top"]]
    assert scores == sorted(scores, reverse=True)


def test_build_model_evidence_no_model_returns_none(tmp_path):
    table = pd.DataFrame({"stamp": ["a"], "ch0_rms": [1.0]})
    assert diagnosis.build_model_evidence(table, tmp_path / "nope.joblib") is None

def test_build_evidence_zpeak_captures_midseries_spike():
    # baseline flat (10), spike in the middle (2), recovers by the last sample (3)
    vals = [1.0] * 10 + [9.0] * 2 + [1.0] * 3
    stamps = [f"2026061{i:02d}_000000" for i in range(len(vals))]
    table = pd.DataFrame({"stamp": stamps, "rpm": [3492.0] * len(vals), "ch0_rms": vals})
    ev = diagnosis.build_evidence(table, EquipmentConfig(baseline_n=10, sensitivity=3.0))
    assert ev["flagged_features"], "mid-series spike must be flagged"
    ff = ev["flagged_features"][0]
    assert ff["feature"] == "ch0_rms"
    assert abs(ff["z_peak"]) > 3.0        # peak captures the spike severity
    assert abs(ff["z"]) < 1.0             # last-sample z is ~0 (recovered)
    assert ff["n_flagged"] == 2
