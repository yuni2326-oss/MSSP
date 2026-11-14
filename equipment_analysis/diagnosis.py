"""분석 결과 → LLM에 줄 '근거 팩' 구성(숫자만; 해석은 LLM 몫)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import anomaly, model
from .equipment import EquipmentConfig

_NON_FEATURE = {"stamp", "fs", "rpm", "error"}


def build_model_evidence(table: pd.DataFrame, model_path, top_k: int = 5) -> dict | None:
    """학습된 다변량 모델(IsolationForest)로 세션 점수화 → 요약. 모델 없으면 None.
    반환: {n_scored, n_anomaly, top:[{stamp, score, top_features}...]}. top은 이상세션 점수
    내림차순 top_k(이상 없으면 전체 상위)."""
    if not Path(model_path).exists():
        return None
    art = model.load_artifact(model_path)
    res = model.score(art, table)
    if len(res) == 0:
        return {"n_scored": 0, "n_anomaly": 0, "top": []}
    anom = res[res["is_anomaly"]]
    pick = (anom if len(anom) else res).sort_values("anomaly_score", ascending=False).head(top_k)
    top = [{"stamp": str(r["stamp"]), "score": float(r["anomaly_score"]),
            "top_features": r["top_features"]} for _, r in pick.iterrows()]
    return {"n_scored": int(len(res)), "n_anomaly": int(res["is_anomaly"].sum()), "top": top}


def build_evidence(table: pd.DataFrame, cfg: EquipmentConfig,
                   features: list[str] | None = None) -> dict:
    if table is None or table.empty or "stamp" not in table.columns:
        return {"n_sessions": 0, "period": None, "rpm": None, "flagged_features": []}
    feats = features or [c for c in table.columns
                         if c not in _NON_FEATURE and table[c].dtype.kind in "fi"]
    flagged = []
    for f in feats:
        sub = table[table[f].notna()]
        col = sub[f].to_numpy(dtype=float)
        if len(col) < 2:
            continue
        rpm = (sub["rpm"].to_numpy(dtype=float)                       # rpm 구간별 baseline(F1, 가변속)
               if cfg.rpm_binned and "rpm" in sub.columns else None)
        z = anomaly.robust_z(col, cfg.baseline_n, stamps=sub["stamp"].astype(str).to_numpy(),
                             baseline_from=cfg.baseline_from, rel_floor=cfg.z_rel_floor,
                             rpm=rpm, rpm_bin=cfg.rpm_bin)
        mask = np.abs(z) > cfg.sensitivity
        if mask.any():
            zf = z[mask]
            z_peak = float(zf[np.argmax(np.abs(zf))])
            flagged.append({"feature": f, "z": float(z[-1]), "z_peak": z_peak,
                            "n_flagged": int(mask.sum())})
    flagged.sort(key=lambda d: abs(d["z_peak"]), reverse=True)
    stamps = [str(s) for s in table["stamp"].tolist()]
    rpm = float(table["rpm"].dropna().median()) if "rpm" in table and table["rpm"].notna().any() else None
    return {
        "n_sessions": int(len(table)),
        "period": (stamps[0], stamps[-1]) if stamps else None,
        "rpm": rpm,
        "flagged_features": flagged[:20],
    }
