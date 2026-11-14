# pump_analysis/model.py
"""다변량 비지도 이상탐지 모델(StandardScaler + IsolationForest): 학습/점수/기여도."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

_NON_FEATURE = {"stamp", "error"}


def _dead_channels(table: pd.DataFrame, activity_floor: float) -> set[str]:
    """노이즈 플로어 채널 식별: chN_rms 중앙값이 가장 활성인 채널의 activity_floor 비율
    미만인 채널(예: 분리/고장난 저레벨 센서). 표준화가 이런 채널의 미세 노이즈를 거대한
    z로 증폭해 가짜 이상을 만드므로 학습 특징에서 통째로 제외한다."""
    levels = {col.split("_")[0]: float(np.nanmedian(table[col].to_numpy(dtype=float)))
              for col in table.columns if col.endswith("_rms") and col.startswith("ch")}
    if not levels:
        return set()
    mx = max(levels.values())
    if mx <= 0:
        return set()
    return {ch for ch, lv in levels.items() if lv < activity_floor * mx}


def feature_matrix(table: pd.DataFrame, feature_names: list[str] | None = None,
                   activity_floor: float = 0.05):
    """(X, names, stamps) 반환.
    feature_names=None(학습): 숫자 특징 - 비특징 - 준상수(분산≈0) - 죽은 채널 선택.
    feature_names 주어짐(점수): 그 컬럼으로 정렬(없으면 ValueError). NaN 포함 행 제외."""
    if feature_names is None:
        num = table.select_dtypes(include="number")
        names = [c for c in num.columns if c not in _NON_FEATURE
                 and float(np.nanstd(table[c].to_numpy(dtype=float))) > 1e-9]
        dead = _dead_channels(table, activity_floor)
        names = [c for c in names if c.split("_")[0] not in dead]
    else:
        # 의도적 설계: 특징 불일치 시 교집합으로 진행하지 않고 명확히 실패한다(안전 실패).
        # 추출기 스키마가 바뀌면 여기서 교집합 정렬+경고로 업그레이드 고려.
        missing = [c for c in feature_names if c not in table.columns]
        if missing:
            raise ValueError(f"점수에 필요한 특징이 캐시에 없습니다: {', '.join(missing)}")
        names = list(feature_names)
    has_stamp = "stamp" in table.columns
    sub = table[(["stamp"] if has_stamp else []) + names].dropna(subset=names)
    X = sub[names].to_numpy(dtype=float)
    stamps = (sub["stamp"].astype(str).to_numpy() if has_stamp
              else np.array([str(i) for i in range(len(sub))]))
    return X, names, stamps


def train(table: pd.DataFrame, contamination="auto", random_state: int = 0,
          activity_floor: float = 0.05) -> dict:
    X, names, _ = feature_matrix(table, activity_floor=activity_floor)
    if len(X) < 10:
        raise ValueError(f"학습 표본 부족(n={len(X)}). 최소 10세션 필요.")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=random_state).fit(Xs)
    train_scores = -iso.score_samples(Xs)
    return {
        "scaler": scaler, "model": iso, "feature_names": names,
        "n_train": int(len(X)),
        "activity_floor": float(activity_floor),
        "score_mean": float(np.mean(train_scores)),
        "score_std": float(np.std(train_scores)),
        "score_p95": float(np.quantile(train_scores, 0.95)),
    }


def score(artifact: dict, table: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    names = artifact["feature_names"]
    X, names, stamps = feature_matrix(table, names)
    Xs = artifact["scaler"].transform(X)
    scores = -artifact["model"].score_samples(Xs)     # 높을수록 이상
    flags = artifact["model"].predict(Xs) == -1
    tops = []
    for i in range(len(X)):
        row = Xs[i]
        idx = np.argsort(np.abs(row))[::-1][:top_k]
        tops.append(", ".join(f"{names[j]} {row[j]:+.1f}σ" for j in idx))
    return pd.DataFrame({
        "stamp": stamps,
        "anomaly_score": scores,
        "is_anomaly": flags,
        "top_features": tops,
    })


def save_artifact(artifact: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"모델 아티팩트가 없습니다: {path}")
    return joblib.load(path)
