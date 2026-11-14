# pump_analysis/anomaly.py
"""비지도 이상탐지: 초기 정상구간(baseline) 대비 robust z-score(MAD)."""
from __future__ import annotations
import numpy as np
import pandas as pd

MODEL_HOOK = None   # Phase 2: callable(table, feature_col)->mask 주입점

def _robust_z_single(v: np.ndarray, baseline_n: int, stamps: np.ndarray | None,
                     baseline_from: str, rel_floor: float) -> np.ndarray:
    """단일 baseline robust z-score(MAD). rpm 미조건 기본 동작."""
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return v
    n = min(baseline_n, len(v))
    base = v[:n]
    if baseline_from and stamps is not None:
        sel = np.asarray(stamps) >= baseline_from
        cand = v[sel]
        if len(cand) > 0:
            base = cand[:min(baseline_n, len(cand))]
    med = np.median(base)
    mad = np.median(np.abs(base - med))
    scale = 1.4826 * mad
    std_base = float(np.std(base))
    if scale < std_base:        # 소표본 MAD 퇴화 방지: baseline 표준편차를 scale 하한으로
        scale = std_base
    rel = rel_floor * abs(float(med))
    if scale < rel:             # 상대 하한: median의 rel_floor 미만 변화는 이상으로 안 봄
        scale = rel
    if scale == 0:              # 상수 baseline(+median 0) → 미세 ε
        scale = 1e-9
    return (v - med) / scale


def robust_z(values: np.ndarray, baseline_n: int, *, stamps: np.ndarray | None = None,
             baseline_from: str = "", rel_floor: float = 0.0,
             rpm: np.ndarray | None = None, rpm_bin: float = 900.0) -> np.ndarray:
    """초기 정상구간(baseline) 대비 robust z-score(MAD).
    baseline_from + stamps 주어지면 baseline=stamp>=baseline_from인 세션의 첫 baseline_n개
    (해당 구간 없으면 맨 앞 baseline_n로 폴백) — 센서 재장착/단위변경 이후 구간을 기준으로.
    rel_floor>0이면 scale 하한을 rel_floor·|median|로 둬 균일 baseline에서 z 폭증 방지.
    **rpm 주어지면 rpm_bin(rpm) 구간별로 baseline·z를 따로 계산(fable F1)** — 가변속(20~120Hz)서
    진폭특징이 속도에 U²~로 변해 단일 baseline이 무효화되는 것 방지(각 속도 이웃 대비 이상 판정).
    rpm≤0(미검출)은 자체 그룹으로 폴백. rpm=None이면 기존 단일 baseline(하위호환)."""
    v = np.asarray(values, dtype=float)
    if rpm is None:
        return _robust_z_single(v, baseline_n, stamps, baseline_from, rel_floor)
    rpm = np.asarray(rpm, dtype=float)
    stamps_arr = np.asarray(stamps) if stamps is not None else None
    z = np.full(len(v), np.nan)
    binidx = np.where(rpm > 0, np.floor(rpm / rpm_bin), -1.0)
    for b in np.unique(binidx):
        m = binidx == b
        st_m = stamps_arr[m] if stamps_arr is not None else None
        z[m] = _robust_z_single(v[m], baseline_n, st_m, baseline_from, rel_floor)
    return z

def flag_anomalies(values: np.ndarray, baseline_n: int = 10,
                   sensitivity: float = 3.0, **kw) -> np.ndarray:
    z = robust_z(values, baseline_n, **kw)
    return np.abs(z) > sensitivity

def detect_trend_anomalies(table: pd.DataFrame, feature_col: str,
                           baseline_n: int = 10, sensitivity: float = 3.0,
                           baseline_from: str = "", rel_floor: float = 0.0,
                           rpm_binned: bool = False, rpm_bin: float = 900.0) -> pd.DataFrame:
    if feature_col not in table.columns:
        raise ValueError(f"특징 컬럼이 없습니다: {feature_col}")
    table = table[table[feature_col].notna()].reset_index(drop=True)  # 손상 세션 NaN 행 제외(거짓음성 방지)
    vals = table[feature_col].to_numpy(dtype=float)
    rpm = (table["rpm"].to_numpy(dtype=float)                         # rpm-binned baseline(F1, 가변속)
           if rpm_binned and "rpm" in table.columns else None)
    z = robust_z(vals, baseline_n, stamps=table["stamp"].astype(str).to_numpy(),
                 baseline_from=baseline_from, rel_floor=rel_floor, rpm=rpm, rpm_bin=rpm_bin)
    return pd.DataFrame({
        "stamp": table["stamp"].astype(str).to_numpy(),
        "value": vals,
        "z": z,
        "is_anomaly": np.abs(z) > sensitivity,
    })
