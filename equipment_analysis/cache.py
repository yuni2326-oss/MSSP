# pump_analysis/cache.py
"""세션별 특징 캐시(CSV). 증분 계산: 신규 stamp만 추가."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import dataio, features

def duplicate_stamps(table: pd.DataFrame, keep_first: bool = False) -> list[str]:
    """특징 벡터가 완전 동일한 행(수집에러로 같은 데이터가 반복 저장된 세션)의 stamp 반환.
    stamp/error는 비교에서 제외. keep_first=False면 그룹 전체, True면 대표 1개 남기고 나머지."""
    if table is None or table.empty or "stamp" not in table.columns:
        return []
    feat = [c for c in table.columns if c not in ("stamp", "error")]
    if not feat:
        return []
    h = pd.util.hash_pandas_object(table[feat], index=False)
    dup_mask = h.duplicated(keep="first" if keep_first else False)
    return sorted(table.loc[dup_mask.values, "stamp"].astype(str))


def load_feature_table(cache_path: str | Path) -> pd.DataFrame:
    p = Path(cache_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype={"stamp": str})

def build_feature_table(data_root: str | Path, cache_path: str | Path,
                        channels: list[str] | None = None,
                        rpm_range: tuple[float, float] = (20.0, 120.0),
                        bearings: dict[str, dict] | None = None,
                        env_band: tuple[float, float] = (2000.0, 10000.0),
                        calibration: dict | None = None,
                        pwm_notch: dict | None = None,
                        resonances: dict | None = None,
                        speed_gate: bool = False) -> pd.DataFrame:
    existing = load_feature_table(cache_path)
    done = set(existing["stamp"]) if not existing.empty else set()
    sessions = dataio.discover_sessions(data_root)
    new_rows = []
    for s in sessions:
        if s.stamp in done:
            continue
        try:
            new_rows.append(features.session_features(s, channels, rpm_range,
                                                       bearings=bearings, env_band=env_band,
                                                       calibration=calibration, pwm_notch=pwm_notch,
                                                       resonances=resonances, speed_gate=speed_gate))
        except Exception as e:  # 손상 세션은 건너뛰고 계속
            new_rows.append({"stamp": s.stamp, "error": str(e)})
    if new_rows:
        df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df = existing
    if not df.empty:
        df = df.sort_values("stamp").reset_index(drop=True)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df
