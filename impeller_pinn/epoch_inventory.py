"""상시 기록 폴더의 목록과 운전 상태 — 2026-10-06~22 `pump data2` 판독(상태 문서 r4).

세 가지를 낸다.

  1. 세션 목록 — 날짜별 세션 수, 처음·마지막 세션.
  2. 일일 압력·온도 로그 — 날짜별 p_d·p_s·T_w 중앙값과 범위, 그리고 p_d의 계단 변화(앞뒤 k점 창의
     중앙값 차가 문턱을 넘는 시각과 앞뒤 수준).
  3. 날마다 표본 세션의 운전 판독 — `pilot_cavitation.session_record`(f_rot·채널 RMS·양정)를 그대로 쓴다.

토출압만 계단처럼 바뀌고 회전·진동이 그대로면 펌프 정지가 아니라 토출압 측정이나 배관의 변화다.
판독은 자료의 상태를 기록할 뿐 baseline 여부를 판정하지 않는다(재동결 규약은 상태 문서 §4.2).

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/epoch_inventory.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from impeller_pinn.pilot_cavitation import CHANNELS, read_log, session_record

DATA = Path("/mnt/nas/data/GY_DATA/pump data2/pump data2")
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "pump_data2_inventory.json"
PRESSURE_ALIASES = {"ps": ("p_suction_bar", "ai1_bar"), "pd": ("p_discharge_bar", "ai0_bar")}
TEMPERATURE_ALIASES = {"Tw": ("ai1_C",), "T_ai0": ("ai0_C",)}     # T_w = ai1 (09-02 판독과 같은 규약)
STEP_K, STEP_BAR = 10, 0.05     # 1분 로그의 앞뒤 10점(10분) 중앙값 차 0.05 bar


def level_steps(t, v, k: int = STEP_K, threshold: float = STEP_BAR) -> list:
    """계단 변화 — 각 점 i에서 v[i−k:i]와 v[i:i+k]의 중앙값 차가 문턱을 넘으면 후보로 삼는다. 이어진 후보
    묶음마다 차가 가장 큰 점의 앞뒤 중앙값이 두 수준이다. 중앙값 차는 계단 앞뒤 k/2점 안에서 거의 같아
    위치를 정하지 못하므로, 위치는 묶음 언저리에서 **새 수준에 더 가까워지는 첫 점**으로 정한다.
    반환: [{time, before, after, index}] (time은 새 수준의 첫 점)."""
    v = np.asarray(v, float)
    n = len(v)
    cand = []
    for i in range(k, n - k + 1):
        before, after = np.nanmedian(v[i - k:i]), np.nanmedian(v[i:i + k])
        if abs(after - before) > threshold:
            cand.append((i, before, after))
    events, group = [], []
    for c in cand + [None]:
        if c is not None and (not group or c[0] == group[-1][0] + 1):
            group.append(c)
            continue
        if group:
            _, b, a = max(group, key=lambda x: abs(x[2] - x[1]))
            lo, hi = max(group[0][0] - k, 0), min(group[-1][0] + k, n - 1)
            j = next((j for j in range(lo, hi + 1) if abs(v[j] - a) < abs(v[j] - b)), group[0][0])
            events.append({"time": np.datetime_as_string(t[j], unit="s"), "before": float(b), "after": float(a),
                           "index": int(j)})
        group = [c] if c is not None else []
    return events


def sample_sessions(names, per_day: int = 2) -> list:
    """날마다 세션 목록의 (j+1)/(per_day+1) 지점(j = 0…per_day−1)을 고른다 — 하루의 앞뒤 끝을 피한다."""
    by_day = {}
    for nm in sorted(names):
        by_day.setdefault(str(nm)[:8], []).append(nm)
    out = []
    for day in sorted(by_day):
        lst = by_day[day]
        idx = sorted({int(len(lst) * (j + 1) / (per_day + 1)) for j in range(per_day)})
        out += [lst[i] for i in idx]
    return out


def _read_daily(folder: Path, prefix: str, aliases: dict):
    """`<prefix>_YYYYMMDD.csv`를 모두 읽어 시간순으로 잇는다."""
    ts, vals = [], {k: [] for k in aliases}
    for f in sorted(folder.glob(f"{prefix}_*.csv")):
        t, v = read_log(f, aliases)
        if t is None:
            continue
        ts.append(t)
        for k in aliases:
            vals[k].append(v[k])
    if not ts:
        return None, None
    t = np.concatenate(ts)
    o = np.argsort(t, kind="stable")
    return t[o], {k: np.concatenate(x)[o] for k, x in vals.items()}


def _daily(t, vals: dict, keys: dict) -> dict:
    """날짜별 중앙값·최소·최대 — keys: {출력 이름: vals의 키}."""
    days = np.array([d.replace("-", "") for d in np.datetime_as_string(t, unit="D")])
    out = {}
    for day in sorted(set(days)):
        m = days == day
        row = {"n": int(m.sum())}
        for name, key in keys.items():
            x = vals[key][m]
            row[f"{name}_median"], row[f"{name}_min"], row[f"{name}_max"] = (
                float(np.nanmedian(x)), float(np.nanmin(x)), float(np.nanmax(x)))
        out[day] = row
    return out


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def inventory(folder=DATA, out=OUT, per_day: int = 2, log=print) -> dict:
    folder = Path(folder)
    files = sorted(folder.glob("*_raw.npz"))
    names = [f.name for f in files]
    per = {}
    for nm in names:
        per[nm[:8]] = per.get(nm[:8], 0) + 1
    t_p, pres = _read_daily(folder, "pressure", PRESSURE_ALIASES)
    t_t, temp = _read_daily(folder, "temperature", TEMPERATURE_ALIASES)
    res = {"folder": str(folder), "n_sessions": len(files), "sessions_per_day": per,
           "first": names[0][:15] if names else None, "last": names[-1][:15] if names else None,
           "pressure": None, "temperature": None, "samples": [],
           "settings": {"step_window_points": STEP_K, "step_threshold_bar": STEP_BAR, "samples_per_day": per_day,
                        "Tw_channel": "ai1_C"}}
    if t_p is not None:
        res["pressure"] = {"daily": _daily(t_p, pres, {"p_d": "pd", "p_s": "ps"}),
                           "p_d_steps": level_steps(t_p, pres["pd"]),
                           "first": np.datetime_as_string(t_p[0], unit="s"),
                           "last": np.datetime_as_string(t_p[-1], unit="s")}
    if t_t is not None:
        res["temperature"] = {"daily": _daily(t_t, temp, {"Tw": "Tw", "T_ai0": "T_ai0"})}
    keep = ("stamp", "f_rot_Hz", "running", "p_d_bar", "p_s_bar", "Tw_C", "head_m") + tuple(
        f"{c}_rms" for c in CHANNELS)
    for nm in sample_sessions(names, per_day):
        rec = session_record(folder / nm, t_p, pres, t_t, temp)
        res["samples"].append({k: rec.get(k) for k in keep})
        log(f"  {nm[:15]} f_rot {rec['f_rot_Hz']:.1f} Hz  p_d {rec['p_d_bar']:.3f} bar  ch0_rms {rec['ch0_rms']:.2f}")
    res = _clean(res)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main(log=print):
    res = inventory(log=log)
    log(f"[pump data2] 세션 {res['n_sessions']} ({res['first']} – {res['last']}), "
        f"p_d 계단 {len(res['pressure']['p_d_steps'])}건 → {OUT}")
    for ev in res["pressure"]["p_d_steps"]:
        log(f"  p_d {ev['before']:.3f} → {ev['after']:.3f} bar @ {ev['time']}")
    return res


if __name__ == "__main__":
    main()
