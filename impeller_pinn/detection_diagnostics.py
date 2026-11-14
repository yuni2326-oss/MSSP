"""§4.4 검출연구 2차 검토 대응 진단 — 리뷰어 [VERIFY] 항목의 실증 계산.

  (A) 오경보율의 경험적 재추정(더 큰 healthy 풀, leave-one-out) + 특징별 오경보 기여도
      → "다중비교만으로는 ≈7%밖에 설명 못 함, 나머지는 짧은 baseline(n=10)의 잡음 많은
        robust-z scale 때문"이라는 가설을 검증.
  (B) 목표대역(faultband) 특징 **단독** 오경보·검출 카운트(다른 24개 특징을 조건화 제거)
      → 단독 오경보가 0~1/16이면 검출주장이 크게 강화되고 초록도 이 조건부 수치를 써야 함.
  (C) 대역중심 불일치(band-centre mismatch) 강건성 스윕 — 추적대역을 주입내용 대비
      0~±500 Hz 밀어 검출률 변화 측정(습윤·회전 보정 오차 내구성 정량화; §5 최우선 후속).
  + Wilson 95% 신뢰구간(n=16).

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/detection_diagnostics.py
"""
from __future__ import annotations
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from equipment_analysis import dataio, anomaly
from equipment_analysis.features import band_rms_spectral
from impeller_pinn import inject as inject_mod
from impeller_pinn.detection_sweep import (
    BASELINE_N, CHANNEL, FAULTBAND_F1, REL_FLOOR, SENSITIVITY, SEVERITIES, SNRS_DB,
    build_injection, features_for_session,
)

DATA_ROOT = "/mnt/nas/data/GY_DATA/pump data"
SCRATCH = Path(tempfile.gettempdir()) / "impeller_pinn" / "detdiag"
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "detection_diagnostics.json"
FAULTBAND_KEY = f"{CHANNEL}_faultband_f1"
MISMATCH_OFFSETS = (-500.0, -400.0, -300.0, -200.0, -100.0, 0.0,
                    100.0, 200.0, 300.0, 400.0, 500.0)
MISMATCH_CELL = (0.3, 0.0)          # (d_max, snr_db) — 무불일치 시 검출 100%인 셀


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항비율 Wilson 95% 신뢰구간(%). k=성공수, n=시도수."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def z_for(value: float, base_vals: np.ndarray) -> float | None:
    """배포 robust_z와 동일 규약으로 단일 세션 값의 z 산출."""
    base_vals = base_vals[~np.isnan(base_vals)]
    if len(base_vals) < 2 or np.isnan(value):
        return None
    v = np.concatenate([base_vals[:BASELINE_N], [value]])
    return float(anomaly.robust_z(v, BASELINE_N, rel_floor=REL_FLOOR)[-1])


# ─────────────────────── (A) 오경보율 LOO 재추정 + 특징별 기여 ───────────────────────
def part_a(pool_sessions, feature_cols, log=print) -> dict:
    """더 큰 healthy 풀에서 leave-one-out 오경보율: 각 세션을 '검사대상'으로 두고,
    나머지 세션 중 BASELINE_N개를 baseline으로 써서 |z|>3 플래그 여부를 센다.
    베이스라인 자체가 짧아서(n=10) 생기는 scale 잡음의 실효 영향을 직접 측정."""
    rows = [features_for_session(s) for s in pool_sessions]
    tbl = pd.DataFrame(rows)
    n = len(rows)
    log(f"[A] LOO pool = {n} healthy sessions, {len(feature_cols)} features")
    flags, per_feat = [], {f: 0 for f in feature_cols}
    for i in range(n):
        others = [j for j in range(n) if j != i]
        base_idx = others[:BASELINE_N]
        flagged_feats = []
        for f in feature_cols:
            bv = tbl.loc[base_idx, f].to_numpy(dtype=float)
            z = z_for(float(tbl.loc[i, f]), bv)
            if z is not None and abs(z) > SENSITIVITY:
                flagged_feats.append(f)
                per_feat[f] += 1
        flags.append(len(flagged_feats) > 0)
    fa = 100.0 * float(np.mean(flags))
    k = int(np.sum(flags))
    lo, hi = wilson(k, n)
    # 특징당 경험적 단일-특징 오경보율 → 독립 가정 하 세션수준 기대율
    per_feat_rate = {f: c / n for f, c in per_feat.items()}
    indep_expected = 100.0 * (1.0 - np.prod([1.0 - r for r in per_feat_rate.values()]))
    nominal_single = 0.0027                       # 양측 가우시안 |z|>3
    nominal_session = 100.0 * (1.0 - (1 - nominal_single) ** len(feature_cols))
    log(f"[A] LOO session-level false-alarm: {fa:.1f}% ({k}/{n}, Wilson95 {lo:.1f}-{hi:.1f}%)")
    log(f"[A] nominal-Gaussian prediction (25 indep. features, |z|>3): {nominal_session:.1f}%")
    log(f"[A] empirical per-feature rates -> independence-implied session rate: {indep_expected:.1f}%")
    top = sorted(per_feat.items(), key=lambda kv: -kv[1])[:10]
    log("[A] top per-feature false-alarm contributions (count/%d):" % n)
    for f, c in top:
        log(f"      {f:28s} {c:3d}  ({100*c/n:5.1f}%)")
    fb_ct = per_feat.get(FAULTBAND_KEY, 0)
    log(f"[A] target-band feature ({FAULTBAND_KEY}) alone: {fb_ct}/{n} = {100*fb_ct/n:.1f}%")
    return {"n_pool": n, "fa_pct": fa, "fa_k": k, "fa_ci": [lo, hi],
            "nominal_session_pct": nominal_session,
            "independence_implied_pct": indep_expected,
            "per_feature_counts": per_feat,
            "faultband_alone_count": fb_ct}


# ───────── (B)+(C) 주입 1회 통과로 특징별 z + 목표대역 단독 + 대역불일치 스윕 ─────────
def part_bc(baseline_sessions, held_sessions, feature_cols, log=print) -> dict:
    base_tbl = pd.DataFrame([features_for_session(s) for s in baseline_sessions])
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # held-out 건전(비주입)에서의 목표대역 단독 오경보
    ho_rows = [features_for_session(s) for s in held_sessions]
    fb_base = base_tbl[FAULTBAND_KEY].to_numpy(dtype=float)
    fb_fa_flags = [abs(z_for(float(r[FAULTBAND_KEY]), fb_base) or 0.0) > SENSITIVITY for r in ho_rows]
    fb_fa_k = int(np.sum(fb_fa_flags)); n_ho = len(held_sessions)
    fb_fa_ci = wilson(fb_fa_k, n_ho)
    log(f"[B] target-band feature ALONE, false alarms on held-out healthy: "
        f"{fb_fa_k}/{n_ho} = {100*fb_fa_k/n_ho:.1f}% (Wilson95 {fb_fa_ci[0]:.1f}-{fb_fa_ci[1]:.1f}%)")

    # baseline의 offset 대역 값(불일치 스윕용) 미리 계산
    off_base: dict[float, list[float]] = {o: [] for o in MISMATCH_OFFSETS}
    for s in baseline_sessions:
        fr, mg = dataio.load_fft(s.fft_paths[1])
        for o in MISMATCH_OFFSETS:
            off_base[o].append(band_rms_spectral(fr, mg, (FAULTBAND_F1[0] + o, FAULTBAND_F1[1] + o)))
    off_base = {o: np.array(v, dtype=float) for o, v in off_base.items()}

    table_fb: dict[float, dict[float, float]] = {}
    mismatch: dict[float, float] = {}
    tag = 0
    for d_max in SEVERITIES:
        row = {}
        for snr_db in SNRS_DB:
            fb_flags, mm_flags = [], {o: [] for o in MISMATCH_OFFSETS}
            for s in held_sessions:
                tag += 1
                ring, freqs, rpm = build_injection(s, d_max, snr_db, seed=tag)
                stamp = f"{s.stamp[:8]}_{tag % 1_000_000:06d}"
                out_dir = SCRATCH / stamp
                inject_mod.make_injected_session(s.raw_path.parent, s.stamp, out_dir,
                                                 CHANNEL, ring, stamp)
                inj = dataio.discover_sessions(str(out_dir))[0]
                fr, mg = dataio.load_fft(inj.fft_paths[1])
                # 목표대역 단독 검출
                v = band_rms_spectral(fr, mg, FAULTBAND_F1)
                z = z_for(v, fb_base)
                fb_flags.append(abs(z or 0.0) > SENSITIVITY)
                # 대역불일치 스윕(대표 셀에서만)
                if (d_max, snr_db) == MISMATCH_CELL:
                    for o in MISMATCH_OFFSETS:
                        vo = band_rms_spectral(fr, mg, (FAULTBAND_F1[0] + o, FAULTBAND_F1[1] + o))
                        zo = z_for(vo, off_base[o])
                        mm_flags[o].append(abs(zo or 0.0) > SENSITIVITY)
                shutil.rmtree(out_dir, ignore_errors=True)
            k = int(np.sum(fb_flags))
            row[snr_db] = 100.0 * k / len(held_sessions)
            lo, hi = wilson(k, len(held_sessions))
            log(f"[B] d_max={d_max} snr={snr_db:+.0f}dB target-band ALONE: "
                f"{row[snr_db]:5.1f}% ({k}/{len(held_sessions)}, Wilson95 {lo:.1f}-{hi:.1f}%)")
            if (d_max, snr_db) == MISMATCH_CELL:
                for o in MISMATCH_OFFSETS:
                    kk = int(np.sum(mm_flags[o]))
                    mismatch[o] = 100.0 * kk / len(held_sessions)
                log("[C] band-centre mismatch sweep at cell "
                    f"(d_max={d_max}, snr={snr_db:+.0f}dB): "
                    + ", ".join(f"{o:+.0f}Hz:{mismatch[o]:.0f}%" for o in MISMATCH_OFFSETS))
        table_fb[d_max] = row
    return {"faultband_fa_k": fb_fa_k, "faultband_fa_n": n_ho, "faultband_fa_ci": list(fb_fa_ci),
            "table_faultband_only": {str(k): v for k, v in table_fb.items()},
            "mismatch_sweep": {str(k): v for k, v in mismatch.items()},
            "mismatch_cell": list(MISMATCH_CELL)}


def main():
    baseline = dataio.discover_sessions(DATA_ROOT, date_from="20260707_112346",
                                        date_to="20260708_235959")[:10]
    held_pool = dataio.discover_sessions(DATA_ROOT, date_from="20260707_162416",
                                         date_to="20260710_235959")
    held = held_pool[::max(1, len(held_pool) // 16)][:16]
    # (A)용 더 큰 풀: baseline+held 창 전체(드리프트 이전 구간)에서 40개 균등 표집
    pool_all = dataio.discover_sessions(DATA_ROOT, date_from="20260707_112346",
                                        date_to="20260710_235959")
    pool = pool_all[::max(1, len(pool_all) // 40)][:40]

    base_probe = features_for_session(baseline[0])
    feature_cols = [c for c in base_probe
                    if c not in ("stamp", "fs", "rpm") and isinstance(base_probe[c], (int, float))]
    print(f"features scored: {len(feature_cols)} (incl. {FAULTBAND_KEY})", flush=True)

    a = part_a(pool, feature_cols)
    bc = part_bc(baseline, held, feature_cols)

    payload = {"part_a_loo_false_alarm": a, "part_bc": bc,
               "baseline_stamps": [s.stamp for s in baseline],
               "held_out_stamps": [s.stamp for s in held],
               "pool_stamps": [s.stamp for s in pool],
               "faultband_hz": list(FAULTBAND_F1),
               "mismatch_offsets_hz": list(MISMATCH_OFFSETS)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {OUT}")
    print("=== DETECTION_DIAGNOSTICS DONE ===")


if __name__ == "__main__":
    main()
