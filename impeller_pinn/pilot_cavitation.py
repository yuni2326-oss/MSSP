"""2026-09-02 예비 캐비테이션 시험(압력센서 설치 후 첫 실측) — σ 산출과 채널별 응답.

사전등록 12단계 ladder가 아니라 **예비 스로틀 시험**이다(런 3개, 흡입 버터플라이를
2~3분간 조였다 푸는 방식). 이 모듈은 세 가지를 한다.

  1. 세션별 σ 산출 — 원고 부록 B.2 식 σ = (p_s,abs − p_v(T_w)) / (½ρU²),
     U = π·D_eye·f_rot, f_rot은 세션마다 ch0 스펙트럼에서 실측(인버터 구동이므로 가변).
  2. Tier-1 동결 자연변동 p95(`tier1_natural_variation.json`)와 σ 구간별 대역에너지 대조.
  3. **귀속 검사** — 계획서 v1.6이 밸브캐비 통제용으로 지정한 흡입 엘보 ch2를 포함해
     전 채널의 스로틀 응답을 비교한다. 임펠러 옆(ch0)과 흡입 엘보(ch2)가 반대 방향으로
     움직이면 관측된 신호는 임펠러 캐비가 아니다.

대역·마스크·추정기 규약은 tier1_natural_variation.py와 동일하게 유지한다.

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/pilot_cavitation.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from equipment_analysis.npz_io import (
    amplitude_spectrum, band_rms, electrical_mask, load_npz_raw,
)
from impeller_pinn.thermal_analysis import p_v_kPa, rho_water

DATA_ROOT = Path("/mnt/nas/data/GY_DATA/pump data")
RUNS = ("cavitation_test", "catitation_test2", "cavitation_test3")   # test2 폴더명은 원본 오타
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "pilot_cavitation_20260902.json"
NAT = GEN / "data" / "paper1" / "tier1_natural_variation.json"

EYE_DIA_M = 0.0308              # 임펠러 아이 직경 (thermal_analysis와 동일)
P_ATM_KPA = 101.325             # 현장 기압계 없음 — 표준대기 가정(§ 민감도 dσ/dp_atm ≈ 0.064 /kPa)
G = 9.80665
N_VANES = 6                     # BPF = 6 × f_rot (349.2 Hz @ 58.2 Hz)
BANDS = {"carrier_2p9_3p4k": (2900.0, 3400.0), "b1_5k": (1000.0, 5000.0),
         "b5_10k": (5000.0, 10000.0), "b10_20k": (10000.0, 20000.0)}
CHANNELS = ("ch0", "ch1", "ch2", "ch3", "ch5", "ch6")
CH_WHERE = {"ch0": "임펠러 옆", "ch1": "모터 연결부", "ch2": "흡입 엘보(밸브 참조)",
            "ch3": "모터 NDE", "ch5": "토출 플랜지", "ch6": "마이크"}
THROTTLE_BAR = -0.05            # 이보다 낮으면 '스로틀 중'으로 본다
TIER1 = (("ch0", "carrier_2p9_3p4k"), ("ch0", "b1_5k"), ("ch0", "b5_10k"), ("ch6", "b1_5k"))


# ---------------------------------------------------------------- 순수 함수

def cavitation_number(p_s_bar_g, T_w_C, f_rot_Hz, p_atm_kPa=P_ATM_KPA, eye_dia_m=EYE_DIA_M):
    """흡입 게이지압[bar]·수온[°C]·축주파수[Hz] → σ와 NPSH_a(속도수두 제외).

    σ = (p_s,abs − p_v(T_w)) / q,  q = ½ρ(T_w)U²,  U = π·D_eye·f_rot.
    NPSH_a = (p_s,abs − p_v)/(ρg) — 유량 미계측이므로 V_s²/2g는 빠져 있다(하한).
    """
    U = np.pi * eye_dia_m * float(f_rot_Hz)
    rho = float(rho_water(T_w_C))
    q_kPa = 0.5 * rho * U ** 2 / 1000.0
    p_abs = float(p_atm_kPa) + float(p_s_bar_g) * 100.0
    pv = float(p_v_kPa(T_w_C))
    return {"sigma": (p_abs - pv) / q_kPa, "NPSHa_m": (p_abs - pv) * 1000.0 / (rho * G),
            "q_kPa": q_kPa, "p_v_kPa": pv, "p_s_abs_kPa": p_abs, "U_m_s": U, "rho": rho}


def pump_head_m(p_d_bar_g, p_s_bar_g, T_w_C):
    """펌프 양정 H = (p_d − p_s)/ρg [m] — 계획서 §11 판정표(ISO 9906 3 % 양정강하)의 기본량.

    흡입을 조이면 p_s가 내려가 H가 **올라간다**. 유량이 고정돼 있지 않으면 이 상승은
    펌프 곡선 위의 운전점 이동이지 캐비테이션 신호가 아니다.
    """
    return (float(p_d_bar_g) - float(p_s_bar_g)) * 1e5 / (float(rho_water(T_w_C)) * G)


def f_rot_from_spectrum(freq, mag, lo=45.0, hi=70.0):
    """ch0 진폭 스펙트럼의 1× 선 — 인버터 구동이라 세션마다 다시 잡는다."""
    s = (np.asarray(freq) >= lo) & (np.asarray(freq) <= hi)
    if not s.any():
        return float("nan"), float("nan")
    i = int(np.argmax(np.asarray(mag)[s]))
    return float(np.asarray(freq)[s][i]), float(np.asarray(mag)[s][i])


def peak_near(freq, mag, f_c, tol=8.0):
    s = (np.asarray(freq) > f_c - tol) & (np.asarray(freq) < f_c + tol)
    return float(np.asarray(mag)[s].max()) if s.any() else 0.0


def bpf_sideband_ratio(freq, mag, f_rot, n_vanes=N_VANES):
    """BPF 측파대비 = ½(BPF−1× + BPF+1×) / BPF — 블레이드패스 게이팅 판별자."""
    f_bpf = n_vanes * f_rot
    b = peak_near(freq, mag, f_bpf)
    if b <= 0:
        return float("nan")
    return 0.5 * (peak_near(freq, mag, f_bpf - f_rot) + peak_near(freq, mag, f_bpf + f_rot)) / b


def db_ratio(x, ref):
    return 20.0 * np.log10(np.asarray(x, float) / float(ref))


# ---------------------------------------------------------------- 로그 읽기

def read_log(path, aliases):
    """timestamp + 별칭 허용 컬럼 → (np.datetime64 배열, {name: 값 배열}). 없으면 (None, None)."""
    path = Path(path)
    if not path.exists():
        return None, None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, None
    t = np.array([np.datetime64(r["timestamp"].replace(" ", "T")) for r in rows])
    out = {}
    for name, alias in aliases.items():
        key = next((a for a in alias if a in rows[0]), None)
        out[name] = np.array([float(r[key]) for r in rows]) if key else np.full(len(rows), np.nan)
    return t, out


def sample_log(t_log, v, t0, half_s):
    """t0±half_s 창의 (평균, 최대−최소, 최근접 시간차[s]). 창이 비면 최근접 1점."""
    if t_log is None or not len(t_log):
        return np.nan, np.nan, np.nan
    dt = (t_log - t0) / np.timedelta64(1, "s")
    near = float(np.min(np.abs(dt)))
    m = np.abs(dt) <= half_s
    if not m.any():
        return float(v[int(np.argmin(np.abs(dt)))]), 0.0, near
    return float(np.nanmean(v[m])), float(np.nanmax(v[m]) - np.nanmin(v[m])), near


# ---------------------------------------------------------------- 스캔

def session_record(path, t_p, pres, t_t, temp):
    chans, fs, meta = load_npz_raw(path)
    stamp = meta["stamp"]
    t0 = np.datetime64(datetime.strptime(stamp, "%Y%m%d_%H%M%S"))
    dur = len(chans["ch0"]) / fs
    t_mid = t0 + np.timedelta64(int(dur * 500), "ms")
    ps, ps_spread, ps_gap = sample_log(t_p, pres["ps"], t_mid, max(dur / 2.0, 3.0))
    pd_, _, _ = sample_log(t_p, pres["pd"], t_mid, max(dur / 2.0, 3.0))
    Tw, _, Tw_gap = sample_log(t_t, temp["Tw"], t_mid, 120.0)

    fr, mg0 = amplitude_spectrum(chans["ch0"], fs)
    keep = electrical_mask(fr)
    f_rot, a_1x = f_rot_from_spectrum(fr, mg0)
    rec = {"stamp": stamp, "dur_s": round(dur, 3), "p_s_bar": ps, "p_s_spread_bar": ps_spread,
           "p_log_gap_s": ps_gap, "p_d_bar": pd_, "Tw_C": Tw, "T_log_gap_s": Tw_gap,
           "f_rot_Hz": f_rot, "a_1x": a_1x,
           "running": bool(a_1x > 0.05 and 55.0 <= f_rot <= 62.0),
           "stationary": bool(np.isfinite(ps_spread) and ps_spread < 0.02)}
    for ch in CHANNELS:
        if ch not in chans:
            continue
        f_, m_ = (fr, mg0) if ch == "ch0" else amplitude_spectrum(chans[ch], fs)
        k_ = keep if ch == "ch0" else electrical_mask(f_)
        for name, band in BANDS.items():
            rec[f"{ch}_{name}"] = band_rms(f_, m_, band, k_)
        rec[f"{ch}_rms"] = float(np.sqrt(np.mean(chans[ch] ** 2)))
        rec[f"{ch}_bpfmod"] = bpf_sideband_ratio(f_, m_, f_rot)
    if np.isfinite(ps) and np.isfinite(Tw) and np.isfinite(f_rot):
        rec.update(cavitation_number(ps, Tw, f_rot))
        if np.isfinite(pd_):
            rec["head_m"] = pump_head_m(pd_, ps, Tw)
    return rec


def scan_run(run, data_root=DATA_ROOT, log=print):
    d = Path(data_root) / run
    t_p, pres = read_log(d / "pressure_20260902.csv",
                         {"ps": ("p_suction_bar", "ai1_bar"), "pd": ("p_discharge_bar", "ai0_bar")})
    # 당일 전체 온도로그는 21:42에 끊긴다 → 런 폴더 로그와 합쳐 쓴다
    t_day, temp_day = read_log(Path(data_root) / "temperature_20260902.csv", {"Tw": ("ai1_C",)})
    t_run, temp_run = read_log(d / "temperature_20260902.csv", {"Tw": ("ai1_C",)})
    if t_run is not None and t_day is not None:
        t_t = np.concatenate([t_day, t_run]); tw = np.concatenate([temp_day["Tw"], temp_run["Tw"]])
        o = np.argsort(t_t); t_t, temp = t_t[o], {"Tw": tw[o]}
    else:
        t_t, temp = (t_run, temp_run) if t_run is not None else (t_day, temp_day)
    files = sorted(d.glob("*_raw.npz"))
    log(f"[{run}] npz={len(files)}")
    return [session_record(f, t_p, pres, t_t, temp) for f in files]


# ---------------------------------------------------------------- 요약

def sigma_bins(recs, edges, key, ref, band_keys):
    """스로틀 세션을 σ 구간으로 묶어 기준(ref) 대비 중앙값 ΔdB."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in recs if np.isfinite(r.get(key, np.nan)) and lo <= r[key] < hi]
        if not sel:
            continue
        row = {"sigma_lo": lo, "sigma_hi": hi, "n": len(sel)}
        for bk in band_keys:
            row[bk] = float(db_ratio(np.median([r[bk] for r in sel]), ref[bk]))
        out.append(row)
    return out


def main(log=print):
    nat = json.load(open(NAT))["natural_variation"]
    p95 = {f"{ch}_{b}": nat[f"{ch}_{b}_logratio_dB"]["p95"] for ch in ("ch0", "ch6") for b in BANDS}
    result = {"scope": {"date": "2026-09-02", "runs": list(RUNS), "p_atm_kPa": P_ATM_KPA,
                        "eye_dia_m": EYE_DIA_M, "n_vanes": N_VANES,
                        "note": "사전등록 ladder 아님 — 예비 스로틀 시험"},
              "frozen_p95_dB": p95, "runs": {}}

    all_recs = {}
    for run in RUNS:
        recs = scan_run(run, log=log)
        all_recs[run] = recs
        run_ok = [r for r in recs if r["running"]]
        base = [r for r in run_ok if r["p_s_bar"] > -0.02]
        thr = [r for r in run_ok if r["p_s_bar"] < THROTTLE_BAR]
        med = lambda rs, k: float(np.median([r[k] for r in rs])) if rs else float("nan")
        result["runs"][run] = {
            "n_sessions": len(recs), "n_running": len(run_ok), "dur_s": recs[0]["dur_s"],
            "f_rot_Hz": med(run_ok, "f_rot_Hz"), "q_kPa": med(run_ok, "q_kPa"),
            "Tw_C": med(run_ok, "Tw_C"),
            "sigma_baseline": med(base, "sigma"), "NPSHa_baseline_m": med(base, "NPSHa_m"),
            "sigma_min": float(np.min([r["sigma"] for r in thr])) if thr else None,
            "NPSHa_min_m": float(np.min([r["NPSHa_m"] for r in thr])) if thr else None,
            "p_s_min_bar": float(np.min([r["p_s_bar"] for r in thr])) if thr else None,
            "p_d_baseline_bar": med(base, "p_d_bar"), "p_d_throttled_bar": med(thr, "p_d_bar"),
            "head_baseline_m": med(base, "head_m"),
            "head_deepest_m": (float(np.median([r["head_m"] for r in
                                                sorted(thr, key=lambda x: x["p_s_bar"])[:max(3, len(thr) // 10)]]))
                               if thr else None),
            "dTw_C": med(thr, "Tw_C") - med(base, "Tw_C") if thr and base else None,
            "n_throttled": len(thr),
        }
        r = result["runs"][run]
        log(f"  f_rot {r['f_rot_Hz']:.1f} Hz  q {r['q_kPa']:.2f} kPa  T_w {r['Tw_C']:.2f} °C  "
            f"σ {r['sigma_baseline']:.2f} → {r['sigma_min'] if r['sigma_min'] else float('nan'):.2f}  "
            f"(NPSH_a {r['NPSHa_baseline_m']:.2f} → {r['NPSHa_min_m'] if r['NPSHa_min_m'] else float('nan'):.2f} m)  "
            f"p_d {r['p_d_baseline_bar']:.4f} → {r['p_d_throttled_bar']:.4f} bar")
        log(f"    양정 H {r['head_baseline_m']:.2f} m → {r['head_deepest_m']:.2f} m "
            f"({r['head_deepest_m'] / r['head_baseline_m']:.1f}×, 정격 30 m 대비 무스로틀 "
            f"{r['head_baseline_m'] / 30.0 * 100:.1f} %) — 유량 고정 없이 곡선을 훑었다")

    # σ 스윕은 1 s 세션 511개의 cavitation_test 하나뿐 — 대조는 여기서 한다
    recs = [r for r in all_recs["cavitation_test"] if r["running"]]
    pre = [r for r in recs if r["p_s_bar"] > -0.02 and r["stamp"] < "20260902_163000"]
    post = [r for r in recs if r["p_s_bar"] > -0.02 and r["stamp"] > "20260902_163240"]
    thr = [r for r in recs if r["p_s_bar"] < THROTTLE_BAR]
    band_keys = [f"{ch}_{b}" for ch in CHANNELS for b in BANDS]
    ref = {bk: float(np.median([r[bk] for r in pre])) for bk in band_keys}
    edges = [0.6, 1.5, 2.0, 2.5, 3.0, 4.0, 4.8, 5.3, 6.0]
    result["sweep"] = {
        "run": "cavitation_test", "n_pre": len(pre), "n_post": len(post), "n_throttled": len(thr),
        "reference_band_rms": ref,
        "bins_dB": sigma_bins(thr, edges, "sigma", ref, band_keys),
        "hysteresis_dB": {bk: float(db_ratio(np.median([r[bk] for r in post]), ref[bk])) for bk in band_keys},
        "bpf_sideband": {"pre": {ch: float(np.median([r[f"{ch}_bpfmod"] for r in pre])) for ch in ("ch0", "ch6")},
                         "throttled": {ch: float(np.median([r[f"{ch}_bpfmod"] for r in thr])) for ch in ("ch0", "ch6")},
                         "natural_p95_ratio_ch0": nat["ch0_bpfmod_ratio"]["p95"]},
    }

    log("\nTier-1 대역 — σ 구간별 ΔdB (기준 = 시험 전 무스로틀 %d세션)" % len(pre))
    log("  σ구간      n  " + "".join(f"{ch}/{b:<12s}" for ch, b in TIER1))
    log("  동결 p95  ---  " + "".join(f"{p95[f'{ch}_{b}']:<16.2f}" for ch, b in TIER1))
    for row in result["sweep"]["bins_dB"]:
        log(f"  {row['sigma_lo']:.1f}-{row['sigma_hi']:.1f} {row['n']:4d}  " +
            "".join(f"{row[f'{ch}_{b}']:+8.1f}{'*' if abs(row[f'{ch}_{b}']) > p95[f'{ch}_{b}'] else ' '}       "
                    for ch, b in TIER1))
    log("  (* = 동결 자연변동 p95 초과)")

    log("\n귀속 검사 — 채널별 응답 [dB] (1–5k / 10–20k)")
    log("  σ구간      n  " + "".join(f"{ch:<14s}" for ch in CHANNELS))
    log("  위치          " + "".join(f"{CH_WHERE[ch]:<14s}" for ch in CHANNELS))
    for row in result["sweep"]["bins_dB"]:
        log(f"  {row['sigma_lo']:.1f}-{row['sigma_hi']:.1f} {row['n']:4d}  " +
            "".join(f"{row[f'{ch}_b1_5k']:+6.1f}/{row[f'{ch}_b10_20k']:+6.1f} " for ch in CHANNELS))

    h = result["sweep"]["hysteresis_dB"]
    log(f"\n복귀 이력(post−pre): ch0 10–20k {h['ch0_b10_20k']:+.1f} dB (p95 {p95['ch0_b10_20k']:.2f}), "
        f"ch0 1–5k {h['ch0_b1_5k']:+.1f}, ch6 1–5k {h['ch6_b1_5k']:+.1f}")
    sb = result["sweep"]["bpf_sideband"]
    log(f"BPF 측파대비: ch0 {sb['pre']['ch0']:.3f} → {sb['throttled']['ch0']:.3f} "
        f"(자연변동 p95 비 {sb['natural_p95_ratio_ch0']:.2f}), ch6 {sb['pre']['ch6']:.3f} → {sb['throttled']['ch6']:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"\nwrote {OUT}")
    return result


if __name__ == "__main__":
    main()
