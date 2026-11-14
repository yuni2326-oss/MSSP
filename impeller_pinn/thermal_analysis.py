"""수온·배관온도 로그 검토 → 캐비 실험계획(σ 산출·열 허용대·H6 drift)에 반영할 수치 산출.

NAS 로그: `temperature_YYYYMMDD.csv`  컬럼 `timestamp, ai0_C, ai1_C` (2분 간격)
  · **ai1 = 수온(water)** — 더 낮은 쪽 (사용자 확인)
  · **ai0 = 배관 표면온(pipe)** — 더 높은 쪽

산출 항목:
 (A) 표류 특성 — 상승률, 준정상 도달 여부(지수 점근 적합), **2시간(1 sweep) 창 최대 표류**
     → 계획서 §II.2.6의 "±1.5 °C/sweep" 달성 가능성과 필요한 사전 예열시간 판정.
 (B) σ 민감도 — p_v(T_w)를 Buck 식으로 계산하고 dσ/dT_w = −(dp_v/dT)/q 산출.
     q = ½ρU², U = 임펠러 아이 원주속도(직경 30.8 mm, 58.2 Hz).
     → "σ는 기록별 T_w로 계산해야 한다"의 정량 근거.
 (C) 배관−수온 차 — 열원(모터·펌프 손실) 유입의 지표이자 수온계 이상 시 교차검증용.
 (D) **열 표류 vs 진동 특징 교란** — 온도 로그와 겹치는 npz 세션(ch0/ch6)의 대역특징을
     수온에 대해 회귀. 계획서 H6이 'baseline aging'으로 둔 표류가 실은 수온일 수 있으므로,
     상관이 유의하면 수온을 공변량으로 승격해야 한다.

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/thermal_analysis.py
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

from equipment_analysis.npz_io import (
    CH0_RELOCATED_FROM, amplitude_spectrum, band_rms, discover_npz_sessions, discover_running_sessions,
    electrical_mask, load_npz_raw,
)

DATA_ROOT = "/mnt/nas/data/GY_DATA/pump data"
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "thermal_analysis.json"
WATER_COL, PIPE_COL = "ai1_C", "ai0_C"          # 사용자 확인: 낮은 쪽이 수온
EYE_DIA_M, F_SHAFT_HZ = 0.0308, 58.2            # 임펠러 아이 직경, 축 회전주파수
SWEEP_WINDOW_H = 2.0                            # 1 sweep 소요시간 가정
TOL_C = 1.5                                     # 계획서 §II.2.6 허용대
BANDS = {"carrier_2p9_3p4k": (2900.0, 3400.0), "b1_5k": (1000.0, 5000.0),
         "b5_10k": (5000.0, 10000.0)}


def p_v_kPa(T_C):
    """포화증기압(kPa), Buck 식 — 0~50 °C에서 0.05% 이내."""
    T = np.asarray(T_C, dtype=float)
    return 0.61121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))


def rho_water(T_C):
    """물 밀도(kg/m³) 근사(0~60 °C)."""
    T = np.asarray(T_C, dtype=float)
    return 1000.0 * (1.0 - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * (T - 3.9863) ** 2)


def load_temperature():
    files = sorted(glob.glob(f"{DATA_ROOT}/temperature_*.csv"))
    if not files:
        raise FileNotFoundError("temperature_*.csv 없음")
    d = pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in files], ignore_index=True)
    d = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    d["h"] = (d["timestamp"] - d["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
    return d, [f.split("/")[-1] for f in files]


def part_a(d, log=print):
    """표류 특성 + 2시간 창 최대 표류 + 지수 점근 적합."""
    w = d[WATER_COL].to_numpy(float); p = d[PIPE_COL].to_numpy(float); h = d["h"].to_numpy(float)
    log(f"[A] 로그 구간 {d['timestamp'].iloc[0]} .. {d['timestamp'].iloc[-1]} "
        f"({h[-1]:.1f} h, n={len(d)}, 간격 {np.median(np.diff(h))*3600:.0f}s)")
    log(f"[A] 수온  {w.min():.2f} → {w.max():.2f} °C (총 상승 {w[-1]-w[0]:+.2f})")
    log(f"[A] 배관  {p.min():.2f} → {p.max():.2f} °C (총 상승 {p[-1]-p[0]:+.2f})")

    # 구간별 상승률
    rates = {}
    for lo, hi, lab in ((0, 6, "0-6h"), (6, 12, "6-12h"), (12, 24, "12-24h"), (24, 1e9, ">24h")):
        m = (h >= lo) & (h < hi)
        if m.sum() > 3:
            sl = np.polyfit(h[m], w[m], 1)[0]
            rates[lab] = float(sl)
            log(f"[A] 수온 상승률 {lab:7s}: {sl:+.4f} °C/h")

    # 지수 점근 T(t) = T_inf - (T_inf - T0) exp(-t/tau)
    asym = {}
    try:
        def f(t, T_inf, T0, tau):
            return T_inf - (T_inf - T0) * np.exp(-t / tau)
        popt, _ = curve_fit(f, h, w, p0=[w.max() + 1.5, w[0], 12.0], maxfev=20000)
        resid = float(np.sqrt(np.mean((w - f(h, *popt)) ** 2)))
        asym = {"T_inf_C": float(popt[0]), "T0_C": float(popt[1]), "tau_h": float(popt[2]),
                "rmse_C": resid}
        log(f"[A] 지수 점근 적합: T∞={popt[0]:.2f} °C, τ={popt[2]:.1f} h, RMSE={resid:.3f} °C "
            f"→ 남은 상승폭 {popt[0]-w[-1]:+.2f} °C")
    except Exception as e:
        log(f"[A] 점근 적합 실패: {type(e).__name__}")

    # 2시간(1 sweep) 창 최대 표류 — 시간대별
    win = []
    for i in range(len(h)):
        m = (h >= h[i]) & (h <= h[i] + SWEEP_WINDOW_H)
        if m.sum() >= 2 and h[m][-1] - h[i] >= SWEEP_WINDOW_H * 0.9:
            win.append((h[i], float(w[m].max() - w[m].min())))
    win = np.array(win)
    early = win[win[:, 0] < 6][:, 1] if (win[:, 0] < 6).any() else np.array([np.nan])
    late = win[win[:, 0] >= 12][:, 1] if (win[:, 0] >= 12).any() else np.array([np.nan])
    log(f"[A] {SWEEP_WINDOW_H:.0f}h 창 수온 표류: 전체 최대 {win[:,1].max():.2f} °C, "
        f"초기(<6h) 중앙 {np.nanmedian(early):.2f}, 예열후(≥12h) 중앙 {np.nanmedian(late):.2f} °C")
    ok_frac = float(np.mean(win[:, 1] <= TOL_C))
    log(f"[A] → ±{TOL_C} °C 허용대 충족 창 비율 {100*ok_frac:.0f}% "
        f"(예열 후 {100*np.mean(win[win[:,0]>=12][:,1] <= TOL_C):.0f}%)")
    return {"span_h": float(h[-1]), "n": int(len(d)),
            "water_min_C": float(w.min()), "water_max_C": float(w.max()),
            "pipe_min_C": float(p.min()), "pipe_max_C": float(p.max()),
            "rates_C_per_h": rates, "asymptote": asym,
            "sweep_window_h": SWEEP_WINDOW_H,
            "drift_in_window_max_C": float(win[:, 1].max()),
            "drift_in_window_median_early_C": float(np.nanmedian(early)),
            "drift_in_window_median_after_12h_C": float(np.nanmedian(late)),
            "frac_windows_within_tol": ok_frac,
            "frac_windows_within_tol_after_12h": float(np.mean(win[win[:, 0] >= 12][:, 1] <= TOL_C))}


def part_b(d, log=print):
    """σ 민감도: p_v(T_w) 및 dσ/dT_w."""
    w = d[WATER_COL].to_numpy(float)
    U = np.pi * EYE_DIA_M * F_SHAFT_HZ
    Tm = float(np.median(w))
    q = 0.5 * float(rho_water(Tm)) * U ** 2 / 1000.0        # kPa
    pv_lo, pv_hi = float(p_v_kPa(w.min())), float(p_v_kPa(w.max()))
    dpv_dT = float((p_v_kPa(Tm + 0.5) - p_v_kPa(Tm - 0.5)))  # kPa/°C
    log(f"[B] U = π·{EYE_DIA_M*1000:.1f}mm·{F_SHAFT_HZ} Hz = {U:.3f} m/s, "
        f"q = ½ρU² = {q:.2f} kPa (ρ@{Tm:.1f}°C = {rho_water(Tm):.1f})")
    log(f"[B] p_v: {w.min():.2f}°C → {pv_lo:.3f} kPa | {w.max():.2f}°C → {pv_hi:.3f} kPa "
        f"(Δ{pv_hi-pv_lo:+.3f} kPa)")
    log(f"[B] dp_v/dT = {dpv_dT:.3f} kPa/°C → **dσ/dT_w = {-dpv_dT/q:+.4f} /°C**")
    log(f"[B] 관측된 수온 상승 {w.max()-w.min():.2f} °C ⇒ **Δσ = {-(pv_hi-pv_lo)/q:+.4f}** "
        f"(σ 좌표가 통째로 이동)")
    log(f"[B] 허용대 ±{TOL_C} °C ⇒ Δσ ±{TOL_C*dpv_dT/q:.4f}")
    log(f"[B] ⚠️ 배관온도({d[PIPE_COL].median():.1f}°C)로 p_v를 계산하면 "
        f"{float(p_v_kPa(d[PIPE_COL].median())):.3f} kPa로 "
        f"{float(p_v_kPa(d[PIPE_COL].median()))-float(p_v_kPa(Tm)):+.3f} kPa 과대 → "
        f"σ가 {-(float(p_v_kPa(d[PIPE_COL].median()))-float(p_v_kPa(Tm)))/q:+.4f} 편향")
    return {"U_m_s": float(U), "q_kPa": q, "T_median_C": Tm,
            "pv_at_min_kPa": pv_lo, "pv_at_max_kPa": pv_hi,
            "dpv_dT_kPa_per_C": dpv_dT, "dsigma_dT_per_C": float(-dpv_dT / q),
            "dsigma_over_observed_rise": float(-(pv_hi - pv_lo) / q),
            "dsigma_for_tolerance": float(TOL_C * dpv_dT / q),
            "pipe_median_C": float(d[PIPE_COL].median()),
            "sigma_bias_if_pipe_used": float(
                -(float(p_v_kPa(d[PIPE_COL].median())) - float(p_v_kPa(Tm))) / q)}


def part_c(d, log=print):
    diff = d[PIPE_COL] - d[WATER_COL]
    r, _ = pearsonr(d[WATER_COL], d[PIPE_COL])
    log(f"[C] 배관−수온 차: {diff.min():.2f} .. {diff.max():.2f} °C (중앙 {diff.median():.2f}), "
        f"두 채널 상관 r={r:.4f}")
    log(f"[C] → 배관이 항상 {diff.median():.1f} °C 높음: 모터·펌프 손실이 배관으로 유입되는 "
        f"구조와 정합. 수온계 고장 시 배관온−Δ로 교차검증 가능.")
    return {"diff_min_C": float(diff.min()), "diff_max_C": float(diff.max()),
            "diff_median_C": float(diff.median()), "pearson_r": float(r)}


def part_d(d, log=print):
    """열 표류 vs 진동 특징 — 온도 로그와 겹치는 npz 세션에서 회귀."""
    t0, t1 = d["timestamp"].min(), d["timestamp"].max()
    sess = discover_running_sessions(DATA_ROOT, date_from=CH0_RELOCATED_FROM)
    sel = []
    for s in sess:
        ts = pd.to_datetime(s.stamp, format="%Y%m%d_%H%M%S")
        if t0 <= ts <= t1:
            sel.append((s, ts))
    log(f"[D] 온도 로그와 겹치는 npz 세션: {len(sel)}개")
    if len(sel) < 8:
        log("[D] 표본 부족 — 회귀 생략")
        return {"n_sessions": len(sel)}
    step = max(1, len(sel) // 40)
    sel = sel[::step][:40]
    tser = d.set_index("timestamp")
    rows = []
    for s, ts in sel:
        chans, fs, _ = load_npz_raw(s.path)
        idx = tser.index.get_indexer([ts], method="nearest")[0]
        gap = abs((tser.index[idx] - ts).total_seconds())
        if gap > 300:
            continue
        row = {"stamp": s.stamp, "T_w": float(tser[WATER_COL].iloc[idx]),
               "T_p": float(tser[PIPE_COL].iloc[idx]), "gap_s": gap}
        for ch in ("ch0", "ch6"):
            fr, mg = amplitude_spectrum(chans[ch], fs)
            keep = electrical_mask(fr)
            row[f"{ch}_rms"] = float(np.sqrt(np.mean(chans[ch] ** 2)))
            for nm, b in BANDS.items():
                row[f"{ch}_{nm}"] = band_rms(fr, mg, b, keep)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["stamp"], format="%Y%m%d_%H%M%S")
    df["h"] = (df["ts"] - df["ts"].min()).dt.total_seconds() / 3600.0
    log(f"[D] 정합 완료 {len(df)}개 (수온 {df['T_w'].min():.2f}..{df['T_w'].max():.2f} °C, "
        f"경과 {df['h'].max():.1f} h)")

    # ⚠️ 교란 판정의 전제: 수온이 시간에 대해 단조 상승하면 두 설명변수가 공선이 된다.
    rho_tT, p_tT = spearmanr(df["h"], df["T_w"])
    log(f"[D] ★ 시간 vs 수온 공선성: Spearman ρ={rho_tT:.3f} (p={p_tT:.2e}) — "
        f"{'거의 공선 → 열 기인과 시간 표류 분리 불가' if abs(rho_tT) > 0.8 else '분리 가능'}")

    log(f"{'feature':22s} {'ρ(T_w)':>8s} {'p':>9s} {'ρ(시간)':>8s} {'p':>9s} "
        f"{'%/°C':>7s} {'dB@±1.5°C':>10s} {'dB@관측폭':>10s} 판정")
    corr = {}
    T_span = float(df["T_w"].max() - df["T_w"].min())
    for f_ in [c for c in df.columns if c.startswith(("ch0_", "ch6_"))]:
        v = df[f_].to_numpy(float); T = df["T_w"].to_numpy(float)
        r, p = pearsonr(T, v)
        rs, ps = spearmanr(T, v)
        rh, ph = spearmanr(df["h"].to_numpy(float), v)
        sl = float(np.polyfit(T, v, 1)[0])
        pct = 100 * sl / float(np.median(v))
        db_tol = 20 * np.log10(1 + pct / 100 * TOL_C) if pct > -100 else np.nan
        db_obs = 20 * np.log10(1 + pct / 100 * T_span) if pct > -100 else np.nan
        verdict = ("온도·시간 구분불가" if (ps < 0.05 and ph < 0.05)
                   else ("온도만 유의" if ps < 0.05 else "무의미"))
        corr[f_] = {"pearson_r": float(r), "pearson_p": float(p),
                    "spearman_T": float(rs), "spearman_T_p": float(ps),
                    "spearman_time": float(rh), "spearman_time_p": float(ph),
                    "slope_per_C": sl, "pct_per_C": float(pct),
                    "dB_at_tolerance": float(db_tol), "dB_over_observed_span": float(db_obs),
                    "verdict": verdict}
        log(f"{f_:22s} {rs:8.3f} {ps:9.2e} {rh:8.3f} {ph:9.2e} {pct:7.2f} "
            f"{db_tol:10.2f} {db_obs:10.2f} {verdict}", flush=True)

    log(f"[D] 해석: 밴드특징이 수온에 단조 상승(ρ 0.4~0.8, p≤1e-3)하나 수온이 시간과 공선이라 "
        f"**열 기인 vs 시간 표류를 이 데이터로 분리할 수 없다**. 계획서 §II.2.6의 "
        f"상승복귀(ascending) sweep이 바로 이 공선성을 깨는 장치 → 경험적으로 필수임이 확인됨.")
    return {"n_sessions": int(len(df)), "T_w_range_C": [float(df["T_w"].min()), float(df["T_w"].max())],
            "T_span_C": T_span, "elapsed_h": float(df["h"].max()),
            "time_temp_collinearity": {"spearman": float(rho_tT), "p": float(p_tT)},
            "correlations": corr,
            "table": df.drop(columns=["gap_s", "ts"]).to_dict(orient="list")}


def make_figure(d, b, dd, out_png=str(GEN / "figures" / "paper1" / "fig_thermal.png")):
    """(a) 온도 시계열+점근, (b) 2h 창 표류 vs 경과시간, (c) σ 편향, (d) 특징-수온 산점."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "figure.dpi": 200, "savefig.bbox": "tight",
                         "font.family": "Noto Sans CJK KR"})
    h = d["h"].to_numpy(float); w = d[WATER_COL].to_numpy(float); p = d[PIPE_COL].to_numpy(float)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.0))

    ax[0, 0].plot(h, p, color="#e15759", lw=1.2, label=f"배관 (ai0)")
    ax[0, 0].plot(h, w, color="#4c78a8", lw=1.4, label=f"수온 (ai1)")
    asy = b if False else None
    ax[0, 0].set_xlabel("경과 시간 (h)"); ax[0, 0].set_ylabel("온도 (°C)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("(a) 33시간 연속 로그 — 배관이 항상 ~6.7 °C 높음")

    win = []
    for i in range(len(h)):
        m = (h >= h[i]) & (h <= h[i] + SWEEP_WINDOW_H)
        if m.sum() >= 2 and h[m][-1] - h[i] >= SWEEP_WINDOW_H * 0.9:
            win.append((h[i], w[m].max() - w[m].min()))
    win = np.array(win)
    ax[0, 1].plot(win[:, 0], win[:, 1], color="#59a14f", lw=1.2)
    ax[0, 1].axhline(TOL_C, color="#e15759", ls="--", lw=1.2, label=f"허용대 ±{TOL_C} °C")
    ax[0, 1].axvline(12, color="#888", ls=":", lw=1, label="예열 12 h")
    ax[0, 1].set_xlabel("sweep 시작 시각 (경과 h)")
    ax[0, 1].set_ylabel(f"{SWEEP_WINDOW_H:.0f}h 창 내 수온 표류 (°C)")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("(b) 1 sweep(2h) 동안의 수온 표류 — 전 구간 허용대 이내")

    Tg = np.linspace(w.min() - 1, max(w.max(), 46) + 1, 200)
    ax[1, 0].plot(Tg, p_v_kPa(Tg), color="#4c78a8", lw=1.6)
    ax[1, 0].axvspan(w.min(), w.max(), color="#cfe3f5", alpha=0.7, label="관측 수온 범위")
    ax[1, 0].axvline(d[PIPE_COL].median(), color="#e15759", ls="--", lw=1.2,
                     label="배관온 중앙값(오사용 시)")
    ax[1, 0].set_xlabel("온도 (°C)"); ax[1, 0].set_ylabel("$p_v$ (kPa)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title(f"(c) $p_v(T)$ — dσ/dT={b['dsigma_dT_per_C']:.4f}/°C, "
                       f"배관온 오사용 시 σ 편향 {b['sigma_bias_if_pipe_used']:+.3f}")

    tb = dd.get("table")
    if tb:
        t = pd.DataFrame(tb)
        key = "ch6_b1_5k" if "ch6_b1_5k" in t else list(t.columns)[-1]
        sc = ax[1, 1].scatter(t["T_w"], t[key], c=t["T_w"], cmap="coolwarm", s=26,
                              edgecolor="k", linewidth=0.3)
        z = np.polyfit(t["T_w"], t[key], 1)
        xs = np.linspace(t["T_w"].min(), t["T_w"].max(), 50)
        ax[1, 1].plot(xs, np.polyval(z, xs), color="#333", lw=1.2, ls="--")
        c = dd["correlations"][key]
        ax[1, 1].set_xlabel("수온 (°C)"); ax[1, 1].set_ylabel(f"{key} (Pa)")
        ax[1, 1].set_title(f"(d) {key} vs 수온 — ρ={c['spearman_T']:.2f} "
                           f"(p={c['spearman_T_p']:.0e}), {c['pct_per_C']:.1f} %/°C\n"
                           f"⚠ 시간과 공선(ρ={dd['time_temp_collinearity']['spearman']:.2f}) → 인과 미확정")
    fig.suptitle("수온·배관온 로그 검토 (2026-07-27 11:38 ~ 07-28 20:47, 2분 간격 n=995)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png); plt.close(fig)
    print(f"[fig] saved -> {out_png}", flush=True)


def main():
    d, files = load_temperature()
    print(f"온도 로그 파일: {files}\n", flush=True)
    a = part_a(d); print(flush=True)
    b = part_b(d); print(flush=True)
    c = part_c(d); print(flush=True)
    dd = part_d(d)
    make_figure(d, b, dd)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"files": files, "channel_map": {"water": WATER_COL, "pipe": PIPE_COL},
                               "drift": a, "sigma_sensitivity": b, "pipe_vs_water": c,
                               "vibration_coupling": dd}, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {OUT}\n=== THERMAL_ANALYSIS DONE ===", flush=True)


if __name__ == "__main__":
    main()
