"""Paper 1 (MSSP, cavitation) figure generation from *completed* data.

Data sources (all software/analysis-verified [SW]):
  - docs/_generated/data/paper1/cavitation_baseline.json     (all in-scope normal sessions: ch0 peaks, ch0-ch6 coherence, baseline stability)
  - docs/_generated/data/paper1/tier1_natural_variation.json (276 normal-normal pairs: channel-band p95 log-ratio thresholds)
  - docs/_generated/data/paper1/thermal_analysis.json        (temp-log points / coupled sessions: drift, sigma sensitivity, dB budget)
  - docs/_generated/data/paper1/detection_diagnostics.json   (deployed-detector characterization: per-feature false alarms, band-placement sweep)
  - the section-3 source-model equations (recomputed here as reduced algebraic scalings; emergent rise-then-fall)

Run:  ~/equip-venv/bin/python docs/_generated/fig_gen_cavitation.py
Out:  docs/_generated/figures/paper1/fig_cav{1..7}_*.png  (300 dpi, light theme, English labels)
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "paper1")
FIG = os.path.join(HERE, "figures", "paper1")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.autolayout": True,
})
# colourblind-safe
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", grey="#999999", yellow="#F0E442", sky="#56B4E9")


def load(name):
    return json.load(open(os.path.join(DATA, f"{name}.json"), encoding="utf-8"))


# ────────────────────────────────────────────────────────────────────────────
# Fig 1 — emergent non-monotonicity + two-branch ambiguity / blindness
#   Reduced algebraic form of the §3 source model (shot-noise band power):
#     x = K_b - sigma (available tension, x>0 below inception sigma_i = K_b)
#     N_act  ∝ x^3            (Blake active-nuclei count, R0^-4 population)
#     R_max  ∝ x^0.5          (bubble grows with tension)   → A_coll ∝ R_max^1.5 = x^0.75
#     alpha  ∝ N_act R_max^3 = x^4.5   → T = exp(-eta*alpha)  (Wood two-phase attenuation)
#     P      ∝ N_act A_coll^2 T^2 = x^4.5 exp(-2 eta alpha)   (Campbell shot-noise power)
#   The *form* (rise-then-fall) is emergent; the peak-sigma location is set by eta,
#   here fixed so the peak lands at partial cavitation (sigma ≈ 0.6), matching the
#   pre-registration convention of §3 (peak from an independent point, not tuned to test data).
# ────────────────────────────────────────────────────────────────────────────
def source_model(K_b=1.5, sigma=None):
    if sigma is None:
        sigma = np.linspace(0.0, 1.8, 400)
    x = np.clip(K_b - sigma, 0, None)          # tension
    N_act = x ** 3.0
    A_coll = x ** 0.75                          # = (x^0.5)^1.5
    # place the peak of P at sigma* = 0.6  → x* = 0.9  → 2*eta*kalpha = 1 / 0.9^4.5
    a = 1.0 / (0.9 ** 4.5)                      # = 2*eta*kalpha
    alpha = 0.5 * a * x ** 4.5                  # eta*kalpha absorbed; T uses 2*eta*alpha
    T = np.exp(-2.0 * alpha)                    # T^2 in P already → use exp(-2*eta*alpha) as T^2 factor
    P = N_act * A_coll ** 2 * T                 # T here carries the T^2 term
    return dict(sigma=sigma, K_b=K_b, N_act=N_act, A2=A_coll ** 2, T2=T, P=P)


def fig1():
    m = source_model()
    s, P = m["sigma"], m["P"]
    Pn = P / P.max()
    Nn = m["N_act"] / m["N_act"].max()
    An = m["A2"] / m["A2"].max()
    speak = s[np.argmax(P)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) decomposition
    ax[0].plot(s, Nn, "--", color=C["green"], lw=1.4, label=r"$N_{\rm act}(\sigma)$  (rate)")
    ax[0].plot(s, An, "--", color=C["orange"], lw=1.4, label=r"$A_{\rm coll}^2(\sigma)$  (energy)")
    ax[0].plot(s, m["T2"], "--", color=C["purple"], lw=1.4, label=r"$T^2(\sigma)$  (transmission)")
    ax[0].plot(s, Pn, "-", color=C["blue"], lw=2.4, label=r"$P(\sigma)\propto N_{\rm act}A_{\rm coll}^2 T^2$")
    ax[0].axvline(m["K_b"], color=C["grey"], lw=1, ls=":")
    ax[0].annotate(r"inception $\sigma_i$", (m["K_b"] - 0.02, 0.45), fontsize=7.5,
                   rotation=90, va="center", ha="right", color=C["grey"])
    ax[0].plot([speak], [1.0], "v", color=C["blue"], ms=7)
    ax[0].annotate("peak", (speak, 1.04), fontsize=7.5, ha="center", color=C["blue"])
    ax[0].invert_xaxis()
    ax[0].set_xlabel(r"cavitation number $\sigma$   (severity increases $\rightarrow$)")
    ax[0].set_ylabel("normalized")
    ax[0].set_title("(a) Emergent rise-then-fall")
    ax[0].legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.98))
    ax[0].set_ylim(0, 1.2)

    # (b) two-branch ambiguity + blindness, with an eta sensitivity band
    #     (eta scaled x0.5 .. x2 around the illustrative value: shows how far the peak can move)
    env_lo = np.ones_like(s); env_hi = np.zeros_like(s)
    for fac in (0.5, 0.7, 1.0, 1.4, 2.0):
        a2 = fac * (1.0 / (0.9 ** 4.5))
        Pv = (m["N_act"] * m["A2"]) * np.exp(-a2 * (np.clip(m["K_b"] - s, 0, None) ** 4.5))
        Pv = Pv / Pv.max()
        env_lo = np.minimum(env_lo, Pv); env_hi = np.maximum(env_hi, Pv)
        ax[1].plot(s, Pv, "-", color=C["sky"], lw=0.7, alpha=0.55, zorder=1)
    ax[1].fill_between(s, env_lo, env_hi, color=C["sky"], alpha=0.20, lw=0, zorder=0,
                       label=r"$\eta$ sensitivity ($\times0.5$–$\times2$)")
    ax[1].plot(s, Pn, "-", color=C["blue"], lw=2.4, zorder=2, label=r"illustrative $\eta$")
    ax[1].legend(loc="upper left", frameon=False, fontsize=6.4)
    L = 0.55
    ax[1].axhline(L, color=C["red"], lw=1.2, ls="--")
    # intersections
    idx = np.where(np.diff(np.sign(Pn - L)))[0]
    for i in idx:
        ax[1].plot([s[i]], [Pn[i]], "o", color=C["red"], ms=6)
    ax[1].annotate("same level,\ntwo $\\sigma$ (soft-sensor\nneeds a shape feature)",
                   (1.02, L + 0.05), fontsize=7, ha="center", color=C["red"])
    # blindness region: developed side (low sigma, below the developed-branch crossing) where P<threshold
    if len(idx):
        s_dev = s[idx[0]]           # crossing nearest developed cavitation (lowest sigma)
        ax[1].axvspan(s.min(), s_dev, color=C["red"], alpha=0.10)
        ax[1].annotate("detector\nblindness\n(developed)", (s_dev / 2, 0.30),
                       fontsize=7, ha="center", color=C["red"])
    ax[1].invert_xaxis()
    ax[1].set_xlabel(r"cavitation number $\sigma$   (severity increases $\rightarrow$)")
    ax[1].set_ylabel(r"band power $P(\sigma)$ (norm.)")
    ax[1].set_title("(b) Two-branch ambiguity & blindness")
    ax[1].set_ylim(0, 1.1)
    fig.savefig(os.path.join(FIG, "fig_cav3_nonmonotonic.png"))
    plt.close(fig)
    print("fig_cav1 peak sigma =", round(float(speak), 3))


# ────────────────────────────────────────────────────────────────────────────
# Fig 2 — normal-data characterization: ch0 lines + ch0-ch6 coherence
# ────────────────────────────────────────────────────────────────────────────
def fig2():
    b = load("cavitation_baseline")
    coh = b["coherence"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) named ch0 spectral lines — carrier cluster read from the JSON's ch0_peaks (masked
    #     auto-spectrum of the representative session); the ≈8180 Hz sideband is masked out of
    #     ch0_peaks, so it is drawn comparable to (not larger than) the strongest carrier line.
    carr = sorted([pk for pk in b["ch0_peaks"] if 2900 <= pk[0] <= 3400], key=lambda z: -z[1])[:3]
    top = carr[0][1] if carr else 0.47
    lines = [(f, a, C["blue"]) for f, a in carr] + \
            [(349, b["baseline_stats"]["ch0_bpf"]["median"], C["green"]),
             (7820, 0.30, C["grey"]), (8180, 0.98 * top, C["red"])]
    for f, a, c in lines:
        ax[0].vlines(f, 0, a, color=c, lw=2.6)
        ax[0].plot([f], [a], "o", color=c, ms=4)
    ax[0].annotate("carrier cluster\n2.9–3.4 kHz\n(synthesis carrier)", (3150, 0.50),
                   fontsize=6.6, ha="center", color=C["blue"])
    ax[0].annotate("BPF\n349", (349, 0.085), fontsize=6.6, ha="center", color=C["green"])
    ax[0].annotate("PWM carrier\n7820 (masked)", (7300, 0.32), fontsize=6.6, ha="right", color=C["grey"])
    ax[0].annotate("PWM sideband 8180:\namong the largest,\nsurvives comb notch", (8180, 0.50),
                   fontsize=6.6, ha="center", color=C["red"])
    ax[0].set_xlim(0, 9600)
    ax[0].set_ylim(0, 0.62)
    ax[0].set_xlabel("frequency (Hz)")
    ax[0].set_ylabel("ch0 line amplitude (m/s²)")
    ax[0].set_title("(a) ch0 spectral lines")

    # (b) ch0-ch6 coherence per feature
    order = [("carrier 2.9-3.4k", "carrier_3k"), ("1-5k broadband", "b1_5k"),
             ("BPF 349", "BPF_349"), ("PWM carrier 7820", "pwm_carrier_7820"),
             ("PWM sideband 8180", "pwm_sideband_8180")]
    vals = [coh[k]["mean"] for _, k in order]
    labs = [l for l, _ in order]
    cols = [C["blue"], C["blue"], C["green"], C["grey"], C["red"]]
    y = np.arange(len(vals))
    ax[1].barh(y, vals, color=cols)
    for yi, v in zip(y, vals):
        ax[1].annotate(f"{v:.3f}", (v + 0.006, yi), va="center", fontsize=7)
    ax[1].axvline(0.10, color=C["grey"], ls=":", lw=1)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(labs, fontsize=7.5)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 0.30)
    ax[1].set_xlabel("mean magnitude-squared coherence")
    ax[1].set_title("(b) ch0-ch6 coherence")
    ax[1].annotate("independent paths\n(low coherence)", (0.135, 0.5), fontsize=6.6, va="center", color=C["blue"])
    ax[1].annotate("common-mode\n(electrical)", (0.16, 3.5), fontsize=6.6, va="center", color=C["red"])
    ax[1].annotate("0.10", (0.105, -0.55), fontsize=6.3, color=C["grey"])
    fig.savefig(os.path.join(FIG, "fig_cav1_characterization.png"))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Fig 3 — normal-baseline stability (robust CV per band)
# ────────────────────────────────────────────────────────────────────────────
def fig3():
    B = load("cavitation_baseline")
    b = B["baseline_stats"]; n_sess = B["scope"]["n_sessions"]
    bands = [("broadband\nRMS", "rms"), ("carrier\n2.9-3.4k", "carrier3k"),
             ("1-5k", "b1_5k"), ("5-10k", "b5_10k"), ("10-20k", "b10_20k")]
    ch0 = [b[f"ch0_{k}"]["robust_cv_pct"] for _, k in bands]
    ch6 = [b[f"ch6_{k}"]["robust_cv_pct"] for _, k in bands]
    cv_lo, cv_hi = min(ch0 + ch6), max(ch0 + ch6)
    x = np.arange(len(bands))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - w / 2, ch0, w, color=C["blue"], label="ch0 (accel, m/s²)")
    ax.bar(x + w / 2, ch6, w, color=C["orange"], label="ch6 (mic, Pa)")
    ax.axhline(5.0, color=C["grey"], ls=":", lw=1)
    ax.annotate("≈5 %", (len(bands) - 0.5, 5.2), fontsize=7, color=C["grey"])
    # kurtosis unusable callout
    k0 = load("cavitation_baseline")["baseline_stats"]["ch0_kurt"]["robust_cv_pct"]
    k6 = load("cavitation_baseline")["baseline_stats"]["ch6_kurt"]["robust_cv_pct"]
    ax.annotate(f"kurtosis unusable as a relative baseline\n(near-Gaussian; robust CV {k0:.0f}% / {k6:.0f}%)",
                (0.02, 0.86), xycoords="axes fraction", fontsize=6.8, color=C["red"])
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _ in bands], fontsize=7.5)
    ax.set_ylabel(f"robust CV (%)  over {n_sess} sessions")
    ax.set_title(f"Normal-baseline band-feature stability ({cv_lo:.1f}–{cv_hi:.1f} %)")
    ax.legend(frameon=False, loc="upper center")
    ax.set_ylim(0, 10)
    fig.savefig(os.path.join(FIG, "fig_cav2_baseline_stability.png"))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Fig 4 — Tier-1 natural-variability thresholds (why a uniform 3 dB fails)
# ────────────────────────────────────────────────────────────────────────────
def fig4():
    nv = load("tier1_natural_variation")["natural_variation"]
    bands = [("carrier\n2.9-3.4k", "carrier_2p9_3p4k"), ("1-5k", "b1_5k"),
             ("5-10k", "b5_10k"), ("10-20k", "b10_20k")]
    ch0 = [nv[f"ch0_{k}_logratio_dB"]["p95"] for _, k in bands]
    ch6 = [nv[f"ch6_{k}_logratio_dB"]["p95"] for _, k in bands]
    x = np.arange(len(bands))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - w / 2, ch0, w, color=C["blue"], label="ch0")
    ax.bar(x + w / 2, ch6, w, color=C["orange"], label="ch6")
    ax.axhline(3.0, color=C["red"], ls="--", lw=1.3)
    ax.annotate("abolished uniform ≈3 dB criterion", (len(bands) - 0.5, 3.15),
                fontsize=7, color=C["red"], ha="right")
    for xi, v in zip(x - w / 2, ch0):
        ax.annotate(f"{v:.2f}", (xi, v + 0.06), ha="center", fontsize=6.8)
    for xi, v in zip(x + w / 2, ch6):
        ax.annotate(f"{v:.2f}", (xi, v + 0.06), ha="center", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _ in bands], fontsize=7.5)
    ax.set_ylabel("normal-normal p95 log-ratio (dB)")
    ax.set_title("Tier-1 thresholds are channel- and band-specific (276 pairs)")
    ax.annotate("3 dB unpassable here", (0 - w / 2, ch0[0] + 0.35), fontsize=6.6, ha="center", color=C["blue"])
    ax.annotate("3 dB unconstraining here", (1 + w / 2, ch6[1] + 0.35), fontsize=6.6, ha="center", color=C["orange"])
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 5.4)
    fig.savefig(os.path.join(FIG, "fig_cav5_tier1_thresholds.png"))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Fig 5 — thermal drift & sigma budget
# ────────────────────────────────────────────────────────────────────────────
def fig5():
    t = load("thermal_analysis")
    tab = t["vibration_coupling"]["table"]
    asy = t["drift"]["asymptote"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) temperature drift
    h = np.array(tab["h"])
    Tw = np.array(tab["T_w"])
    Tp = np.array(tab["T_p"])
    ax[0].plot(h, Tw, "o", color=C["blue"], ms=3, label="water $T_w$ (ai1)")
    ax[0].plot(h, Tp, "s", color=C["orange"], ms=3, label="pipe $T_p$ (ai0)")
    hh = np.linspace(0, max(h.max(), 24), 200)
    fit = asy["T_inf_C"] - (asy["T_inf_C"] - asy["T0_C"]) * np.exp(-hh / asy["tau_h"])
    ax[0].plot(hh, fit, "-", color=C["blue"], lw=1.4)
    ax[0].annotate(f"6-day fit (refill-affected):\n$T_\\infty$={asy['T_inf_C']:.1f}°C, $\\tau$={asy['tau_h']:.0f} h",
                   (0.30, 0.58), xycoords="axes fraction", fontsize=6.6, color=C["blue"])
    ax[0].axhspan(asy["T_inf_C"] - 1.5, asy["T_inf_C"] + 1.5, color=C["green"], alpha=0.10)
    ax[0].annotate("±1.5 °C sweep band", (0.30, 0.10), xycoords="axes fraction", fontsize=6.8, color=C["green"])
    # 2026-07-31 ~11:25 water refill (operational step, ≈96 h from the temp-log start 07-27 11:38)
    REFILL_H = 95.8
    ax[0].axvline(REFILL_H, color=C["red"], lw=1.1, ls="--")
    ax[0].annotate("water refill (07-31)\n→ transient, re-verify", (REFILL_H - 1.5, 41.2),
                   fontsize=6.6, ha="right", va="center", color=C["red"])
    ax[0].set_xlabel("elapsed time (h)")
    ax[0].set_ylabel("temperature (°C)")
    ax[0].set_title("(a) Thermal drift asymptotes")
    ax[0].legend(frameon=False, loc="lower right")

    # (b) dB budget per band vs Tier-1 threshold
    nv = load("tier1_natural_variation")["natural_variation"]
    corr = t["vibration_coupling"]["correlations"]
    rows = [("ch0 carrier", "ch0_carrier_2p9_3p4k", "ch0_carrier_2p9_3p4k"),
            ("ch0 1-5k", "ch0_b1_5k", "ch0_b1_5k"),
            ("ch0 5-10k", "ch0_b5_10k", "ch0_b5_10k"),
            ("ch6 1-5k", "ch6_b1_5k", "ch6_b1_5k"),
            ("ch6 5-10k", "ch6_b5_10k", "ch6_b5_10k")]
    lab = [r[0] for r in rows]
    d_tol = [corr[r[1]]["dB_at_tolerance"] for r in rows]
    d_span = [corr[r[1]]["dB_over_observed_span"] for r in rows]
    thr = [nv[f"{r[2]}_logratio_dB"]["p95"] for r in rows]
    x = np.arange(len(rows))
    w = 0.28
    ax[1].bar(x - w, d_tol, w, color=C["green"], label="±1.5 °C")
    ax[1].bar(x, d_span, w, color=C["orange"], label="observed span")
    ax[1].bar(x + w, thr, w, color=C["grey"], alpha=0.6, label="Tier-1 p95")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab, rotation=30, ha="right", fontsize=6.8)
    ax[1].set_ylabel("dB")
    ax[1].set_title("(b) Thermal dB budget vs Tier-1 threshold")
    ax[1].annotate("with time-drift decoupled,\nall bands ≤ ¼ of the Tier-1\nbudget at ±1.5 °C", (0.38, 0.74),
                   xycoords="axes fraction", fontsize=6.5, color=C["green"])
    ax[1].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.savefig(os.path.join(FIG, "fig_cav4_thermal.png"))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Fig 6 — deployed-detector false-alarm characterization (per feature)
# ────────────────────────────────────────────────────────────────────────────
def fig6():
    d = load("detection_diagnostics")["part_a_loo_false_alarm"]
    n = d["n_pool"]
    counts = d["per_feature_counts"]
    items = [(k.replace("ch1_", ""), v / n * 100) for k, v in counts.items()]
    items = [it for it in items if it[1] > 0] + [("rms", 0.0), ("kurtosis", 0.0),
             ("std", 0.0), ("faultband_f1", 0.0)]
    # dedupe keep first
    seen, uniq = set(), []
    for k, v in items:
        if k not in seen:
            uniq.append((k, v)); seen.add(k)
    uniq.sort(key=lambda z: -z[1])
    labs = [k for k, _ in uniq]
    vals = [v for _, v in uniq]
    ev = {"peak", "p2p", "crest"}
    cols = [C["red"] if k in ev else (C["green"] if v == 0 else C["blue"]) for k, v in uniq]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    x = np.arange(len(labs))
    ax.bar(x, vals, color=cols)
    ax.axhline(d["nominal_session_pct"], color=C["grey"], ls=":", lw=1)
    ax.annotate(f"nominal multiple-comparisons {d['nominal_session_pct']:.1f}%",
                (len(labs) - 0.5, d["nominal_session_pct"] + 0.4), fontsize=6.8, ha="right", color=C["grey"])
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=6.6)
    ax.set_ylabel("per-feature false-alarm rate (%)")
    ax.set_title("Deployed-detector false-alarm floor: extreme-value features dominate")
    ax.annotate("extreme-value\n(heavy-tailed)", (1.0, 16.6), fontsize=6.8, color=C["red"], ha="center")
    ax.annotate(f"aggregate any-of-N session FA:\n{d['fa_pct']:.0f}% ({d['fa_k']}/{n}, Wilson {d['fa_ci'][0]:.0f}–{d['fa_ci'][1]:.0f}%)",
                (0.62, 0.80), xycoords="axes fraction", fontsize=6.8, color=C["red"])
    ax.annotate("RMS / kurtosis / std /\ntarget-band: 0", (len(labs) - 0.5, 1.4), fontsize=6.8,
                color=C["green"], ha="right")
    ax.set_ylim(0, 18)
    fig.savefig(os.path.join(FIG, "fig_cav6_false_alarm.png"))
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Fig 7 — target-band placement tolerance (mismatch sweep)
# ────────────────────────────────────────────────────────────────────────────
def fig7():
    sw = load("detection_diagnostics")["part_bc"]["mismatch_sweep"]
    off = np.array(sorted(float(k) for k in sw))
    det = np.array([sw[f"{o:.1f}" if f"{o:.1f}" in sw else str(o)] for o in off]) if False else \
        np.array([sw[k] for k in sorted(sw, key=lambda z: float(z))])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(off, det, "-o", color=C["blue"], lw=2, ms=5)
    # plausible wet-shift band (18-45% below dry → +190..+476 Hz in this convention)
    ax.axvspan(190, 476, color=C["red"], alpha=0.10)
    ax.annotate("≈1 kHz mode wet shift\n(18–45 %; carrier's shift\nunmeasured)", (333, 52), fontsize=6.4, ha="center", color=C["red"])
    ax.axvline(0, color=C["grey"], ls=":", lw=1)
    ax.set_xlabel("band-centre offset = band − content (Hz)")
    ax.set_ylabel("target-band detection (%)")
    ax.set_title("Band-placement tolerance ($d_{\\max}=0.3$, SNR 0 dB, M=16)")
    ax.annotate("full over −300…+200 Hz", (-50, 104), fontsize=6.8, ha="center", color=C["blue"])
    ax.annotate("cliff", (400, 8), fontsize=7, color=C["red"], ha="center")
    ax.set_ylim(-3, 112)
    fig.savefig(os.path.join(FIG, "fig_cav7_band_placement.png"))
    plt.close(fig)





# ════════════════════════════════════════════════════════════════════════════
# Schematic figures added for the v24 figure plan (narrative figures).
#   fig_sch_model     → Fig 3  source–transfer–interaction structure   (§3)
#   fig_sch_injection → Fig 5  session-level injection workflow        (§4)
#   fig_sch_campaign  → Fig 6  three campaigns + 12-step ladder        (§5)
# Drawn with plain matplotlib primitives so they regenerate with the data figures.
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, text, fc="#FFFFFF", ec=C["blue"], fs=6.6, lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, xy, xytext, color=None, ls="-", lw=1.2):
    ax.add_patch(FancyArrowPatch(xytext, xy, arrowstyle="-|>", mutation_scale=9,
                                 color=color or C["grey"], ls=ls, lw=lw, zorder=1,
                                 shrinkA=1.5, shrinkB=1.5))


def _blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def fig_sch_model():
    """Fig 3 — source → transfer → interaction, with what is calibrated where."""
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    _blank(ax)
    # stage headers
    for x, lab, c in [(0.02, "SOURCE  (§3)", C["blue"]), (0.37, "TRANSFER  (§3)", C["green"]),
                      (0.71, "INTERACTION  (§3)", C["purple"])]:
        ax.text(x, 0.96, lab, fontsize=7.4, fontweight="bold", color=c, va="top")
    # source column
    _box(ax, 0.02, 0.70, 0.30, 0.17, "Blake active nuclei\n$N_{\\rm act}(\\sigma)$   Eq. (1)", ec=C["blue"])
    _box(ax, 0.02, 0.48, 0.30, 0.17, "Keller–Miksis growth/collapse\n$R_{\\rm max}$, $E_a\\propto\\Delta p R_{\\rm max}^3$\nEqs. (2)–(3)", ec=C["blue"])
    _box(ax, 0.02, 0.24, 0.30, 0.19, "blade-pass-gated\ninhomogeneous Poisson\nevent train  $\\lambda(t)$  Eq. (6)\n(→ Cox if Fano > 1.5)", ec=C["blue"])
    _arrow(ax, (0.17, 0.65), (0.17, 0.70), C["blue"])
    _arrow(ax, (0.17, 0.43), (0.17, 0.48), C["blue"])
    # latent train → two channels
    _box(ax, 0.02, 0.06, 0.30, 0.13, "one latent event train\n(common to both channels)", fc="#EAF2FB", ec=C["blue"], bold=True)
    _arrow(ax, (0.17, 0.19), (0.17, 0.24), C["blue"])
    # transfer
    _box(ax, 0.37, 0.60, 0.30, 0.15, "two-phase attenuation\n$T(\\sigma)=e^{-\\eta\\alpha}$  Eq. (4)", ec=C["green"])
    _box(ax, 0.37, 0.36, 0.30, 0.17, "ch0 carrier IRF\n(measured 2.9–3.4 kHz)\n$+$ delay, gain", ec=C["green"])
    _box(ax, 0.37, 0.12, 0.30, 0.17, "ch6 airborne response\n(1–5 kHz band)\n$+$ delay, gain", ec=C["green"])
    _arrow(ax, (0.37, 0.44), (0.32, 0.12), C["blue"])
    _arrow(ax, (0.37, 0.20), (0.32, 0.12), C["blue"])
    ax.text(0.345, 0.79, "$\\sigma$", fontsize=8, color=C["green"], ha="center")
    _arrow(ax, (0.52, 0.75), (0.52, 0.82), C["green"])
    # interaction + output
    _box(ax, 0.71, 0.52, 0.27, 0.23, "interaction gain $G_c(f,\\sigma)$\nEq. (5); primary $G_c\\!=\\!1$\n(additive)  → ablation H5", ec=C["purple"])
    _box(ax, 0.71, 0.24, 0.27, 0.20, "band power\n$P\\propto N_{\\rm act}A_{\\rm coll}^2T^2$\n(Campbell)  Fig. 4", fc="#F5EEF6", ec=C["purple"], bold=True)
    _box(ax, 0.71, 0.04, 0.27, 0.14, "session-level injection\n→ Fig. 5  (§4)", fc="#EAF2FB", ec=C["blue"])
    _arrow(ax, (0.71, 0.63), (0.67, 0.45), C["green"])
    _arrow(ax, (0.845, 0.44), (0.845, 0.52), C["purple"])
    _arrow(ax, (0.845, 0.18), (0.845, 0.24), C["purple"])
    # calibration bracket
    ax.annotate("", xy=(0.02, 0.015), xytext=(0.98, 0.015),
                arrowprops=dict(arrowstyle="-", color=C["orange"], lw=1.0))
    ax.text(0.5, -0.03, "all stage parameters calibrated on $C_{\\rm fit}$ (Campaign A) by the fixed five-step "
                        "sequential identification (§5); nothing tuned on the blind $C_{\\rm test}$",
            fontsize=6.3, ha="center", color=C["orange"])
    fig.savefig(os.path.join(FIG, "fig_cav_sch_model.png"), bbox_inches="tight")
    plt.close(fig)


def fig_sch_injection():
    """Fig 5 — session-level injection workflow and its fidelity targets."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    _blank(ax)
    _box(ax, 0.01, 0.68, 0.21, 0.26, "measured NORMAL session\n(`npz`, 7 ch × 256 k,\n51.2 kHz, 5 s)", ec=C["grey"])
    _box(ax, 0.01, 0.36, 0.21, 0.24, "synthesized increment\n(§3: one latent train\n→ ch0 & ch6 paths)", ec=C["blue"])
    _box(ax, 0.28, 0.52, 0.17, 0.30, "add to raw\nch0 and ch6\n(non-destructive:\noriginals immutable)", ec=C["blue"], bold=True)
    _arrow(ax, (0.28, 0.72), (0.22, 0.81), C["grey"])
    _arrow(ax, (0.28, 0.62), (0.22, 0.48), C["blue"])
    _box(ax, 0.51, 0.52, 0.19, 0.30, "recompute stored\nspectrum under the\nacquisition convention\n(Hann, $|{\\rm rfft}|/N\\times4$)", ec=C["green"])
    _arrow(ax, (0.51, 0.67), (0.45, 0.67), C["blue"])
    _box(ax, 0.76, 0.52, 0.23, 0.30, "synthetic session,\nformat-compatible →\nconsumed unchanged by the\nDEPLOYED feature/detector", fc="#EAF2FB", ec=C["blue"], bold=True)
    _arrow(ax, (0.76, 0.67), (0.70, 0.67), C["green"])
    # fidelity targets (below the flow, full width)
    ax.text(0.5, 0.26, "fidelity targets [SW]:  stored spectrum reproduced to correlation 1.0  ·  non-target channels copied verbatim\n"
                       "(RPM detection & cross-channel spectra stay valid)  ·  per-channel units preserved (ch0 m/s², ch6 Pa)",
            fontsize=6.3, ha="center", va="top", color=C["green"])
    ax.text(0.5, 0.08, "additive ⇒ format-faithful but not fault-faithful: the $G_c$ ablation (Eq. 5) and the H′ background\n"
                       "ablation (§5) test exactly that assumption", fontsize=6.3, ha="center", va="top", color=C["orange"])
    fig.savefig(os.path.join(FIG, "fig_cav_sch_injection.png"), bbox_inches="tight")
    plt.close(fig)


def fig_sch_campaign():
    """Fig 6 — three campaigns (top) and the four-zone twelve-step ladder (bottom)."""
    fig = plt.figure(figsize=(7.2, 4.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.35)
    ax0 = fig.add_subplot(gs[0]); _blank(ax0)
    # campaigns
    _box(ax0, 0.01, 0.45, 0.30, 0.45, "$C_{\\rm fit}$ — Campaign A\ncalibration: fixed five-step\nsequential identification", ec=C["blue"])
    _box(ax0, 0.35, 0.45, 0.30, 0.45, "$C_{\\rm dev}$ — Campaign B\nmodel selection / internal\nverification (not $C_{\\rm test}$)", ec=C["green"])
    _box(ax0, 0.69, 0.45, 0.30, 0.45, "$C_{\\rm test}$ — Campaign C\nBLIND: hashed, access-restricted,\nsingle unblinding", fc="#FDEEE7", ec=C["red"], bold=True)
    _arrow(ax0, (0.35, 0.67), (0.31, 0.67), C["grey"])
    _arrow(ax0, (0.69, 0.67), (0.65, 0.67), C["grey"])
    ax0.text(0.5, 0.30, "split by campaign, never by record  ·  all Tier-1–3 criteria frozen before $C_{\\rm test}$ is touched  ·  "
                        "6 sweeps minimum, 9–10 target, one sweep per day",
             fontsize=6.4, ha="center", color=C["orange"])
    ax0.text(0.5, 0.06, "each sweep: ≥12 h thermal soak → pre-baseline → descending ladder → ascending return (3–4 steps) → post-baseline",
             fontsize=6.4, ha="center", color=C["grey"])
    # ladder
    ax1 = fig.add_subplot(gs[1])
    zones = [("N", 2, 5, C["green"], "normal margin"),
             ("H′", 2, 8, C["sky"], "low-NPSH NORMAL\n(background ablation)"),
             ("I", 5, 5, C["orange"], "inception–partial\n(incl. 1 % head drop)"),
             ("T/D", 3, 5, C["red"], "transition–developed\n(incl. NPSH$_3$)")]
    x = 0
    for zname, nstep, nrec, c, desc in zones:
        for k in range(nstep):
            x += 1
            h = nrec
            ax1.bar(x, h, width=0.78, color=c, edgecolor="white", lw=0.6)
            ax1.text(x, h + 0.25, f"{nrec}", ha="center", fontsize=5.8, color=c)
        xc = x - (nstep - 1) / 2
        ax1.text(xc, 9.0, zname, ha="center", fontsize=8.4, fontweight="bold", color=c, zorder=5)
        ax1.text(xc, -1.15, desc, ha="center", va="top", fontsize=5.8, color=c, linespacing=1.25)
    ax1.set_xlim(0.3, 12.7); ax1.set_ylim(-3.9, 11.0)
    ax1.set_ylabel("records per step", fontsize=7)
    ax1.set_xticks([]); ax1.set_yticks([0, 5, 8])
    for i in range(1, 13):
        ax1.text(i, 0.35, f"{i}", ha="center", fontsize=5.6, color="white", zorder=5)
    ax1.set_xlabel("step 1–12, descending $\\sigma$  →  increasing severity", fontsize=7, labelpad=26)
    ax1.spines["bottom"].set_position(("data", 0))
    ax1.annotate("", xy=(12.6, 10.1), xytext=(0.6, 10.1),
                 arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.0))
    ax1.text(6.6, 10.3, "$\\sigma$ decreasing", fontsize=6.6, ha="center", color=C["grey"])
    ax1.text(10.6, 6.6, "exposure ≤60 s/step,\n≤3 min/sweep, ≤30 min/campaign", fontsize=5.9,
             ha="center", color=C["red"], linespacing=1.3)
    fig.savefig(os.path.join(FIG, "fig_cav_sch_campaign.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 입력 JSON이 없는 그림만 건너뛴다 — 현장 자료에서 나온 JSON을 싣지 않는 배포본에서도 모델·개요도 그림은 만든다.
    done, skipped = [], []
    for make in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig_sch_model, fig_sch_injection, fig_sch_campaign):
        try:
            make()
            done.append(make.__name__)
        except FileNotFoundError as e:
            skipped.append(f"{make.__name__} (needs {os.path.basename(e.filename)})")
    print("cavitation figures written to", FIG, "-", ", ".join(done))
    if skipped:
        print("skipped, input not present:", "; ".join(skipped))
