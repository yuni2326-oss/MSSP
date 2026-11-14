"""Tier-1 합격기준의 H-측 사전확정 — "실측↔실측 자연변동"을 정상 189세션에서 산출.

사전등록 계획서 §5.1은 합격 수치를 "C_cal 내부 재현성의 95분위"로 정의한다(합성↔실측 거리가
실측↔실측 자연변동을 넘지 않아야 한다는 원칙). 그 원칙의 **정상 데이터 측**은 캐비테이션 실험
없이 지금 확정할 수 있다: 동일 구성(ch0 이동 이후) 정상 세션쌍 사이의 거리 분포가 곧
"같은 상태인데도 생기는 자연 변동"의 하한이다. 여기서 산출한 95분위는 동결 시 Tier-1 임계의
**하한(floor)** 으로 쓰고, Sweep A 취득 후 같은 σ 내 C_cal 쌍으로 재산출해 둘 중 큰 값을 채택한다.

산출 지표(계획서 §5.1과 1:1 대응):
  · 대역 에너지 log-ratio |dB| — carrier(2.9-3.4k) / 1-5k / 5-10k / 10-20k
  · ch0-ch6 coherence 곡선 RMSE
  · spectral kurtosis 곡선 RMSE
  · BPF(349Hz) 변조 깊이 비
전기 성분(PWM 콤 + 8180Hz 사이드밴드)은 npz_io.electrical_mask로 제외.

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/tier1_natural_variation.py
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import coherence, stft

from equipment_analysis.npz_io import (
    CH0_RELOCATED_FROM, amplitude_spectrum, band_rms, discover_npz_sessions, discover_running_sessions,
    electrical_mask, load_npz_raw,
)

DATA_ROOT = "/mnt/nas/data/GY_DATA/pump data"
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "tier1_natural_variation.json"
N_SESSIONS = 24                  # 쌍 수 = C(24,2) = 276
BANDS = {"carrier_2p9_3p4k": (2900.0, 3400.0), "b1_5k": (1000.0, 5000.0),
         "b5_10k": (5000.0, 10000.0), "b10_20k": (10000.0, 20000.0)}
F_ROT, F_BPF = 58.2, 349.2
NPERSEG_COH = 8192               # 동결 대상 estimator 설정
NPERSEG_SK = 4096


def spectral_kurtosis(x, fs, nperseg=NPERSEG_SK):
    """STFT 크기의 4차 정규화 모멘트(주파수별) — SK 곡선."""
    f, _, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    P = np.abs(Z) ** 2
    m2 = P.mean(axis=1)
    m4 = (P ** 2).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(m2 > 0, m4 / (m2 ** 2) - 2.0, np.nan)
    return f, sk


def bpf_mod_depth(freq, mag):
    """BPF 변조 깊이 = 평균 측파대 진폭 / BPF 진폭."""
    def pk(fc, tol=8.0):
        s = (freq > fc - tol) & (freq < fc + tol)
        return float(mag[s].max()) if s.any() else 0.0
    b = pk(F_BPF)
    return (0.5 * (pk(F_BPF - F_ROT) + pk(F_BPF + F_ROT)) / b) if b > 0 else np.nan


def session_descriptors(path):
    chans, fs, _ = load_npz_raw(path)
    d = {}
    for ch in ("ch0", "ch6"):
        fr, mg = amplitude_spectrum(chans[ch], fs)
        keep = electrical_mask(fr)
        for name, band in BANDS.items():
            d[f"{ch}_{name}"] = band_rms(fr, mg, band, keep)
        d[f"{ch}_bpfmod"] = bpf_mod_depth(fr, mg)
        fsk, sk = spectral_kurtosis(chans[ch], fs)
        m = (fsk >= 1000) & (fsk <= 20000)
        d[f"{ch}_sk"] = sk[m]
    f_c, Cxy = coherence(chans["ch0"], chans["ch6"], fs=fs, nperseg=NPERSEG_COH)
    m = (f_c >= 1000) & (f_c <= 20000)
    d["coh"] = Cxy[m]
    return d


def main():
    sess = discover_running_sessions(DATA_ROOT, date_from=CH0_RELOCATED_FROM)
    step = max(1, len(sess) // N_SESSIONS)
    use = sess[::step][:N_SESSIONS]
    print(f"[H-side] {len(use)} normal sessions sampled from {len(sess)} in-configuration "
          f"({use[0].stamp} .. {use[-1].stamp}) -> {len(use)*(len(use)-1)//2} pairs", flush=True)

    desc = []
    for i, s in enumerate(use):
        desc.append(session_descriptors(s.path))
        if (i + 1) % 8 == 0:
            print(f"   ... {i+1}/{len(use)}", flush=True)

    dist = {k: [] for k in
            [f"{ch}_{b}_logratio_dB" for ch in ("ch0", "ch6") for b in BANDS]
            + [f"{ch}_sk_rmse" for ch in ("ch0", "ch6")]
            + [f"{ch}_bpfmod_ratio" for ch in ("ch0", "ch6")] + ["coh_rmse"]}
    for a, b in itertools.combinations(range(len(desc)), 2):
        A, B = desc[a], desc[b]
        for ch in ("ch0", "ch6"):
            for band in BANDS:
                va, vb = A[f"{ch}_{band}"], B[f"{ch}_{band}"]
                if va > 0 and vb > 0:
                    dist[f"{ch}_{band}_logratio_dB"].append(abs(20 * np.log10(va / vb)))
            ska, skb = A[f"{ch}_sk"], B[f"{ch}_sk"]
            ok = np.isfinite(ska) & np.isfinite(skb)
            if ok.any():
                dist[f"{ch}_sk_rmse"].append(float(np.sqrt(np.mean((ska[ok] - skb[ok]) ** 2))))
            ra, rb = A[f"{ch}_bpfmod"], B[f"{ch}_bpfmod"]
            if np.isfinite(ra) and np.isfinite(rb) and rb > 0:
                r = ra / rb
                dist[f"{ch}_bpfmod_ratio"].append(max(r, 1.0 / r))     # 1 이상으로 정규화
        dist["coh_rmse"].append(float(np.sqrt(np.mean((A["coh"] - B["coh"]) ** 2))))

    print("\n=== H-side natural variation between same-configuration normal sessions ===", flush=True)
    print(f"{'metric':32s} {'median':>9s} {'p90':>9s} {'p95':>9s} {'max':>9s}", flush=True)
    stats = {}
    for k, v in dist.items():
        arr = np.array(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        stats[k] = {"n_pairs": int(arr.size), "median": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95)),
                    "max": float(arr.max())}
        print(f"{k:32s} {stats[k]['median']:9.4f} {stats[k]['p90']:9.4f} "
              f"{stats[k]['p95']:9.4f} {stats[k]['max']:9.4f}", flush=True)

    print("\n=== Frozen Tier-1 acceptance FLOORS (H-side p95; final = max(this, C_cal p95)) ===", flush=True)
    for k in ("ch0_carrier_2p9_3p4k_logratio_dB", "ch0_b1_5k_logratio_dB", "ch0_b5_10k_logratio_dB",
              "ch6_b1_5k_logratio_dB", "coh_rmse", "ch0_sk_rmse", "ch0_bpfmod_ratio"):
        if k in stats:
            print(f"   {k:32s} <= {stats[k]['p95']:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": {"channels": ["ch0", "ch6"], "format": "npz",
                  "config_from": CH0_RELOCATED_FROM, "n_sessions_used": len(use),
                  "n_pairs": len(use) * (len(use) - 1) // 2,
                  "first": use[0].stamp, "last": use[-1].stamp},
        "estimator_settings": {"coherence_nperseg": NPERSEG_COH, "sk_nperseg": NPERSEG_SK,
                               "sk_overlap": 0.5, "window": "hann",
                               "analysis_band_hz": [1000, 20000],
                               "electrical_mask": "PWM comb 7820x{1,2,3} +-300k Hz, plus 8180+-40 Hz"},
        "bands_hz": {k: list(v) for k, v in BANDS.items()},
        "natural_variation": stats,
    }, indent=2))
    print(f"\nSaved -> {OUT}\n=== TIER1_NATURAL_VARIATION DONE ===", flush=True)


if __name__ == "__main__":
    main()
