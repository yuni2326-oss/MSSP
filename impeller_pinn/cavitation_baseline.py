"""캐비테이션 논문(Draft B) 스코프 확정 분석 — ch0(임펠러 옆) + ch6(공기 마이크), npz 형식.

논문 §2를 이 두 채널·현재 계측구성으로 좁힌 근거를 산출한다:
  (A) ch0 자기스펙트럼 피크(전기성분 제외) → **합성 반송파 선택**. 기존 6채널 OMA의
      7462/8538Hz(흡입측 ch2)는 ch0에서 약함 → ch0의 지배 반송파는 ~3.0-3.4kHz 조립체 공진.
  (B) ch0-ch6 **2채널 코히런스**로 공통모드 전기 아티팩트 판별(높은 코히런스) vs 독립
      구조전달/공기전달 경로(낮은 코히런스) — 6채널 FDD가 불가한 스코프에서의 대체 검증.
  (C) 현재 구성(ch0 이동 이후) npz 세션들의 **baseline 안정성** — 논문이 주장할 수 있는
      정상모델의 산포(robust CV). 교차시대 풀링 없이 단일 구성 내에서만 계산.
  (D) BPF(349Hz) 및 ±f_rot 측파대 — 캐비 판별지표의 기저 가시성.

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/cavitation_baseline.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.signal import coherence

from equipment_analysis.npz_io import (
    CH0_RELOCATED_FROM, amplitude_spectrum, band_rms, discover_npz_sessions, discover_running_sessions,
    electrical_mask, load_npz_raw,
)

DATA_ROOT = "/mnt/nas/data/GY_DATA/pump data"
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "cavitation_baseline.json"
F_ROT, F_BPF = 58.2, 349.2            # 축 1x, 블레이드통과(6깃)
CARRIER_BAND = (2900.0, 3400.0)       # ch0 조립체 공진 클러스터(본 스크립트 A에서 확인)
BANDS = {"b1_5k": (1000.0, 5000.0), "b5_10k": (5000.0, 10000.0), "b10_20k": (10000.0, 20000.0)}
HIST_OMA = (619.0, 2194.0, 7462.0, 8538.0)   # 구 6채널 OMA 반송파(다른 구성/채널)


def peaks(freq, mag, keep, n=8, sep=150.0, lo=200.0, hi=25000.0):
    sel = (freq > lo) & (freq < hi) & keep
    f, m = freq[sel], mag[sel]
    out = []
    for i in np.argsort(m)[::-1]:
        if all(abs(f[i] - p[0]) > sep for p in out):
            out.append((float(f[i]), float(m[i])))
        if len(out) >= n:
            break
    return out


def sess_features(chans, fs, keep_cache=None):
    row = {}
    for ch in ("ch0", "ch6"):
        x = chans[ch]
        fr, mg = amplitude_spectrum(x, fs)
        keep = keep_cache if keep_cache is not None else electrical_mask(fr)
        row[f"{ch}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        row[f"{ch}_kurt"] = float(((x - x.mean()) ** 4).mean() / x.var() ** 2 - 3.0)
        row[f"{ch}_carrier3k"] = band_rms(fr, mg, CARRIER_BAND, keep)
        for name, b in BANDS.items():
            row[f"{ch}_{name}"] = band_rms(fr, mg, b, keep)

        def pk(fc, tol=8.0):
            s = (fr > fc - tol) & (fr < fc + tol)
            return float(mg[s].max()) if s.any() else 0.0
        row[f"{ch}_1x"] = pk(F_ROT)
        row[f"{ch}_bpf"] = pk(F_BPF)
        sb = 0.5 * (pk(F_BPF - F_ROT) + pk(F_BPF + F_ROT))
        row[f"{ch}_bpf_sb"] = sb
        row[f"{ch}_bpf_sb_ratio"] = sb / row[f"{ch}_bpf"] if row[f"{ch}_bpf"] > 0 else 0.0
    return row


def main():
    sess = discover_running_sessions(DATA_ROOT, date_from=CH0_RELOCATED_FROM)
    print(f"[scope] npz sessions in current configuration (ch0 relocated {CH0_RELOCATED_FROM}~): "
          f"{len(sess)}  ({sess[0].stamp} .. {sess[-1].stamp})", flush=True)

    chans, fs, meta = load_npz_raw(sess[len(sess) // 2].path)
    print(f"[scope] fs={fs:.0f} Hz, n_ch={meta['n_channels']}, n={meta['n_samples']}; "
          f"file unit field='{meta['unit_field']}' -> per-channel units ch0={meta['units'][0]}, "
          f"ch6={meta['units'][6]}  (⚠ file mislabels ch6)", flush=True)

    fr, mg0 = amplitude_spectrum(chans["ch0"], fs)
    keep = electrical_mask(fr)
    print("\n[A] ch0 (near-impeller) dominant peaks, electrical lines excluded:", flush=True)
    pk0 = peaks(fr, mg0, keep)
    for f_, m_ in pk0:
        print(f"      {f_:8.1f} Hz   {m_:.5f}", flush=True)
    print("    ch0 level at historical 6-ch-OMA carriers (different channel/config):", flush=True)
    hist = {}
    for fc in HIST_OMA:
        s = (fr > fc - 60) & (fr < fc + 60)
        hist[fc] = float(mg0[s].max())
        print(f"      {fc:8.1f} Hz   {hist[fc]:.5f}", flush=True)
    carrier_level = band_rms(fr, mg0, CARRIER_BAND, keep)
    print(f"    -> ch0 carrier band {CARRIER_BAND[0]:.0f}-{CARRIER_BAND[1]:.0f} Hz RMS = {carrier_level:.4f} "
          f"vs 7462/8538 Hz peaks {hist[7462.0]:.5f}/{hist[8538.0]:.5f}", flush=True)

    f_c, Cxy = coherence(chans["ch0"], chans["ch6"], fs=fs, nperseg=8192)
    coh = {}
    print("\n[B] ch0-ch6 magnitude-squared coherence (electrical = common-mode, high):", flush=True)
    for lab, (lo, hi) in {"BPF_349": (300, 400), "carrier_3k": (2900, 3400),
                          "pwm_carrier_7820": (7700, 7950), "pwm_sideband_8180": (8150, 8210),
                          "b1_5k": (1000, 5000), "b10_20k": (10000, 20000)}.items():
        s = (f_c >= lo) & (f_c < hi)
        coh[lab] = {"mean": float(Cxy[s].mean()), "max": float(Cxy[s].max())}
        print(f"      {lab:20s} mean={coh[lab]['mean']:.3f}  max={coh[lab]['max']:.3f}", flush=True)

    # 기준선 통계는 전체 in-scope 세션으로 계산한다(밴드 RMS는 세션당 1회 FFT로 저비용).
    # 예전에는 30개 부분표본을 썼으나 근거가 없어 전수로 바꿨다(리뷰 M2).
    sub = sess
    print(f"\n[C] baseline stability over all {len(sub)} in-scope sessions "
          f"(single configuration — no cross-epoch pooling, no subsampling):", flush=True)
    rows = []
    for s in sub:
        c, fss, _ = load_npz_raw(s.path)
        rows.append(sess_features(c, fss))
    stats = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=float)
        med = float(np.median(v)); mad = float(1.4826 * np.median(np.abs(v - med)))
        stats[k] = {"median": med, "mad": mad,
                    "robust_cv_pct": (100 * mad / abs(med)) if med else float("nan"),
                    "min": float(v.min()), "max": float(v.max())}
    for k in ("ch0_rms", "ch0_carrier3k", "ch0_b1_5k", "ch0_b5_10k", "ch0_kurt",
              "ch6_rms", "ch6_carrier3k", "ch6_b1_5k", "ch6_b5_10k", "ch6_kurt"):
        st = stats[k]
        print(f"      {k:16s} median={st['median']:9.4f}  robustCV={st['robust_cv_pct']:6.1f}%  "
              f"[{st['min']:.4f}, {st['max']:.4f}]", flush=True)

    print("\n[D] BPF discriminator baseline visibility (median over the same sessions):", flush=True)
    for ch in ("ch0", "ch6"):
        print(f"      {ch}: 1x={stats[f'{ch}_1x']['median']:.5f}  BPF={stats[f'{ch}_bpf']['median']:.5f}  "
              f"sideband={stats[f'{ch}_bpf_sb']['median']:.5f}  "
              f"sb/BPF={stats[f'{ch}_bpf_sb_ratio']['median']:.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": {"channels": ["ch0", "ch6"], "format": "npz",
                  "config_from": CH0_RELOCATED_FROM, "n_sessions": len(sess),
                  "first": sess[0].stamp, "last": sess[-1].stamp, "fs": fs,
                  "units": {"ch0": "m/s^2", "ch6": "Pa"},
                  "file_unit_field": meta["unit_field"]},
        "ch0_peaks": pk0, "ch0_at_historical_oma": {str(k): v for k, v in hist.items()},
        "ch0_carrier_band": list(CARRIER_BAND), "ch0_carrier_band_rms": carrier_level,
        "coherence": coh, "baseline_stats": stats, "n_baseline_sessions": len(sub),
    }, indent=2))
    print(f"\nSaved -> {OUT}\n=== CAVITATION_BASELINE DONE ===", flush=True)


if __name__ == "__main__":
    main()
