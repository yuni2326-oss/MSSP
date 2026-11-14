"""정량 검출연구(§3.6/§4.4 프로토콜의 실제 실행): severity × in-band SNR × 회전당1회 반복률
스윕을 실측 healthy 세션에 주입하고, 배포된 robust-z 검출기(equipment_analysis.anomaly)로
탐지확률·오경보율·ROC를 산출한다. 순전히 소프트웨어([SW]) — GPU/torch 불필요
(classical_ritz가 이미 PINN과 ≤~1%로 동등함이 확립됨, §4.1 Table 2).

프로토콜(논문 §3.6과 정합):
  요인 — severity d_max ∈ {0.1,0.2,0.3,0.45,0.6} (베인 근원부 x_d=0.2,r_d=0.12 가우시안 손상,
         classical Ritz 3모드, Nyquist 대역제한), in-band SNR ∈ {-10,-5,0,+5,+10} dB
         (주입모드 대역 vs 실측채널 동일대역 에너지비), 반복=세션 검출 rpm 기준 회전당1회
         (±2% 주기지터).
  데이터 — M개 held-out 정상세션(각 셀에 1회 주입) + baseline_n=10 별도 baseline 세션(등록부
         baseline_from 기준) + held-out 세션 자체(비주입)로 오경보율.
  검출기 — anomaly.robust_z(baseline_n=10, sensitivity=3.0, rel_floor=0.05), ch1 특징 전체
         (session_features, registry pwm_notch/rpm_range 반영), 플래그="어떤 특징이든 |z|>threshold".
"""
from __future__ import annotations
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from equipment_analysis import dataio, anomaly
from equipment_analysis.features import session_features, time_features, detect_rpm, band_rms_spectral
from impeller_pinn.classical_ritz import solve_ritz
from impeller_pinn.beam_modes import cantilever_frequencies
from impeller_pinn.synth import impact_ringdown
from impeller_pinn.modal import impact_train
from impeller_pinn import inject as inject_mod

GEO = dict(L=0.030, h=0.0012, E=193e9, rho=8000)   # §4.2 cantilever geometry
DAMAGE_X0, DAMAGE_R = 0.2, 0.12                     # §4.2 root-region Gaussian, same as Fig 5/Table 3
NYQUIST_MARGIN = 0.98                                # band-limit strictly under Nyquist
REG = dict(rpm_range=(50.0, 70.0),
          pwm_notch={"carrier": 7820.0, "n_harm": 3, "width": 300.0})
BASELINE_N, SENSITIVITY, REL_FLOOR = 10, 3.0, 0.05
CHANNEL = "ch1"
ZETAS = (0.02, 0.03, 0.04)
# 고정폭 기본모드 추적대역: 건전 f1(1057.9Hz)~최대심각도 d_max=0.6 이동 f1(≈874Hz)를 여유있게
# 포괄(±마진). 기존 ch1 특징(RMS/crest/kurtosis 등 광대역)은 baseline 자연변동 대비 이 협대역
# 주입을 SNR+10dB에서도 못 잡음(파일럿 확인) — resonance_features와 같은 원리(band_rms_spectral
# 재사용)로 '합성엔진이 아는 목표주파수를 등록'하는 것이 실제 배포에서 자연스러운 확장(§4.4에서
# as-shipped vs 확장 두 구성을 모두 정직하게 보고).
FAULTBAND_F1 = (700.0, 1220.0)


def fault_band_feature(fr: np.ndarray, mg: np.ndarray) -> float:
    return band_rms_spectral(fr, mg, FAULTBAND_F1)


def damaged_mode_freqs(d_max: float, fs: float) -> list[float]:
    """classical Ritz(맹목, §4.1의 PINN과 동등) 손상 모드주파수, Nyquist 아래로 대역제한."""
    dfun = (lambda xt, dm=d_max: dm * np.exp(-((xt - DAMAGE_X0) ** 2) / (DAMAGE_R ** 2)))
    modes = solve_ritz(**GEO, n_modes=3, damage=dfun, n_trial=7)
    fnyq = NYQUIST_MARGIN * fs / 2.0
    return [m["f"] for m in modes if m["f"] < fnyq]


def _band_rms(sig: np.ndarray, fs: float, freqs: list[float], halfwidth_hz: float = 150.0) -> float:
    """freqs 각각 ±halfwidth_hz 대역의 합산 에너지에서 RMS(FFT 마스킹)."""
    n = len(sig)
    X = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = np.zeros_like(f, dtype=bool)
    for fc in freqs:
        mask |= (np.abs(f - fc) <= halfwidth_hz)
    power = np.sum(np.abs(X[mask]) ** 2) / n ** 2 * 2.0   # 단측 파워 근사
    return float(np.sqrt(max(power, 0.0)))


def snr_scale(ringdown_train: np.ndarray, real_channel: np.ndarray, fs: float,
             mode_freqs: list[float], target_snr_db: float,
             halfwidth_hz: float = 150.0) -> np.ndarray:
    """ringdown_train을 real_channel과 동일 대역(모드주파수 ±halfwidth)에서
    target_snr_db(=20log10(rms_inj/rms_real))를 만족하도록 스케일."""
    rms_real = _band_rms(real_channel, fs, mode_freqs, halfwidth_hz)
    rms_ring = _band_rms(ringdown_train, fs, mode_freqs, halfwidth_hz)
    if rms_ring <= 0 or rms_real <= 0:
        return ringdown_train
    target_rms = rms_real * (10.0 ** (target_snr_db / 20.0))
    return ringdown_train * (target_rms / rms_ring)


def build_injection(session: dataio.Session, d_max: float, snr_db: float,
                    jitter: float = 0.02, seed: int = 0) -> np.ndarray:
    """세션의 실측 rpm으로 회전당1회 충격열을 만들어 target SNR로 스케일한 주입 파형(raw 길이) 반환."""
    t, chans, fs = dataio.load_raw(session.raw_path)
    real = chans[CHANNEL]
    n = len(real)
    dur = n / fs                                     # 정확히 n=len(real) 샘플이 되도록(라운딩 불일치 방지)
    rpm = 0.0
    for idx in sorted(session.fft_paths):
        fr, mg = dataio.load_fft(session.fft_paths[idx])
        rpm = detect_rpm(fr, mg, REG["rpm_range"])
        if rpm > 0:
            break
    if rpm <= 0:
        rpm = 3493.0                                  # registry 상수속도 폴백
    f_rot = rpm / 60.0
    freqs = damaged_mode_freqs(d_max, fs)
    single = impact_ringdown(fs, dur, freqs, ZETAS[:len(freqs)], [1.0, 0.5, 0.25][:len(freqs)])
    train = impact_train(single, fs, period_s=1.0 / f_rot, dur=dur, jitter=jitter, seed=seed)
    if len(train) != n:                              # 라운딩 안전망: raw 길이에 정확히 맞춤
        train = train[:n] if len(train) > n else np.pad(train, (0, n - len(train)))
    return snr_scale(train, real, fs, freqs, snr_db), freqs, rpm


def score_session(features_row: dict, baseline_table: pd.DataFrame,
                  feature_cols: list[str]) -> dict:
    """features_row(단일 세션 특징 dict)를 baseline_table(healthy N행) 대비 robust_z로 채점.
    반환: flagged(bool, 어떤 특징이든 |z|>sensitivity), z_by_feature(dict), max_abs_z."""
    z_by_feature = {}
    for f in feature_cols:
        if f not in features_row or pd.isna(features_row[f]):
            continue
        base_vals = baseline_table[f].dropna().to_numpy(dtype=float)
        if len(base_vals) < 2:
            continue
        v = np.concatenate([base_vals[:BASELINE_N], [features_row[f]]])
        z = anomaly.robust_z(v, BASELINE_N, rel_floor=REL_FLOOR)
        z_by_feature[f] = float(z[-1])
    if not z_by_feature:
        return {"flagged": False, "z_by_feature": {}, "max_abs_z": 0.0}
    max_abs = max(abs(z) for z in z_by_feature.values())
    return {"flagged": max_abs > SENSITIVITY, "z_by_feature": z_by_feature, "max_abs_z": max_abs}


def flagged_at_threshold(features_row: dict, baseline_table: pd.DataFrame,
                         feature_cols: list[str], threshold: float) -> bool:
    for f in feature_cols:
        if f not in features_row or pd.isna(features_row[f]):
            continue
        base_vals = baseline_table[f].dropna().to_numpy(dtype=float)
        if len(base_vals) < 2:
            continue
        v = np.concatenate([base_vals[:BASELINE_N], [features_row[f]]])
        z = anomaly.robust_z(v, BASELINE_N, rel_floor=REL_FLOOR)
        if abs(float(z[-1])) > threshold:
            return True
    return False


def features_for_session(session: dataio.Session, *, with_faultband: bool = True) -> dict:
    """생산 session_features(ch1, registry pwm_notch/rpm_range 반영)에, 확장구성 시
    ch1_faultband_f1(고정 700-1220Hz 기본모드 추적대역, resonances 패턴과 동일 원리)을 병합."""
    feats = session_features(session, channels=[CHANNEL], rpm_range=REG["rpm_range"],
                             pwm_notch=REG["pwm_notch"])
    if with_faultband and 1 in session.fft_paths:
        fr, mg = dataio.load_fft(session.fft_paths[1])
        feats[f"{CHANNEL}_faultband_f1"] = fault_band_feature(fr, mg)
    return feats


SEVERITIES = (0.1, 0.2, 0.3, 0.45, 0.6)
SNRS_DB = (-10.0, -5.0, 0.0, 5.0, 10.0)


def run_grid(baseline_sessions: list, held_out_sessions: list, scratch_dir: Path, *,
            severities=SEVERITIES, snrs=SNRS_DB, feature_cols: list[str] | None = None,
            log=print) -> dict:
    """전체 severity×SNR 그리드 실행: baseline 특징표 구축 → held-out(비주입) 오경보율 →
    각 셀 held-out 세션마다 주입+채점 → 탐지확률(%). feature_cols=None이면 baseline 첫 세션의
    수치 특징 전부(오류 컬럼 제외) 사용."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    log(f"[grid] baseline={len(baseline_sessions)} held_out={len(held_out_sessions)} "
       f"severities={severities} snrs={snrs}")
    base_rows = [features_for_session(s) for s in baseline_sessions]
    base_table = pd.DataFrame(base_rows)
    if feature_cols is None:
        feature_cols = [c for c in base_table.columns
                       if c not in ("stamp", "fs", "rpm") and base_table[c].dtype.kind in "fi"]
    log(f"[grid] scoring against {len(feature_cols)} features")

    ho_rows = [features_for_session(s) for s in held_out_sessions]
    fa_flags = [score_session(r, base_table, feature_cols)["flagged"] for r in ho_rows]
    false_alarm_rate = 100.0 * float(np.mean(fa_flags))
    log(f"[grid] false-alarm rate on {len(held_out_sessions)} held-out healthy: "
       f"{false_alarm_rate:.1f}%")

    table6 = {}
    all_cells = {}
    tag = 0
    for d_max in severities:
        row = {}
        for snr_db in snrs:
            flags, zmax = [], []
            for s in held_out_sessions:
                tag += 1
                feats = inject_and_features(s, d_max, snr_db, scratch_dir, seed=tag, tag=tag)
                sc = score_session(feats, base_table, feature_cols)
                flags.append(sc["flagged"]); zmax.append(sc["max_abs_z"])
            det_pct = 100.0 * float(np.mean(flags))
            row[snr_db] = det_pct
            all_cells[(d_max, snr_db)] = {"det_pct": det_pct, "zmax": zmax, "flags": flags}
            log(f"[grid] d_max={d_max} snr={snr_db:+.0f}dB -> detection {det_pct:5.1f}% "
               f"(z_max median {np.median(zmax):.1f})")
        table6[d_max] = row
    return {"table6": table6, "false_alarm_rate": false_alarm_rate, "cells": all_cells,
           "feature_cols": feature_cols, "base_table": base_table}


def roc_curve(held_out_sessions: list, base_table: pd.DataFrame, feature_cols: list[str],
             scratch_dir: Path, d_max: float, snr_db: float, *,
             thresholds=None, seed0: int = 9000) -> dict:
    """고정 severity/SNR에서 threshold(민감도) 스윕 ROC: TPR=주입세션 검출률,
    FPR=held-out 건전세션(비주입) 오탐률(같은 threshold)."""
    if thresholds is None:
        thresholds = np.concatenate([np.arange(0.5, 5.01, 0.25), np.arange(5.5, 20.01, 1.0)])
    inj_feats = []
    for i, s in enumerate(held_out_sessions):
        feats = inject_and_features(s, d_max, snr_db, scratch_dir, seed=seed0 + i, tag=seed0 + i)
        inj_feats.append(feats)
    ho_feats = [features_for_session(s) for s in held_out_sessions]
    tpr, fpr = [], []
    for th in thresholds:
        tpr.append(100.0 * np.mean([flagged_at_threshold(f, base_table, feature_cols, th)
                                    for f in inj_feats]))
        fpr.append(100.0 * np.mean([flagged_at_threshold(f, base_table, feature_cols, th)
                                    for f in ho_feats]))
    return {"thresholds": list(thresholds), "tpr": tpr, "fpr": fpr}


def inject_and_features(session: dataio.Session, d_max: float, snr_db: float,
                        scratch_dir: Path, seed: int = 0, tag: int = 0) -> dict:
    """세션에 (d_max,snr_db) 주입 → 임시 세션 생성 → 특징 계산 → 임시파일 정리 → 특징dict 반환.
    tag: 호출자가 부여하는 고유 정수(그리드 셀 인덱스 등) — 임시 세션 스탬프 충돌 방지용."""
    ring, freqs, rpm = build_injection(session, d_max, snr_db, seed=seed)
    # dataio._STAMP_RE는 \d{8}_\d{6} 형식만 세션으로 인식 → 날짜부는 원 세션 유지,
    # 시각부는 호출자 tag(고유 정수)의 6자리 표현으로 대체(임시파일, 사용 후 즉시 삭제).
    out_stamp = f"{session.stamp[:8]}_{tag % 1_000_000:06d}"
    out_dir = scratch_dir / out_stamp
    inject_mod.make_injected_session(session.raw_path.parent, session.stamp, out_dir,
                                     CHANNEL, ring, out_stamp)
    inj_session = dataio.discover_sessions(str(out_dir))[0]
    feats = features_for_session(inj_session)
    shutil.rmtree(out_dir, ignore_errors=True)
    return feats
