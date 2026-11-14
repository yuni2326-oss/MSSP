"""모달중첩: (손상)고유쌍 → 충격 과도신호. Ritz 모드(φ_n, f_n)로 감쇠 다중모드 링다운 합성.

임펄스 충격이 여러 굽힘 모드를 동시에 여기 → 응답 = Σ_n φ_n(x_s)·q_n(t),
q_n(t)=A_n·e^{-ζω_n t}·sin(ω_{d,n}t). 가속도계 측정이므로 모드 가중 A_n∝ω_n·φ_n(x_s)·φ_n(x_i)
(변위는 ∝1/ω_n; 가속도는 2차 미분으로 ω²배 → 실효 ∝ω_n, 충격의 고주파 강조 특성과 부합).
손상 eigenpair를 쓰면 하강한 주파수·변형된 모드형이 결함 시그니처가 된다.

전달경로 갭(임펠러 vane↔케이싱 센서)은 미모델(설계문서 후속) — φ_n(x_s)는 vane 상 대표점 근사."""
from __future__ import annotations
import numpy as np

from .synth import impact_ringdown


def _phi_at(shape, x: float) -> float:
    import torch
    xt = torch.tensor([[float(x)]], dtype=torch.float64)
    return float(shape(xt).detach().reshape(-1)[0])


def modal_impact_response(modes, fs: float, dur: float, *,
                          x_sensor: float = 0.6, x_impact: float = 0.9,
                          zeta=0.02, t0: float = 0.0, accel: bool = True,
                          noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """모드 집합(solve_eigen_modes_ritz 반환)으로 충격 과도신호[fs·dur 샘플] 합성.

    modes: [{f_pinn, shape}] (shape: callable x̃-tensor→φ). x_sensor/x_impact∈[0,1] vane 좌표.
    zeta: 스칼라 또는 모드별 리스트(구조 감쇠비). accel=True면 가속도 가중 A_n∝ω_n·φ_s·φ_i.
    amp은 최대 |A|=1로 상대 정규화(절대 크기는 scale_to_rms로 별도 조정)."""
    freqs, amps, zetas = [], [], []
    for i, m in enumerate(modes):
        f = float(m["f_pinn"]); w = 2 * np.pi * f
        ps = _phi_at(m["shape"], x_sensor)
        pi = _phi_at(m["shape"], x_impact)
        a = ps * pi * (w if accel else 1.0 / w)
        freqs.append(f)
        amps.append(a)
        zetas.append(zeta[i] if hasattr(zeta, "__len__") else zeta)
    amax = max((abs(a) for a in amps), default=1.0) or 1.0
    amps = [a / amax for a in amps]
    return impact_ringdown(fs, dur, freqs, zetas, amps, t0=t0, noise=noise, seed=seed)


def impact_train(single: np.ndarray, fs: float, period_s: float, dur: float,
                 jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """단발 충격 링다운을 회전주기마다 반복 배치 → 주기적 충격열[fs·dur 샘플].
    임펠러 결함(이물질 타격·손상 블레이드)은 회전당 1회 임펄스 → 충격열이 물리적. 결과는
    kurtosis↑(충격성)·포락선 f_rot 피크·RMS↑로 검출특징을 형성. jitter=주기 요동(비율, 랜덤성)."""
    n = int(round(fs * dur))
    out = np.zeros(n)
    step = max(int(round(period_s * fs)), 1)
    L = len(single)
    rng = np.random.default_rng(seed)
    start = 0
    while start < n:
        end = min(start + L, n)
        out[start:end] += single[:end - start]
        dj = int(rng.normal(0.0, jitter * step)) if jitter > 0 else 0
        start += max(step + dj, 1)
    return out


def scale_to_rms(sig: np.ndarray, target_rms: float) -> np.ndarray:
    """신호를 목표 RMS로 스케일(주입 진폭을 실신호 대비 검출가능 수준으로; Phase A 한계③ 보정)."""
    sig = np.asarray(sig, dtype=float)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms <= 0:
        return sig
    return sig * (target_rms / rms)
