"""충격 링다운 합성: 임펄스가 굽힘 모드를 여기 → 감쇠 사인 합. (Phase B에서 PINN 해로 교체)"""
from __future__ import annotations
import numpy as np


def impact_ringdown(fs: float, dur: float, freqs, zetas, amps,
                    t0: float = 0.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """fs[Hz]·dur[s] 신호. t≥t0에서 Σ aᵢ·exp(-ζᵢ·2π fᵢ·τ)·sin(2π fᵢ·τ), τ=t-t0. +선택 잡음."""
    n = int(round(fs * dur))
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    sig = rng.normal(0.0, noise, n) if noise > 0 else np.zeros(n)
    m = t >= t0
    tau = t[m] - t0
    for f, z, a in zip(freqs, zetas, amps):
        sig[m] += a * np.exp(-z * 2 * np.pi * f * tau) * np.sin(2 * np.pi * f * tau)
    return sig


def shift_freqs(freqs, ratio: float) -> list[float]:
    """손상에 의한 강성저하 → 고유주파수 하강(ratio<1). 건전 대비 손상 모델."""
    return [f * ratio for f in freqs]
