"""고전 다항식 Rayleigh–Ritz 캔틸레버 고유치 솔버 — PINN 대조 베이스라인(torch 불필요, numpy/scipy).

Euler–Bernoulli 정규화(x̃=x/L∈[0,1], 고정단 φ(0)=φ'(0)=0):
    [(1−d(x̃)) φ'']'' = Λ φ,     Rayleigh: Λ = ∫(1−d)(φ'')² dx̃ / ∫φ² dx̃.
시행함수 ψ_k(x̃)=x̃^{k+1}(고정단 자동만족). 일반화 고유치 K c = Λ M c,
    K_ij=∫(1−d) ψ_i'' ψ_j'' dx̃,  M_ij=∫ ψ_i ψ_j dx̃.
주파수 f_n = √Λ_n · ω_b/(2π),  ω_b=(h/√12)√(E/ρ)/L²  (beam_modes와 동일 스케일 → PINN·해석해와 직접 비교).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import eigh


def solve_ritz(L, h, E, rho, n_modes=3, damage=None, n_trial=7, n_grid=4001):
    """(1−d) 가중 캔틸레버 고유주파수[Hz] n_modes개 + 내부노드수 반환. damage: x̃(ndarray)→d(x̃) 또는 None."""
    x = np.linspace(0.0, 1.0, n_grid)
    w = 1.0 - (np.asarray(damage(x), dtype=float) if damage is not None else 0.0)
    ks = np.arange(1, n_trial + 1)
    psi = np.stack([x ** (k + 1) for k in ks])                    # ψ_k
    d2 = np.stack([(k + 1) * k * x ** (k - 1) for k in ks])       # ψ_k''
    K = np.array([[np.trapezoid(w * d2[i] * d2[j], x) for j in range(n_trial)] for i in range(n_trial)])
    M = np.array([[np.trapezoid(psi[i] * psi[j], x) for j in range(n_trial)] for i in range(n_trial)])
    Lam, C = eigh(K, M)                                           # 오름차순
    omega_b = (h / math.sqrt(12.0)) * math.sqrt(E / rho) / (L ** 2)
    out = []
    for n in range(min(n_modes, n_trial)):
        lam = float(Lam[n])
        f = math.sqrt(max(lam, 0.0)) * omega_b / (2 * math.pi)
        shape = C[:, n] @ psi
        s = shape[x > 0.03]                                       # 고정단 근방 아티팩트 제외
        nodes = int(np.sum(np.diff(np.sign(s)) != 0))
        out.append({"f": f, "Lambda": lam, "nodes": nodes})
    return out
