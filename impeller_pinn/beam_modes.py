"""캔틸레버(외팔보) 고유주파수 — Euler-Bernoulli. 임펠러 베인을 얇은 캔틸레버로 근사."""
from __future__ import annotations
import math

# 캔틸레버 고유값 (β_n L): 1~4차
_BETA_L = (1.8751041, 4.6940911, 7.8547574, 10.9955407)


def cantilever_frequencies(L: float, h: float, E: float, rho: float, n_modes: int = 4) -> list[float]:
    """직사각 단면(두께 h) 캔틸레버 굽힘 고유주파수(Hz).
    f_n = (β_n L)²/(2π L²) · (h/√12)·√(E/ρ).  (폭 b는 상쇄되어 무관 = 2D 단위폭)
    n_modes>4면 4로 절단(고유값 4차까지만 표), n_modes<=0면 빈 리스트 반환."""
    if min(L, h, E, rho) <= 0:
        raise ValueError("L,h,E,rho는 양수여야 합니다")
    coef = (h / math.sqrt(12.0)) * math.sqrt(E / rho) / (L ** 2)
    n = min(n_modes, len(_BETA_L))
    return [((_BETA_L[i] ** 2) / (2.0 * math.pi)) * coef for i in range(n)]
