import pytest
from impeller_pinn.beam_modes import cantilever_frequencies

def test_steel_cantilever_first_mode():
    # 강 캔틸레버 L=100mm,h=4mm,E=200GPa,ρ=7850 → 1차 ≈ 326Hz(Euler-Bernoulli 해석해)
    f = cantilever_frequencies(L=0.1, h=0.004, E=200e9, rho=7850, n_modes=4)
    assert f[0] == pytest.approx(326.0, rel=0.02)
    # 모드비 f2/f1 = (4.694/1.875)^2 ≈ 6.267 (경계값 고정비)
    assert f[1] / f[0] == pytest.approx(6.267, rel=0.01)
    assert f[2] / f[0] == pytest.approx(17.55, rel=0.02)

def test_sus316_thin_vane_high_freq():
    # SUS304 얇은 베인(추정 L30/h1.2mm) → 1차 ~1kHz, 3차 ~18kHz(관측 고주파와 부합)
    f = cantilever_frequencies(L=0.030, h=0.0012, E=193e9, rho=8000)
    assert 900 < f[0] < 1200
    assert 15000 < f[2] < 22000
