"""열해석 보조함수 테스트 — p_v(Buck)·물밀도·σ 민감도 부호와 자릿수."""
import numpy as np
import pytest

from impeller_pinn.thermal_analysis import p_v_kPa, rho_water


def test_pv_reference_values():
    # 표준 포화증기압(문헌): 20°C 2.339, 25°C 3.169, 40°C 7.381 kPa
    assert float(p_v_kPa(20.0)) == pytest.approx(2.339, rel=0.01)
    assert float(p_v_kPa(25.0)) == pytest.approx(3.169, rel=0.01)
    assert float(p_v_kPa(40.0)) == pytest.approx(7.381, rel=0.01)


def test_pv_monotonic_increasing():
    T = np.linspace(0, 50, 51)
    assert np.all(np.diff(p_v_kPa(T)) > 0)


def test_rho_water_reference_and_monotonic_above_4C():
    assert float(rho_water(4.0)) == pytest.approx(1000.0, abs=0.5)
    assert float(rho_water(38.0)) == pytest.approx(993.0, abs=1.0)
    T = np.linspace(10, 50, 41)
    assert np.all(np.diff(rho_water(T)) < 0)      # 4°C 이상에선 온도↑ → 밀도↓


def test_sigma_sensitivity_sign_and_magnitude():
    """dσ/dT_w = −(dp_v/dT)/q < 0, 38°C·q≈15.7kPa에서 -0.02~-0.03 /°C."""
    U = np.pi * 0.0308 * 58.2
    q = 0.5 * float(rho_water(38.2)) * U ** 2 / 1000.0
    dpv = float(p_v_kPa(38.7) - p_v_kPa(37.7))
    dsig = -dpv / q
    assert q == pytest.approx(15.7, abs=0.5)
    assert -0.03 < dsig < -0.02


def test_using_pipe_temperature_biases_sigma_more_than_thermal_drift():
    """배관온(45°C) 오사용 편향 > 관측 수온 표류(35.7→40.1°C)로 인한 σ 이동."""
    U = np.pi * 0.0308 * 58.2
    q = 0.5 * float(rho_water(38.2)) * U ** 2 / 1000.0
    drift = abs(float(p_v_kPa(40.07) - p_v_kPa(35.70))) / q
    pipe_bias = abs(float(p_v_kPa(45.0) - p_v_kPa(38.2))) / q
    assert drift == pytest.approx(0.099, abs=0.01)
    assert pipe_bias > drift
