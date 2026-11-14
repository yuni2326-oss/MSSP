"""09-02 예비시험 산출 보조함수 — σ 식·1× 검출·BPF 측파대비·로그 별칭."""
import numpy as np
import pytest

from impeller_pinn.pilot_cavitation import (
    bpf_sideband_ratio, cavitation_number, f_rot_from_spectrum, read_log, sample_log,
)


def test_sigma_reference_value():
    """무스로틀(p_s≈0 bar g)·38.5 °C·58 Hz에서 σ≈6, NPSH_a≈9.7 m."""
    r = cavitation_number(-0.001, 38.5, 58.0)
    assert r["U_m_s"] == pytest.approx(np.pi * 0.0308 * 58.0, rel=1e-12)
    assert r["q_kPa"] == pytest.approx(15.6, abs=0.3)
    assert r["sigma"] == pytest.approx(6.04, abs=0.05)
    assert r["NPSHa_m"] == pytest.approx(9.70, abs=0.05)


def test_sigma_decreases_with_suction_pressure():
    p = np.linspace(-0.85, 0.0, 20)
    s = [cavitation_number(x, 38.0, 58.0)["sigma"] for x in p]
    assert np.all(np.diff(s) > 0)                      # p_s 상승 → σ 상승
    assert cavitation_number(-0.83, 38.0, 58.0)["sigma"] < 1.0


def test_sigma_decreases_with_water_temperature():
    """p_v(T)↑ 이므로 dσ/dT_w < 0, 38 °C 부근에서 −0.02 ~ −0.03 /°C."""
    a = cavitation_number(0.0, 37.5, 58.0)["sigma"]
    b = cavitation_number(0.0, 38.5, 58.0)["sigma"]
    assert b < a
    assert (b - a) == pytest.approx(-0.022, abs=0.008)


def test_sigma_scales_inverse_square_with_speed():
    a = cavitation_number(0.0, 38.0, 58.0)["sigma"]
    b = cavitation_number(0.0, 38.0, 116.0)["sigma"]
    assert b == pytest.approx(a / 4.0, rel=1e-9)


def test_f_rot_picks_the_1x_line():
    f = np.arange(0.0, 200.0, 0.5)
    m = np.zeros_like(f)
    m[np.argmin(np.abs(f - 58.4))] = 0.3
    m[np.argmin(np.abs(f - 116.8))] = 0.9          # 2×가 더 커도 대역 밖이라 무시돼야 한다
    f_rot, a = f_rot_from_spectrum(f, m)
    assert f_rot == pytest.approx(58.4, abs=0.5)
    assert a == pytest.approx(0.3)


def test_bpf_sideband_ratio_known_geometry():
    """BPF=6×f_rot 진폭 1.0, 양 측파대 0.4/0.6 → 비 = 0.5."""
    f_rot = 58.0
    f = np.arange(0.0, 600.0, 0.5)
    m = np.zeros_like(f)
    for fc, amp in ((6 * f_rot, 1.0), (6 * f_rot - f_rot, 0.4), (6 * f_rot + f_rot, 0.6)):
        m[np.argmin(np.abs(f - fc))] = amp
    assert bpf_sideband_ratio(f, m, f_rot) == pytest.approx(0.5, rel=1e-6)


def test_bpf_ratio_nan_without_bpf_line():
    f = np.arange(0.0, 600.0, 0.5)
    assert np.isnan(bpf_sideband_ratio(f, np.zeros_like(f), 58.0))


def test_read_log_accepts_both_header_spellings(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("timestamp,p_discharge_bar,p_suction_bar\n2026-09-02 16:25:36,0.084,-0.001\n")
    b = tmp_path / "b.csv"
    b.write_text("timestamp,ai0_bar,ai1_bar\n2026-09-02 16:25:36,0.084,-0.001\n")
    alias = {"ps": ("p_suction_bar", "ai1_bar"), "pd": ("p_discharge_bar", "ai0_bar")}
    for path in (a, b):
        t, v = read_log(path, alias)
        assert len(t) == 1 and v["ps"][0] == pytest.approx(-0.001) and v["pd"][0] == pytest.approx(0.084)


def test_read_log_missing_file_returns_none(tmp_path):
    assert read_log(tmp_path / "nope.csv", {"ps": ("x",)}) == (None, None)


def test_sample_log_window_and_nearest_fallback():
    t = np.array([np.datetime64("2026-09-02T16:00:00") + np.timedelta64(5 * i, "s") for i in range(5)])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, spread, gap = sample_log(t, v, np.datetime64("2026-09-02T16:00:05"), 5.0)
    assert mean == pytest.approx(2.0) and spread == pytest.approx(2.0) and gap == 0.0
    mean, spread, gap = sample_log(t, v, np.datetime64("2026-09-02T16:10:00"), 5.0)
    assert mean == pytest.approx(5.0) and spread == 0.0 and gap == pytest.approx(580.0)


def test_pump_head_sign_and_magnitude():
    """무스로틀 0.082/−0.001 bar → H ≈ 0.85 m, 깊은 스로틀 0.102/−0.812 → H ≈ 9.4 m."""
    from impeller_pinn.pilot_cavitation import pump_head_m
    assert pump_head_m(0.0820, -0.0012, 38.5) == pytest.approx(0.85, abs=0.03)
    assert pump_head_m(0.1022, -0.8115, 38.5) == pytest.approx(9.38, abs=0.05)


def test_pump_head_rises_when_suction_is_throttled():
    """토출이 고정돼 있어도 흡입을 조이면 H는 올라간다 — 캐비 신호가 아니라 운전점 이동."""
    from impeller_pinn.pilot_cavitation import pump_head_m
    h = [pump_head_m(0.082, ps, 38.0) for ps in (-0.0, -0.2, -0.4, -0.8)]
    assert np.all(np.diff(h) > 0)
