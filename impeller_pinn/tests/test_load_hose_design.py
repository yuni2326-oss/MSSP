"""토출 부하 호스 설계값 — 설치도·상태 문서에 인용한 수치를 고정한다."""
import pytest

from impeller_pinn.load_hose_design import (
    dp_flowmeter, hose_loss, layout, required_cv, valve_alone_sigma, water_nu,
)


def test_water_viscosity_at_38C():
    assert water_nu(38.0) == pytest.approx(0.68e-6, rel=0.02)


def test_three_quarter_hose_friction_at_rated_flow():
    h = hose_loss()
    assert h["v"] == pytest.approx(3.53, abs=0.01)
    assert h["hf_m_per_m"] == pytest.approx(0.684, abs=0.01)
    assert h["hf_bar_per_m"] == pytest.approx(0.0666, abs=0.001)


def test_recommended_layouts_pass_both_criteria():
    for L, trim, sig in ((22.0, 1.19, 2.62), (25.0, 0.99, 3.15)):
        r = layout(L)
        assert r["trim_bar"] == pytest.approx(trim, abs=0.02)
        assert r["sigma_trim"] == pytest.approx(sig, abs=0.05)
        assert r["comp_ok"] and r["sigma_ok"]


def test_thirty_metres_leaves_too_little_compensation():
    r = layout(30.0)
    assert r["trim_bar"] == pytest.approx(0.66, abs=0.02)
    assert not r["comp_ok"]


def test_twenty_metres_leaves_too_little_cavitation_margin():
    r = layout(20.0)
    assert r["sigma_trim"] == pytest.approx(2.36, abs=0.05)
    assert r["comp_ok"] and not r["sigma_ok"]


def test_single_valve_cavitates():
    assert valve_alone_sigma() == pytest.approx(1.36, abs=0.02)


def test_friction_segment_works_as_a_flowmeter():
    fm = dp_flowmeter()
    assert fm["dp_bar"] == pytest.approx(0.734, abs=0.01)
    assert fm["q_resolution_pct"] < 0.2


def test_needle_valve_cannot_pass_rated_flow():
    """¾″ 니들밸브 Cv ≈ 1–2 < 필요 Cv."""
    assert required_cv(1.0, 1.0) == pytest.approx(4.16, abs=0.05)
    assert required_cv(1.0, 0.2) == pytest.approx(9.3, abs=0.1)
