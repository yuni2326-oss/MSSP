# tests/test_anomaly.py
import numpy as np
import pandas as pd
import pytest
from equipment_analysis import anomaly

def test_flag_step_change():
    vals = np.concatenate([np.full(20, 1.0) + np.random.default_rng(0).normal(0, 0.01, 20),
                           np.full(5, 2.0)])   # 뒤 5개 스텝 상승
    mask = anomaly.flag_anomalies(vals, baseline_n=10, sensitivity=3.0)
    assert mask[-5:].all()       # 상승 구간 검출
    assert not mask[:10].any()   # 기준선 정상

def test_low_false_positive_rate_on_normal():
    # 3σ 임계는 정상 데이터에서도 소표본 scale 추정 탓에 본질적으로 작은 오탐이 가능하다.
    # 이 테스트는 '대량 오탐'(scale 과소추정 버그, 수정 전 ~30%) 재발 방지가 목적.
    vals = 1.0 + np.random.default_rng(1).normal(0, 0.01, 30)
    mask = anomaly.flag_anomalies(vals, baseline_n=10, sensitivity=3.0)
    assert mask.mean() < 0.1     # 정상 데이터 오탐률 10% 미만

def test_robust_z_baseline_from_shifts_window():
    # 앞 5개(옛 위치, 값 100) + baseline_from 이후 10개(정상 10) + 마지막 이상(20).
    stamps = np.array([f"20260101_{i:06d}" for i in range(5)]
                      + [f"20260707_{i:06d}" for i in range(11)])
    vals = np.array([100.0] * 5 + [10.0] * 10 + [20.0])
    z = anomaly.robust_z(vals, baseline_n=10, stamps=stamps,
                         baseline_from="20260707_000000", rel_floor=0.05)
    # baseline=이후 첫10개(=10) → med=10, rel_floor·|med|=0.5. 마지막 20 → z=(20-10)/0.5=20.
    assert z[-1] == pytest.approx(20.0, rel=1e-6)
    assert abs(z[5]) < 1e-9                       # baseline 구간 값(10)의 z≈0


def test_robust_z_rel_floor_caps_explosion():
    base = 5.0 + np.random.default_rng(0).normal(0, 1e-6, 10)   # 거의 상수 baseline
    vals = np.concatenate([base, [6.0]])                        # median 대비 +20%
    z_nofloor = anomaly.robust_z(vals, baseline_n=10, rel_floor=0.0)
    z_floor = anomaly.robust_z(vals, baseline_n=10, rel_floor=0.05)
    assert abs(z_nofloor[-1]) > 1000                             # 하한 없으면 폭증
    assert abs(z_floor[-1]) == pytest.approx(1.0 / (0.05 * 5.0), rel=1e-2)  # =4.0 완화


def test_robust_z_baseline_from_fallback_when_no_match():
    vals = np.array([1.0] * 10 + [5.0])
    stamps = np.array([f"20260101_{i:06d}" for i in range(11)])
    z = anomaly.robust_z(vals, baseline_n=10, stamps=stamps, baseline_from="20991231_000000")
    assert abs(z[0]) < 1e-6 and np.isfinite(z[-1])   # 매칭 없음 → 앞 10개로 폴백


def test_detect_trend_anomalies_df():
    df = pd.DataFrame({"stamp": [f"s{i}" for i in range(25)],
                       "ch0_kurtosis": [0.0]*20 + [9.0]*5})
    out = anomaly.detect_trend_anomalies(df, "ch0_kurtosis", baseline_n=10, sensitivity=3.0)
    assert out["is_anomaly"].sum() == 5
    assert set(out.columns) >= {"stamp", "value", "z", "is_anomaly"}

def test_detect_trend_anomalies_skips_nan_rows():
    # 손상 세션(NaN)이 baseline 구간에 있어도 거짓음성 없이 실제 이상을 탐지해야 한다.
    df = pd.DataFrame({
        "stamp": [f"s{i}" for i in range(26)],
        "ch0_kurtosis": [np.nan] + [0.0] * 20 + [9.0] * 5,   # 첫 행 NaN(손상 세션)
    })
    out = anomaly.detect_trend_anomalies(df, "ch0_kurtosis", baseline_n=10, sensitivity=3.0)
    assert len(out) == 25                  # NaN 행 제외됨
    assert out["is_anomaly"].sum() == 5    # 실제 이상은 여전히 탐지(거짓음성 아님)
    assert not out["value"].isna().any()


def test_robust_z_rpm_binned_handles_speed():
    # 두 속도군(저속 진폭~1, 고속 진폭~10)이 모두 정상. 단일 baseline이면 고속군이 대량 오탐,
    # rpm-binned면 각 속도 이웃 대비라 정상. 고속군 이상치는 여전히 검출(F1 수정).
    rng = np.random.default_rng(0)
    lo = 1.0 + rng.normal(0, 0.03, 40)
    hi = 10.0 + rng.normal(0, 0.3, 40)
    vals = np.concatenate([lo, hi])
    rpm = np.concatenate([np.full(40, 1800.0), np.full(40, 6000.0)])
    z_single = anomaly.robust_z(vals, baseline_n=40)
    assert np.abs(z_single[40:]).max() > 10                 # 단일 baseline: 고속군 오탐
    z_binned = anomaly.robust_z(vals, baseline_n=40, rpm=rpm, rpm_bin=900.0)
    assert np.abs(z_binned).max() < 5                       # rpm-binned: 정상 둘 다 통과
    vals2 = vals.copy(); vals2[60] = 20.0                   # 고속군 실제 이상치
    z2 = anomaly.robust_z(vals2, baseline_n=40, rpm=rpm, rpm_bin=900.0)
    assert abs(z2[60]) > 5                                  # 이웃(고속) 대비 여전히 검출


def test_robust_z_rpm_none_backward_compatible():
    # rpm=None이면 기존 단일 baseline 동작과 동일(하위호환).
    vals = np.array([1.0, 1.1, 0.9, 1.0, 5.0])
    assert np.allclose(anomaly.robust_z(vals, baseline_n=4),
                       anomaly.robust_z(vals, baseline_n=4, rpm=None))
