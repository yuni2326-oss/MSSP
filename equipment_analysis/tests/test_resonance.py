"""구조 공진 주파수 추적 특징 테스트 (손상→공진 하향이동 포착)."""
import numpy as np
import pandas as pd

from equipment_analysis import features, dataio


def test_resonance_tracks_shifted_peak():
    # 앵커 7462Hz, 실제 피크는 7200Hz(손상으로 하향이동) → 추적 주파수·음(-)의 분율이동.
    freq = np.linspace(0, 12000, 6001)          # df=2Hz
    mag = np.zeros_like(freq)
    mag[int(np.argmin(np.abs(freq - 7200)))] = 1.0
    out = features.resonance_features(freq, mag, [7462.0], search_frac=0.06)
    assert abs(out["res7462_f"] - 7200) <= 4                    # 피크 추적
    assert out["res7462_shift"] < 0                            # 하향
    assert abs(out["res7462_shift"] - (7200 - 7462) / 7462) < 0.005


def test_resonance_no_energy_returns_anchor():
    # 대역 내 에너지 없으면 앵커 그대로·이동 0(정보 없음 → 이상 아님).
    freq = np.linspace(0, 12000, 6001)
    mag = np.zeros_like(freq)
    out = features.resonance_features(freq, mag, [7462.0, 8538.0])
    assert out["res7462_f"] == 7462.0 and out["res7462_shift"] == 0.0
    assert out["res8538_f"] == 8538.0 and out["res8538_shift"] == 0.0


def test_session_features_resonances(tmp_path):
    # session_features(resonances=...)가 채널별 추적특징 생성, None이면 미생성(하위호환).
    fs, n = 24000.0, 24000
    t = np.arange(n) / fs
    st = "20260716_120000"
    pd.DataFrame({"time_s": t, **{f"ch{i}": np.zeros(n) for i in range(3)}}).to_csv(
        tmp_path / f"{st}_raw.csv", index=False)
    # ch2 fft: 7300Hz에 피크(앵커 7462의 -2.2% 이동)
    fr = np.linspace(0, 12000, 6001)
    mg = np.zeros_like(fr); mg[int(np.argmin(np.abs(fr - 7300)))] = 1.0
    pd.DataFrame({"freq": fr, "mag": mg}).to_csv(tmp_path / f"{st}_fft_ch2.csv", index=False)
    sess = {s.stamp: s for s in dataio.discover_sessions(str(tmp_path))}[st]
    none = features.session_features(sess)
    tracked = features.session_features(sess, resonances={"ch2": [7462.0]})
    assert "ch2_res7462_f" not in none                          # None=미생성(하위호환)
    assert abs(tracked["ch2_res7462_f"] - 7300) <= 4            # 추적됨
    assert tracked["ch2_res7462_shift"] < 0
