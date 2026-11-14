"""npz 리더 테스트 — 합성 npz로 스키마·단위예외·전기마스크·대역RMS 검증(실데이터 불필요)."""
import numpy as np
import pytest

from equipment_analysis.npz_io import (
    ACCEL_UNIT, MIC_CHANNEL, MIC_UNIT, PUMP_RMS_FLOOR, amplitude_spectrum, band_rms, channel_units,
    discover_npz_sessions, electrical_mask, is_pump_running, load_npz_raw,
)

FS = 51200.0


def _write(tmp_path, stamp="20260721_120000", n_ch=7, n=8192):
    t = np.arange(n) / FS
    data = np.stack([np.sin(2 * np.pi * (100 * (i + 1)) * t) for i in range(n_ch)]).astype(np.float32)
    p = tmp_path / f"{stamp}_raw.npz"
    np.savez(p, data=data, sample_rate=np.float64(FS), unit=np.str_("m/s^2"), timestamp=np.str_(stamp))
    return p


def test_channel_units_flags_microphone_exception():
    u = channel_units(7)
    assert u[MIC_CHANNEL] == MIC_UNIT == "Pa"
    assert all(x == ACCEL_UNIT for i, x in enumerate(u) if i != MIC_CHANNEL)


def test_load_npz_raw_schema_and_unit_override(tmp_path):
    p = _write(tmp_path)
    chans, fs, meta = load_npz_raw(p)
    assert fs == FS
    assert set(chans) == {f"ch{i}" for i in range(7)}
    assert chans["ch0"].dtype == np.float64          # float32 -> float64 승격
    assert meta["unit_field"] == "m/s^2"             # 파일이 말하는 값
    assert meta["units"][6] == "Pa"                  # 실제 단위(파일 오라벨 교정)
    assert meta["units"][0] == "m/s^2"
    assert meta["n_channels"] == 7


def test_discover_npz_sessions_and_date_filter(tmp_path):
    _write(tmp_path, "20260720_180159")
    _write(tmp_path, "20260721_100038")
    _write(tmp_path, "20260722_090000")
    (tmp_path / "20260721_100038_raw.csv").write_text("not npz")   # CSV는 무시
    alls = discover_npz_sessions(tmp_path)
    assert [s.stamp for s in alls] == ["20260720_180159", "20260721_100038", "20260722_090000"]
    post = discover_npz_sessions(tmp_path, date_from="20260721_100038")
    assert [s.stamp for s in post] == ["20260721_100038", "20260722_090000"]


def test_electrical_mask_excludes_comb_and_the_8180_sideband():
    freq = np.arange(0, 25000, 1.0)
    keep = electrical_mask(freq)
    # PWM 콤 k=1..3 중심은 제외
    for k in (1, 2, 3):
        assert not keep[int(round(7820.0 * k))]
    # carrier+BPF 사이드밴드(콤 규칙으로는 안 걸리는데 ch0 최대선) 제외
    assert not keep[8180]
    # 구조 반송파 대역(2.9-3.4k)은 유지
    assert keep[3000] and keep[3200]


def test_band_rms_respects_mask_and_scales_with_amplitude():
    n = 8192
    t = np.arange(n) / FS
    x1 = np.sin(2 * np.pi * 3000 * t)
    f, m1 = amplitude_spectrum(x1, FS)
    r1 = band_rms(f, m1, (2900, 3400))
    r2 = band_rms(f, 2 * m1, (2900, 3400))
    assert r1 > 0
    assert r2 == pytest.approx(2 * r1, rel=1e-9)
    # 3000Hz 성분을 마스크로 지우면 대역 RMS가 급감
    keep = np.abs(f - 3000.0) > 50.0
    assert band_rms(f, m1, (2900, 3400), keep) < 0.05 * r1


def test_is_pump_running_excludes_pump_off():
    # 정상 운전: ch0 RMS가 floor 위 (실측 정상 중앙값 ~7)
    running = {"ch0": np.full(1000, 7.0)}
    assert is_pump_running(running)
    # 펌프 비운전(드라이브 통전·모터 정지): ch0 RMS ~0.87 (실측 20260727_082305/182253)
    off = {"ch0": np.full(1000, 0.87)}
    assert not is_pump_running(off)
    # 경계: floor 바로 아래/위
    assert not is_pump_running({"ch0": np.full(10, PUMP_RMS_FLOOR - 0.01)})
    assert is_pump_running({"ch0": np.full(10, PUMP_RMS_FLOOR + 0.01)})
    # ch0 없으면 보수적으로 운전 취급(제외하지 않음)
    assert is_pump_running({"ch1": np.zeros(10)})
