"""상시 기록 폴더 판독 — 세션 목록, 일일 압력·온도 로그, 토출압 계단, 표본 세션의 운전 판독."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from impeller_pinn.epoch_inventory import inventory, level_steps, sample_sessions

FS = 51200.0


def _session(folder, stamp, f_rot=58.2, amp=8.0, n=25600):
    t = np.arange(n) / FS
    rng = np.random.default_rng(int(stamp[-6:]))
    data = np.vstack([amp * np.sin(2 * np.pi * f_rot * t) + 0.1 * rng.standard_normal(n)
                      for _ in range(7)])
    np.savez(folder / f"{stamp}_raw.npz", data=data, sample_rate=np.float64(FS),
             unit=np.array("m/s^2"), timestamp=np.array(stamp))


def _logs(folder, day, start, minutes, pd_of):
    with open(folder / f"pressure_{day}.csv", "w") as f:
        f.write("timestamp,p_discharge_bar,p_suction_bar\n")
        for k in range(minutes):
            t = start + timedelta(minutes=k)
            f.write(f"{t:%Y-%m-%d %H:%M:%S},{pd_of(t):.5f},-0.00118\n")
    with open(folder / f"temperature_{day}.csv", "w") as f:
        f.write("timestamp,ai0_C,ai1_C\n")
        for k in range(minutes):
            t = start + timedelta(minutes=k)
            f.write(f"{t:%Y-%m-%d %H:%M:%S},41.8,31.8\n")


@pytest.fixture
def folder(tmp_path):
    step = datetime(2026, 10, 15, 12, 0, 0)
    pd_of = lambda t: 0.084 if t < step else 0.004          # noqa: E731
    for day, d0 in (("20261014", datetime(2026, 10, 14, 0, 0, 0)), ("20261015", datetime(2026, 10, 15, 0, 0, 0))):
        _logs(tmp_path, day, d0, 24 * 60, pd_of)
        for hh in (6, 12, 18):
            _session(tmp_path, f"{day}_{hh:02d}0000")
    return tmp_path


def test_level_steps_find_one_step_and_its_levels():
    t = np.array([np.datetime64("2026-10-14T00:00") + np.timedelta64(k, "m") for k in range(120)])
    v = np.where(np.arange(120) < 61, 0.084, 0.004) + 0.003 * np.sin(np.arange(120))
    ev = level_steps(t, v, k=10, threshold=0.05)
    assert len(ev) == 1
    assert ev[0]["time"] == "2026-10-14T01:01:00"
    assert ev[0]["before"] == pytest.approx(0.084, abs=0.004) and ev[0]["after"] == pytest.approx(0.004, abs=0.004)


def test_level_steps_ignore_noise_below_the_threshold():
    t = np.array([np.datetime64("2026-10-14T00:00") + np.timedelta64(k, "m") for k in range(120)])
    v = 0.084 + 0.02 * np.sin(np.arange(120))
    assert level_steps(t, v, k=10, threshold=0.05) == []


def test_sample_sessions_pick_two_per_day():
    names = [f"2026101{d}_{h:02d}0000_raw.npz" for d in (4, 5) for h in range(0, 24, 3)]
    picked = sample_sessions(names, per_day=2)
    assert [p[:8] for p in picked] == ["20261014", "20261014", "20261015", "20261015"]
    assert len(set(picked)) == 4


def test_inventory_counts_sessions_finds_the_discharge_step_and_reads_running_state(folder, tmp_path):
    out = tmp_path / "inv.json"
    res = inventory(folder, out=out, per_day=2)
    assert res["n_sessions"] == 6
    assert res["sessions_per_day"] == {"20261014": 3, "20261015": 3}
    assert res["first"] == "20261014_060000" and res["last"] == "20261015_180000"
    steps = res["pressure"]["p_d_steps"]
    assert len(steps) == 1 and steps[0]["time"].startswith("2026-10-15T12:00")
    assert res["pressure"]["daily"]["20261014"]["p_d_median"] == pytest.approx(0.084, abs=1e-6)
    assert res["temperature"]["daily"]["20261015"]["Tw_median"] == pytest.approx(31.8, abs=1e-6)
    for s in res["samples"]:
        assert s["running"] and s["f_rot_Hz"] == pytest.approx(58.2, abs=2.1)
        assert s["ch0_rms"] == pytest.approx(8.0 / np.sqrt(2), rel=0.02)
    assert out.exists()
