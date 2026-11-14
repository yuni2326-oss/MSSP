import pytest
from equipment_analysis import registry
from equipment_analysis.equipment import EquipmentConfig

_REG = {
    "base": "/mnt/nas/data",
    "defaults": {"equipment_type": "generic", "rpm_lo": 20, "rpm_hi": 120,
                 "baseline_n": 10, "sensitivity": 3.0},
    "folders": {"GY_DATA/pump data": {"equipment_type": "pump", "sensitivity": 2.5,
                                      "channel_labels": {"ch0": "모터DE"},
                                      "bearings": {"ch3": {"model": "6203"},
                                                   "ch4": {"model": "6204"}},
                                      "env_band": [2000, 9000],
                                      "calibration": {"before": "20260707_112346", "scale": 10.0},
                                      "activity_floor": 0.0,
                                      "baseline_from": "20260707_112346", "z_rel_floor": 0.05}},
}

def test_resolve_registered_folder_merges_defaults():
    cfg = registry.resolve_config("GY_DATA/pump data", _REG)
    assert isinstance(cfg, EquipmentConfig)
    assert cfg.equipment_type == "pump"
    assert cfg.sensitivity == 2.5           # folder override
    assert cfg.baseline_n == 10             # from defaults
    assert cfg.channel_labels["ch0"] == "모터DE"

def test_resolve_reads_bearings_and_env_band():
    cfg = registry.resolve_config("GY_DATA/pump data", _REG)
    assert cfg.bearings == {"ch3": {"model": "6203"}, "ch4": {"model": "6204"}}
    assert cfg.env_band == (2000.0, 9000.0)          # 리스트→튜플 정규화
    # bearings/env_band는 알려진 키라 extra로 새지 않는다.
    assert "bearings" not in cfg.extra and "env_band" not in cfg.extra


def test_resolve_reads_calibration_and_activity_floor():
    cfg = registry.resolve_config("GY_DATA/pump data", _REG)
    assert cfg.calibration == {"before": "20260707_112346", "scale": 10.0}
    assert cfg.activity_floor == 0.0
    assert "calibration" not in cfg.extra and "activity_floor" not in cfg.extra


def test_resolve_reads_baseline_window():
    cfg = registry.resolve_config("GY_DATA/pump data", _REG)
    assert cfg.baseline_from == "20260707_112346"
    assert cfg.z_rel_floor == 0.05
    assert "baseline_from" not in cfg.extra and "z_rel_floor" not in cfg.extra
    d = registry.resolve_config("unknown/dir", _REG)
    assert d.baseline_from == "" and d.z_rel_floor == 0.0     # 미설정 → 현행 동작


def test_resolve_unregistered_folder_uses_defaults():
    cfg = registry.resolve_config("unknown/dir", _REG)
    assert cfg.equipment_type == "generic"
    assert cfg.sensitivity == 3.0
    assert cfg.bearings == {}                          # 미설정 → 빈 dict(포락선 미계산)
    assert cfg.env_band == (2000.0, 10000.0)           # 기본 대역
    assert cfg.calibration == {}                       # 미설정 → 보정 없음
    assert cfg.activity_floor == 0.05                  # 기본(model.feature_matrix 기본과 일치)

def test_get_base_env_override(monkeypatch):
    monkeypatch.setenv("EQUIP_DATA_BASE", "/nas/data")
    assert registry.get_base(_REG) == "/nas/data"
    monkeypatch.delenv("EQUIP_DATA_BASE")
    assert registry.get_base(_REG) == "/mnt/nas/data"

def test_folder_path_blocks_escape(monkeypatch):
    monkeypatch.setenv("EQUIP_DATA_BASE", "/mnt/nas/data")
    with pytest.raises(ValueError):
        registry.folder_path("../../etc", _REG)
