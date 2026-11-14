"""중앙 레지스트리: 폴더→장비config 해석, base 경로(env override), 폴더별 산출물 경로."""
from __future__ import annotations
import os
from pathlib import Path
import yaml
from .equipment import EquipmentConfig
from .dataio import safe_resolve

_KNOWN = {"equipment_type", "channel_labels", "rpm_lo", "rpm_hi", "baseline_n",
          "sensitivity", "bearings", "env_band", "calibration", "activity_floor",
          "baseline_from", "z_rel_floor", "pwm_notch", "resonances", "rpm_binned", "rpm_bin"}


def _registry_path() -> Path:
    return Path(__file__).with_name("registry.yaml")


def load_registry(path=None) -> dict:
    p = Path(path) if path else _registry_path()
    if not p.exists():
        return {"base": "/mnt/nas/data", "defaults": {}, "folders": {}}
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_base(reg=None) -> str:
    reg = reg if reg is not None else load_registry()
    return os.environ.get("EQUIP_DATA_BASE") or reg.get("base", "/mnt/nas/data")


def resolve_config(folder: str, reg=None) -> EquipmentConfig:
    reg = reg if reg is not None else load_registry()
    defaults = reg.get("defaults", {}) or {}
    entry = (reg.get("folders", {}) or {}).get(folder, {}) or {}
    merged = {**defaults, **entry}
    extra = {k: v for k, v in merged.items() if k not in _KNOWN}
    return EquipmentConfig(
        equipment_type=merged.get("equipment_type", "generic"),
        channel_labels=merged.get("channel_labels", {}) or {},
        rpm_lo=float(merged.get("rpm_lo", 20.0)),
        rpm_hi=float(merged.get("rpm_hi", 120.0)),
        baseline_n=int(merged.get("baseline_n", 10)),
        sensitivity=float(merged.get("sensitivity", 3.0)),
        bearings=merged.get("bearings", {}) or {},
        env_band=tuple(float(x) for x in merged.get("env_band", (2000.0, 10000.0))),
        calibration=merged.get("calibration", {}) or {},
        activity_floor=float(merged.get("activity_floor", 0.05)),
        baseline_from=str(merged.get("baseline_from", "") or ""),
        z_rel_floor=float(merged.get("z_rel_floor", 0.0)),
        pwm_notch=merged.get("pwm_notch", {}) or {},
        resonances=merged.get("resonances", {}) or {},
        rpm_binned=bool(merged.get("rpm_binned", True)),
        rpm_bin=float(merged.get("rpm_bin", 900.0)),
        extra=extra,
    )


def folder_path(folder: str, reg=None) -> Path:
    """base 하위로만 해석(경로 탈출 차단)."""
    return safe_resolve(get_base(reg), folder)


def artifacts_dir(folder: str) -> Path:
    slug = folder.strip("/").replace("/", "__").replace(" ", "_")
    d = Path(__file__).with_name("artifacts") / slug
    d.mkdir(parents=True, exist_ok=True)
    return d
