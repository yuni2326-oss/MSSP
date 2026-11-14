from equipment_analysis.equipment import EquipmentConfig, diagnose_hook

def test_generic_findings_from_flagged():
    cfg = EquipmentConfig(equipment_type="pump")
    ev = {"flagged_features": [{"feature": "ch2_kurtosis", "z": 5.2, "z_peak": 5.2, "n_flagged": 3}]}
    out = diagnose_hook(cfg, ev)
    assert any("ch2_kurtosis" in s for s in out)

def test_motor_hook_is_present_but_noop_now():
    cfg = EquipmentConfig(equipment_type="motor")
    ev = {"flagged_features": []}
    # 모터 규칙은 아직 없음(데이터 확보 후) → 범용 소견만, 예외 없이 동작
    assert diagnose_hook(cfg, ev) == []


def test_bearing_findings_from_flagged_env():
    cfg = EquipmentConfig(
        equipment_type="pump",
        bearings={"ch4": {"model": "6204"}},
        channel_labels={"ch4": "모터 구동단(DE)"},
    )
    ev = {"flagged_features": [
        {"feature": "ch4_env_bpfo", "z": 5.2, "z_peak": 5.2, "n_flagged": 3},
        {"feature": "ch0_rms", "z": 4.0, "z_peak": 4.0, "n_flagged": 2},   # 베어링 특징 아님
    ]}
    out = diagnose_hook(cfg, ev)
    joined = " ".join(out)
    assert "모터 구동단(DE)" in joined and "6204" in joined and "외륜" in joined
    assert "ch4_env_bpfo" in joined
    # 베어링 소견은 env 특징에만(ch0_rms는 베어링 소견 대상 아님)
    assert not any("ch0_rms" in s and "베어링" in s for s in out)


def test_no_bearing_findings_when_unconfigured():
    cfg = EquipmentConfig(equipment_type="pump")   # bearings 미설정
    ev = {"flagged_features": [{"feature": "ch4_env_bpfo", "z": 5.2, "z_peak": 5.2, "n_flagged": 3}]}
    out = diagnose_hook(cfg, ev)
    assert not any("베어링" in s for s in out)      # 규칙 소견 없음(범용만)


def _ev(*feats):
    # feats: (feature, z_peak) → evidence dict
    return {"flagged_features": [{"feature": f, "z": z, "z_peak": z, "n_flagged": 3} for f, z in feats]}


def test_pattern_findings_misalignment_and_looseness():
    from equipment_analysis import equipment
    cfg = equipment.EquipmentConfig(equipment_type="pump", channel_labels={"ch1": "모터 연결부"})
    out = equipment._pattern_findings(cfg, _ev(("ch1_ord_2x_ratio", 6.0)))
    assert any("정렬불량" in s for s in out)
    out2 = equipment._pattern_findings(cfg, _ev(("ch1_ord_sum_ratio", 5.0), ("ch1_ord_0p5x_ratio", 4.0)))
    assert any("느슨함" in s for s in out2)


def test_pattern_findings_cavitation_and_structural():
    from equipment_analysis import equipment
    cfg = equipment.EquipmentConfig(equipment_type="pump", channel_labels={"ch2": "흡입측"})
    out = equipment._pattern_findings(cfg, _ev(("ch2_hf_rms", 5.0)))
    assert any("HF" in s and "캐비테이션" in s for s in out)
    # 공진 하향(음의 z) → 구조 열화
    out2 = equipment._pattern_findings(cfg, _ev(("ch2_res7462_shift", -6.0)))
    assert any("구조 열화" in s for s in out2)
    # 공진 상향(양의 z)은 구조 열화로 보지 않음
    out3 = equipment._pattern_findings(cfg, _ev(("ch2_res7462_shift", 6.0)))
    assert not any("구조 열화" in s for s in out3)


def test_diagnose_hook_includes_pattern_findings():
    from equipment_analysis import equipment
    cfg = equipment.EquipmentConfig(equipment_type="pump")
    findings = equipment.diagnose_hook(cfg, _ev(("ch1_ord_2x_ratio", 7.0)))
    assert any("정렬불량" in s for s in findings)
