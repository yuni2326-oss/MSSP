"""장비 설정 + 장비타입별 진단 규칙 확장점."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EquipmentConfig:
    equipment_type: str = "generic"
    channel_labels: dict = field(default_factory=dict)
    rpm_lo: float = 20.0
    rpm_hi: float = 120.0
    baseline_n: int = 10
    sensitivity: float = 3.0
    bearings: dict = field(default_factory=dict)      # 채널→베어링 스펙 {ch: {model|n,d,D,theta}} (포락선 분석)
    env_band: tuple = (2000.0, 10000.0)               # 포락선 밴드패스 대역(Hz)
    calibration: dict = field(default_factory=dict)   # {before: stamp, scale: float} 이전 세션 raw/fft 스케일 보정
    activity_floor: float = 0.05                      # 죽은 채널 제외 임계(최대채널 rms 대비). 0이면 전 채널 포함
    baseline_from: str = ""                           # robust_z baseline 앵커 스탬프(재장착/보정 이후 정상구간). ""=맨앞
    z_rel_floor: float = 0.0                          # robust_z scale 상대 하한(rel·|median|). z 폭증 방지
    pwm_notch: dict = field(default_factory=dict)     # VFD PWM 콤 제거 {carrier, n_harm, width}(Hz). {}=미적용
    resonances: dict = field(default_factory=dict)    # 공진 추적 {ch: [f0..]}(Hz, OMA/PINN 검증). 손상→하향이동 포착
    rpm_binned: bool = True                            # rpm 구간별 baseline(가변속 F1). rpm 컬럼 있을 때 적용
    rpm_bin: float = 900.0                             # rpm 구간폭(=15Hz)
    extra: dict = field(default_factory=dict)   # 장비별 추가 파라미터(pole_count, line_freq_hz 등)


def _generic_findings(evidence: dict) -> list[str]:
    out: list[str] = []
    for f in evidence.get("flagged_features", []):
        out.append(f"{f['feature']} 기준선 이탈 (최대 z {f['z_peak']:+.1f}, 최근 z {f['z']:+.1f}, 이탈 {f['n_flagged']}회)")
    return out


_DEFECT_KR = {"bpfo": "외륜", "bpfi": "내륜", "bsf": "볼/롤러", "ftf": "케이지"}


def _bearing_findings(cfg: "EquipmentConfig", evidence: dict) -> list[str]:
    """포락선 결함주파수 특징(chN_env_{bpfo|bpfi|bsf|ftf}) 이탈 → 베어링 결함 소견.
    채널→베어링(cfg.bearings)·라벨(cfg.channel_labels) 결합. bearings 설정 채널만."""
    out: list[tuple[float, str]] = []
    for f in evidence.get("flagged_features", []):
        name = f["feature"]
        parts = name.split("_env_")
        if len(parts) != 2:
            continue
        ch, defect = parts[0], parts[1]
        if ch not in cfg.bearings or defect not in _DEFECT_KR:
            continue
        label = cfg.channel_labels.get(ch, ch)
        model = cfg.bearings[ch].get("model", "")
        model_s = f"{model} " if model else ""
        out.append((abs(f["z_peak"]),
                    f"{name} 이탈(최대 z={f['z_peak']:+.1f}, {f['n_flagged']}회) → "
                    f"{label} {model_s}베어링 {_DEFECT_KR[defect]} 결함 의심"))
    out.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in out]


def _motor_findings(cfg: "EquipmentConfig", evidence: dict) -> list[str]:
    """모터 고유 규칙 진입점(현재 no-op). 모터 데이터/사양(pole_count, line_freq_hz,
    베어링 모델) 확보 후 베어링 결함주파수(BPFO/BPFI/BSF/FTF)·라인주파수 사이드밴드
    규칙을 여기에 채운다. hook은 지금부터 호출되므로 규칙만 추가하면 즉시 반영된다."""
    return []


def _pattern_findings(cfg: "EquipmentConfig", evidence: dict) -> list[str]:
    """결함 패턴 규칙(fable 정정, ISO 13373-3 계열): 이탈 특징 조합→결함유형 해석 힌트.
    모두 '의심'(evidence)이지 단정 아님, 혼동원인 가드 포함. z는 rpm-binned robust-z(F1)."""
    idx: dict[str, list] = {}
    for f in evidence.get("flagged_features", []):
        name = f["feature"]
        if name.startswith("ch") and "_" in name:
            ch, base = name.split("_", 1)
        else:
            ch, base = "", name
        idx.setdefault(base, []).append((f["z_peak"], ch))

    def top(base):
        return max(idx[base], key=lambda t: abs(t[0])) if base in idx else None

    def lbl(ch):
        return cfg.channel_labels.get(ch, ch) if ch else ""

    out: list[tuple[float, str]] = []
    if "ord_2x_ratio" in idx:
        z, ch = top("ord_2x_ratio")
        if z > 0:
            out.append((abs(z), f"{lbl(ch)} 2x/1x 상승(z={z:+.1f}) → 평행 정렬불량/커플링 마모 의심 "
                        "(가드: 전자기 2×f_out·BSF≈2×·느슨함과 혼동 가능, 축방향 센서 없어 각도 정렬불량은 약관측)"))
    if "ord_sum_ratio" in idx and "ord_0p5x_ratio" in idx:
        z = max(abs(top("ord_sum_ratio")[0]), abs(top("ord_0p5x_ratio")[0]))
        out.append((z, "하모닉합↑ + 0.5x 서브싱크↑ → 기계적 느슨함 의심 "
                    "(가드: 부분유량 재순환도 유사 서브싱크 생성)"))
    if "ord_bpf_ratio" in idx:
        z, ch = top("ord_bpf_ratio")
        out.append((abs(z), f"{lbl(ch)} BPF(6×)/1x 상승(z={z:+.1f}) → 임펠러/베인·유동(off-BEP) 변화 의심 (운전점 의존)"))
    for b in ("hf_rms", "spec_flatness"):
        if b in idx:
            z, ch = top(b)
            out.append((abs(z), f"{lbl(ch)} HF 대역 변화({b} z={z:+.1f}, 양측) → HF 유압 이상(캐비테이션/재순환 가능; "
                        "유량데이터 없어 단정 불가, 심각도-비단조)"))
            break
    if "bandkurt" in idx:
        z, ch = top("bandkurt")
        if z > 0:
            out.append((abs(z), f"{lbl(ch)} 대역첨도↑(z={z:+.1f}) → 임펄스성 결함(초기 베어링/rub) 의심; 포락선 SNR과 대조"))
    for base in list(idx):
        if base.startswith("res") and base.endswith("_shift"):
            z, ch = top(base)
            if z < 0:                            # 공진 하향 이동만 열화 신호
                out.append((abs(z), f"{lbl(ch)} {base} 하향(z={z:+.1f}) → 구조 열화(균열·침식·마운트 느슨함) 의심; 공진 주파수 하강"))
    out.sort(key=lambda t: t[0], reverse=True)
    return [s for _, s in out]


def diagnose_hook(cfg: EquipmentConfig, evidence: dict) -> list[str]:
    """장비타입별 도메인 규칙 → 소견 리스트. 범용 규칙 + 결함패턴 규칙 + 장비별 규칙."""
    findings = _generic_findings(evidence)
    findings += _pattern_findings(cfg, evidence)
    if cfg.bearings:
        findings += _bearing_findings(cfg, evidence)
    if cfg.equipment_type == "motor":
        findings += _motor_findings(cfg, evidence)
    return findings
