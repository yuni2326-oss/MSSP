"""토출 부하 호스 설계 — 정격점(1.0 L/s·30 m)을 만드는 호스 마찰 + 트림밸브 배치 산출.

09-02 예비시험에서 무스로틀 양정이 0.86 m(정격의 2.9 %)였던 원인은 토출에 부하가 없는
자유순환이었다(상태 문서 §3.5·§3.6). 정격점을 만들려면 1.0 L/s에서 약 29 m(2.84 bar)를
더 잃게 해야 한다. 밸브 하나로 전부 떨어뜨리면 σ_ISA ≈ 1.36으로 밸브가 캐비해 09-02의
흡입 밸브(ch2 +20 dB)와 같은 오염을 토출측에 만든다. 그래서 부하의 대부분을 ¾″ 호스의
분산 마찰에 맡기고, 트림밸브는 배압이 받쳐 주는 지점에서 미세조절과 흡입 스로틀 보상만
맡는 배치를 쓴다. 이 모듈은 그 배치의 길이·트림 몫·캐비 여유·차압 유량계 분해능을 산출한다.

  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/load_hose_design.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from impeller_pinn.thermal_analysis import p_v_kPa, rho_water

GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "load_hose_design.json"
G = 9.80665
P_ATM_BAR = 1.01325
Q_RATED = 1.0e-3            # 3.6 m³/h
H_ADDED_M = 30.0 - 0.85     # 정격 양정 − 현재 정수두+마찰
EPS_HOSE = 1.0e-5           # 평활 고무/특편사 호스(보수적)
SUM_K = 3.0                 # 진입 0.4 + 바브 2×0.7 + 티 0.3 + 침수 출구 1.0
Z_SUB_M = 0.8               # 트림밸브가 탱크 수면보다 낮은 높이(바닥 설치)
SUCTION_DROP_BAR = 0.8      # 흡입을 −0.8 bar(g)까지 조일 때 트림이 돌려줘야 할 몫
MIN_COMP_BAR = 0.85         # 보상폭 판정 하한(여유 포함)
MIN_SIGMA = 2.5             # 트림 캐비 여유 판정 하한(ISA 정의)


def water_nu(T_C):
    """동점성계수 ν [m²/s] — Vogel 식 점도 / 밀도."""
    mu = 2.414e-5 * 10 ** (247.8 / (float(T_C) + 273.15 - 140.0))
    return mu / float(rho_water(T_C))


def friction_factor(Re, eps, D):
    """Colebrook 마찰계수(고정점 반복)."""
    f = 0.02
    for _ in range(60):
        f = (-2.0 * np.log10(eps / (3.7 * D) + 2.51 / (Re * np.sqrt(f)))) ** -2
    return float(f)


def hose_loss(Q=Q_RATED, D=0.019, T_C=38.0, eps=EPS_HOSE):
    """호스 한 종류의 유속·Re·마찰계수와 단위길이 손실 [m/m, bar/m], 속도수두 [bar]."""
    rho = float(rho_water(T_C))
    v = Q / (np.pi * D ** 2 / 4)
    Re = v * D / water_nu(T_C)
    f = friction_factor(Re, eps, D)
    hf_m = f / D * v ** 2 / (2 * G)
    return {"v": v, "Re": Re, "f": f, "hf_m_per_m": hf_m,
            "hf_bar_per_m": rho * G * hf_m / 1e5, "vh_bar": rho * v ** 2 / 2 / 1e5,
            "vh_m": v ** 2 / (2 * G), "rho": rho}


def sigma_isa(P1_abs, P2_abs, pv_abs):
    """밸브 캐비테이션 지수(ISA): σ = (P1 − p_v) / (P1 − P2)."""
    return (P1_abs - pv_abs) / (P1_abs - P2_abs)


def layout(L_total, trim_at=10.0, Q=Q_RATED, D=0.019, T_C=38.0):
    """캠록 → trim_at [m]에 트림밸브 → L_total [m]에서 탱크 출구인 배치를 평가한다."""
    h = hose_loss(Q, D, T_C)
    rho = h["rho"]
    loss_m = h["hf_m_per_m"] * L_total + SUM_K * h["vh_m"]
    trim_bar = rho * G * (H_ADDED_M - loss_m) / 1e5
    after_m = h["hf_m_per_m"] * (L_total - trim_at) + 1.0 * h["vh_m"] + 0.7 * h["vh_m"]
    P2 = P_ATM_BAR + rho * G * (after_m + Z_SUB_M) / 1e5
    P1 = P2 + trim_bar
    pv = float(p_v_kPa(T_C)) / 100.0
    sig = sigma_isa(P1, P2, pv) if trim_bar > 0 else float("inf")
    return {"L_total_m": L_total, "trim_at_m": trim_at, "hose_loss_m": loss_m,
            "trim_bar": trim_bar, "sigma_trim": sig,
            "comp_ok": bool(trim_bar >= MIN_COMP_BAR), "sigma_ok": bool(sig >= MIN_SIGMA)}


def valve_alone_sigma(T_C=38.0):
    """부하 전량(2.84 bar)을 밸브 하나로 떨어뜨리고 뒤가 대기압 근처일 때의 σ."""
    rho = float(rho_water(T_C))
    P2 = P_ATM_BAR + rho * G * Z_SUB_M / 1e5
    P1 = P2 + rho * G * H_ADDED_M / 1e5
    return sigma_isa(P1, P2, float(p_v_kPa(T_C)) / 100.0)


def dp_flowmeter(seg_m=10.0, Q=Q_RATED, D=0.019, T_C=38.0, quant_bar=0.00129):
    """밸브 없는 계측 구간의 차압 p_d − p2와 유량 분해능. ΔP ∝ Q^1.75(난류 평활관)."""
    h = hose_loss(Q, D, T_C)
    dp = h["hf_bar_per_m"] * seg_m + (0.7 + 0.4) * h["vh_bar"]
    res_bar = np.sqrt(2.0) * quant_bar
    return {"dp_bar": dp, "q_resolution_pct": res_bar / dp / 1.75 * 100.0,
            "temp_coeff_pct_per_C": 0.25 * 0.021 * 100.0}


def required_cv(Q_Lps, dp_bar):
    """밸브에 필요한 Cv(US gpm / √psi) — 니들밸브가 부적합한 근거."""
    return (Q_Lps * 15.850) / np.sqrt(dp_bar * 14.504)


def main(log=print):
    rows = []
    log("호스 내경별 @1.0 L/s, 38 °C")
    for name, D in (("1/2", 0.0127), ("5/8", 0.0160), ("3/4", 0.0190), ("1", 0.0254)):
        h = hose_loss(D=D)
        rows.append({"hose_in": name, "D_m": D, **{k: h[k] for k in ("v", "Re", "f", "hf_m_per_m")}})
        log(f"  {name:>3s}″  v {h['v']:.2f} m/s  Re {h['Re']:.0f}  h_f {h['hf_m_per_m']:.3f} m/m")
    lays = [layout(L) for L in (20.0, 22.0, 25.0, 30.0)]
    log("\n¾″, 트림밸브 10 m 지점")
    for r in lays:
        log(f"  총 {r['L_total_m']:4.0f} m  트림 {r['trim_bar']:.2f} bar  σ {r['sigma_trim']:.2f}  "
            f"보상폭 {'OK' if r['comp_ok'] else '부족'}  σ {'OK' if r['sigma_ok'] else '부족'}")
    va = valve_alone_sigma(); fm = dp_flowmeter()
    log(f"\n밸브 단독 σ = {va:.2f}")
    log(f"계측 구간 10 m: p_d − p2 = {fm['dp_bar']:.3f} bar, Q 분해능 {fm['q_resolution_pct']:.2f} %")
    log(f"필요 Cv: 1.0 L/s·1.0 bar {required_cv(1.0, 1.0):.1f}, 0.2 bar {required_cv(1.0, 0.2):.1f}")
    out = {"conditions": {"Q_m3s": Q_RATED, "T_C": 38.0, "H_added_m": H_ADDED_M, "eps_m": EPS_HOSE,
                          "sum_K": SUM_K, "z_sub_m": Z_SUB_M, "min_comp_bar": MIN_COMP_BAR,
                          "min_sigma": MIN_SIGMA},
           "hoses": rows, "layouts": lays, "valve_alone_sigma": va, "dp_flowmeter": fm,
           "required_cv": {"dp_1bar": required_cv(1.0, 1.0), "dp_0p2bar": required_cv(1.0, 0.2)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f"\nwrote {OUT}")
    return out


if __name__ == "__main__":
    main()
