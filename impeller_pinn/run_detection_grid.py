"""§3.6/§4.4 정량 검출연구 실제 실행 — 논문 Table 6/Fig 8/최소검출심각도 산출.
  PYTHONPATH=/home/super/project ~/equip-venv/bin/python impeller_pinn/run_detection_grid.py
결과를 JSON으로 scratch에 저장(논문 그림/표 생성 단계에서 재사용)."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path

from equipment_analysis import dataio
from impeller_pinn.detection_sweep import run_grid, roc_curve

DATA_ROOT = "/mnt/nas/data/GY_DATA/pump data"
SCRATCH = Path(tempfile.gettempdir()) / "impeller_pinn" / "detsweep_run"
GEN = Path(__file__).resolve().parents[1] / "docs" / "_generated"     # 저장소 기준 — 모노리포에서는 /home/super/project/docs/_generated
OUT = GEN / "data" / "paper1" / "detection_results.json"


def main():
    baseline = dataio.discover_sessions(DATA_ROOT, date_from="20260707_112346",
                                        date_to="20260708_235959")[:10]
    # 초기 시도(07-15~19)는 ch1_spec_flatness 자연 드리프트(07-11 전후 계단변화, 0.045->0.06)로
    # 오경보율 100%(진단으로 확인) — baseline과 완전히 무관한 장기 드리프트가 z를 지배해 주입
    # 여부와 상관없이 모든 held-out이 이미 플래그됨. baseline과 시간적으로 가까운(같은 주,
    # 드리프트 이전) 창으로 교정 — 오경보율이 25특징 OR-규칙의 정상적 다중비교 수준(37.5%,
    # 별도 검증)으로 회복.
    held_pool = dataio.discover_sessions(DATA_ROOT, date_from="20260707_162416",
                                         date_to="20260710_235959")
    held = held_pool[::max(1, len(held_pool) // 16)][:16]
    print(f"baseline: {[s.stamp for s in baseline]}")
    print(f"held-out ({len(held)}): {[s.stamp for s in held]}")

    result = run_grid(baseline, held, SCRATCH)
    print("\n=== ROC at d_max=0.3, snr=-10dB (near the detection knee) ===")
    roc = roc_curve(held, result["base_table"], result["feature_cols"], SCRATCH,
                    d_max=0.3, snr_db=-10.0)
    for th, tpr, fpr in zip(roc["thresholds"], roc["tpr"], roc["fpr"]):
        print(f"  th={th:5.2f}  TPR={tpr:5.1f}%  FPR={fpr:5.1f}%")

    payload = {
        "baseline_stamps": [s.stamp for s in baseline],
        "held_out_stamps": [s.stamp for s in held],
        "false_alarm_rate": result["false_alarm_rate"],
        "table6": {str(k): v for k, v in result["table6"].items()},
        "roc": roc,
        "feature_cols": result["feature_cols"],
        "cells_summary": {f"{k[0]}_{k[1]}": {"det_pct": v["det_pct"],
                                             "zmax_median": float(__import__("numpy").median(v["zmax"]))}
                         for k, v in result["cells"].items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {OUT}")
    print("=== DETECTION_GRID DONE ===")


if __name__ == "__main__":
    main()
