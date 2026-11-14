# equipment_analysis/train_model.py
"""폴더 단위 이상탐지 모델 학습 CLI.

흐름: 특징 캐시 빌드(env 포함) → 수집에러 중복(특징 완전동일 세션) 검출·제외 →
IsolationForest 학습 → 아티팩트 저장. Tool `model_health`가 읽는 경로에 저장한다.

사용:
    PYTHONPATH=/path/to/project python -m equipment_analysis.train_model "GY_DATA/pump data"
옵션:
    --no-dedup        수집에러 중복 제외 비활성(전체 학습)
    --keep-first      중복 그룹당 대표 1개는 유지(기본은 그룹 전체 제외)
    --quarantine      중복 원본 raw/fft 파일을 격리 이동(.bak) + 캐시 정리(파괴적, 되돌리기 가능)
    --contamination   IsolationForest contamination('auto' 또는 0~0.5 실수, 기본 auto)

⚠️ 모델 pickle 호환: 이 스크립트는 실행 환경의 sklearn 버전으로 저장한다. 모델을 로드하는
   컨테이너(Tool `model_health`=open-webui)와 sklearn 버전이 다르면 InconsistentVersionWarning /
   오작동 위험 → 소비 컨테이너 안에서 실행할 것.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

from . import registry, cache, model, dataio


def quarantine_sessions(data_root, stamps, quarantine_dir) -> int:
    """중복 세션의 raw/fft 파일을 quarantine_dir로 이동하며 `.bak` 접미사를 붙인다.
    같은 파일시스템이면 rename이라 즉시. 접미사로 discover_sessions/list_data_folders 스캔을
    회피(격리폴더가 데이터폴더로 안 잡힘). 이동한 파일 수 반환."""
    data_root = Path(data_root)
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for st in stamps:
        files = list(data_root.glob(f"{st}_raw.csv")) + list(data_root.glob(f"{st}_fft_ch*.csv"))
        for f in files:
            shutil.move(str(f), str(quarantine_dir / (f.name + ".bak")))
            moved += 1
    return moved


def train_folder(folder: str, reg=None, *, cache_path=None, model_path=None,
                 dedup: bool = True, keep_first: bool = False, quarantine: bool = False,
                 contamination="auto") -> dict:
    """folder의 특징표를 빌드→(중복 제외)→학습→저장. 요약 dict 반환."""
    reg = reg if reg is not None else registry.load_registry()
    cfg = registry.resolve_config(folder, reg)
    data_root = str(registry.folder_path(folder, reg))
    cache_path = Path(cache_path) if cache_path else registry.artifacts_dir(folder) / "features.csv"
    model_path = Path(model_path) if model_path else registry.artifacts_dir(folder) / "health_model.joblib"

    table = cache.build_feature_table(data_root, cache_path,
                                      rpm_range=(cfg.rpm_lo, cfg.rpm_hi),
                                      bearings=cfg.bearings, env_band=cfg.env_band,
                                      calibration=cfg.calibration, pwm_notch=cfg.pwm_notch,
                                      resonances=cfg.resonances)
    n_total = len(table)
    dups = cache.duplicate_stamps(table, keep_first=keep_first) if dedup else []
    if dups:
        if quarantine:
            qdir = Path(data_root).parent / f"_DUP_QUARANTINE_{Path(data_root).name.replace(' ', '_')}"
            quarantine_sessions(data_root, dups, qdir)
        table = table[~table["stamp"].isin(dups)].reset_index(drop=True)
        if quarantine:
            table.to_csv(cache_path, index=False)   # 캐시도 정리(격리와 정합)

    art = model.train(table, contamination=contamination, activity_floor=cfg.activity_floor)
    model.save_artifact(art, model_path)
    return {
        "folder": folder,
        "n_sessions": int(n_total),
        "n_duplicates": len(dups),
        "n_train": int(art["n_train"]),
        "n_features": len(art["feature_names"]),
        "env_features": [n for n in art["feature_names"] if "_env_" in n],
        "model_path": str(model_path),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="설비 이상탐지 모델 학습(수집에러 중복 제외 포함)")
    p.add_argument("folder", help="데이터 폴더(base 상대), 예: 'GY_DATA/pump data'")
    p.add_argument("--no-dedup", action="store_true", help="수집에러 중복 제외 비활성")
    p.add_argument("--keep-first", action="store_true", help="중복 그룹당 대표 1개 유지")
    p.add_argument("--quarantine", action="store_true", help="중복 원본 파일 격리 이동 + 캐시 정리")
    p.add_argument("--contamination", default="auto", help="'auto' 또는 0~0.5 실수")
    a = p.parse_args(argv)
    try:
        contamination = float(a.contamination)
    except ValueError:
        contamination = a.contamination
    res = train_folder(a.folder, dedup=not a.no_dedup, keep_first=a.keep_first,
                       quarantine=a.quarantine, contamination=contamination)
    print(f"[train_model] {res['folder']}: 세션 {res['n_sessions']}개 "
          f"(중복 {res['n_duplicates']}개 제외) → 학습표본 {res['n_train']}, "
          f"특징 {res['n_features']}개 (env {len(res['env_features'])}개) → {res['model_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
