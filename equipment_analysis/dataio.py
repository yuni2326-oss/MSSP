# pump_analysis/dataio.py
"""세션 검색, 안전한 경로 해석, raw/fft CSV 로드(인코딩 폴백)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

_STAMP_RE = re.compile(r"^(\d{8}_\d{6})_(raw|fft_ch(\d))\.csv$")

@dataclass
class Session:
    stamp: str
    raw_path: Path
    fft_paths: dict[int, Path] = field(default_factory=dict)

def safe_resolve(data_root: str | Path, name: str) -> Path:
    """name을 data_root 하위로만 해석. 경로 탈출 시 ValueError."""
    root = Path(data_root).resolve()
    target = (root / name).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"경로가 DATA_ROOT 밖을 가리킵니다: {name}")
    return target

def discover_sessions(data_root: str | Path, date_from: str = "", date_to: str = "") -> list[Session]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT 디렉토리가 없습니다: {root}")
    sessions: dict[str, Session] = {}
    for p in sorted(root.iterdir()):
        m = _STAMP_RE.match(p.name)
        if not m:
            continue
        stamp, kind = m.group(1), m.group(2)
        s = sessions.get(stamp)
        if s is None:
            s = Session(stamp=stamp, raw_path=Path())
            sessions[stamp] = s
        if kind == "raw":
            s.raw_path = p
        else:
            s.fft_paths[int(m.group(3))] = p
    out = [s for s in sessions.values() if s.raw_path != Path()]
    out.sort(key=lambda s: s.stamp)
    if date_from:
        out = [s for s in out if s.stamp >= date_from]
    if date_to:
        out = [s for s in out if s.stamp <= date_to]
    return out

def read_csv_fallback(path: str | Path) -> pd.DataFrame:
    """utf-8 → cp949 → euc-kr 순으로 시도."""
    last = None
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError) as e:
            last = e
    raise ValueError(f"인코딩을 해석할 수 없습니다: {path} ({last})")

def load_raw(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    df = read_csv_fallback(path)
    if "time_s" not in df.columns:
        raise ValueError(f"raw 파일에 time_s 컬럼이 없습니다: {path}")
    t = df["time_s"].to_numpy(dtype=float)
    chans = {c: df[c].to_numpy(dtype=float) for c in df.columns if c != "time_s"}
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
    fs = (1.0 / dt) if dt > 0 else 0.0
    return t, chans, fs

def load_fft(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = read_csv_fallback(path)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError(f"fft 파일 형식 오류: {path}")
    return df[cols[0]].to_numpy(dtype=float), df[cols[1]].to_numpy(dtype=float)

def list_data_folders(base: str | Path) -> list[str]:
    """base 하위에서 세션(<stamp>_raw.csv)이 1개 이상 있는 폴더를 base-상대 경로로 반환(재귀)."""
    base = Path(base)
    if not base.is_dir():
        raise FileNotFoundError(f"base 디렉토리가 없습니다: {base}")
    found: set[str] = set()
    for p in base.rglob("*_raw.csv"):
        if _STAMP_RE.match(p.name):
            found.add(str(p.parent.relative_to(base)))
    return sorted(found)
