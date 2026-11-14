"""실데이터 e2e 데모(테스트 아님): post-07-07 정상 세션에 베인 손상 링다운 주입 →
session_features → robust_z로 정상 vs 손상 특징 대비 출력. 프로덕션 캐시/원본 불변(임시 out)."""
from __future__ import annotations
import tempfile
from pathlib import Path
from equipment_analysis import registry, dataio, features
from impeller_pinn.beam_modes import cantilever_frequencies
from impeller_pinn.synth import impact_ringdown
from impeller_pinn.inject import make_injected_session

FOLDER = "GY_DATA/pump data"

def main():
    root = Path(registry.folder_path(FOLDER))
    sess = [s for s in dataio.discover_sessions(str(root)) if s.stamp >= "20260707_112346"]
    src = sess[0]
    t, chans, fs = dataio.load_raw(src.raw_path)
    n = len(t)
    freqs = cantilever_frequencies(0.030, 0.0012, 193e9, 8000)
    print(f"베인 모드(Hz): {[round(f) for f in freqs]} | fs={fs:.0f} N={n}")
    ring = impact_ringdown(fs, n/fs, freqs, [0.02]*len(freqs), [5,2.5,1.2,0.6], t0=0.0)
    with tempfile.TemporaryDirectory() as td:
        inj = make_injected_session(root, src.stamp, td, "ch1", ring, "20260711_000001")
        healthy = features.session_features(src)
        damaged = features.session_features(dataio.discover_sessions(td)[0])
    for k in ["ch1_rms", "ch1_peak", "ch1_crest", "ch1_kurtosis"]:
        print(f"  {k}: 정상={healthy[k]:.4f} → 손상={damaged[k]:.4f}")

if __name__ == "__main__":
    main()
