# tests/test_plotting.py
import base64
import numpy as np
from equipment_analysis import plotting

def test_downsample():
    x = np.arange(100000.0); y = np.sin(x)
    xs, ys = plotting.downsample(x, y, 5000)
    assert len(xs) <= 5000 and len(xs) == len(ys)

def _is_data_uri(s):
    assert s.startswith("![") and "data:image/png;base64," in s
    b64 = s.split("base64,", 1)[1].rstrip(")")
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"   # PNG 시그니처

def test_plot_waveform():
    t = np.linspace(0, 0.1, 5120)
    ch = {"ch0": np.sin(2*np.pi*58.2*t), "ch1": np.cos(2*np.pi*58.2*t)}
    _is_data_uri(plotting.plot_waveform(t, ch))

def test_plot_spectrum():
    freq = np.arange(0, 512, 0.2); mag = np.random.default_rng(0).random(len(freq))
    _is_data_uri(plotting.plot_spectrum(freq, mag, fmax=300))

def test_plot_trend():
    stamps = [f"s{i}" for i in range(25)]
    vals = np.array([1.0]*20 + [2.0]*5)
    flagged = np.array([False]*20 + [True]*5)
    _is_data_uri(plotting.plot_trend(stamps, vals, flagged, title="ch0 rms"))
