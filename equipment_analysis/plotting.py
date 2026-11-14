# pump_analysis/plotting.py
"""matplotlib(Agg) 그래프를 markdown data URI 문자열로 반환."""
from __future__ import annotations
import base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"![plot](data:image/png;base64,{b64})"

def downsample(x: np.ndarray, y: np.ndarray, max_points: int):
    x = np.asarray(x); y = np.asarray(y)
    if max_points <= 0 or len(x) <= max_points:
        return x, y
    step = int(np.ceil(len(x) / max_points))
    return x[::step], y[::step]

def plot_waveform(t, channels: dict, max_points: int = 5000) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, sig in channels.items():
        xs, ys = downsample(t, sig, max_points)
        ax.plot(xs, ys, linewidth=0.7, label=name)
    ax.set_xlabel("time (s)"); ax.set_ylabel("amplitude")
    ax.set_title("Waveform"); ax.legend(loc="upper right", fontsize=8)
    return fig_to_data_uri(fig)

def plot_spectrum(freq, mag, fmax: float = 0.0, max_points: int = 5000) -> str:
    freq = np.asarray(freq); mag = np.asarray(mag)
    if fmax and fmax > 0:
        sel = freq <= fmax; freq, mag = freq[sel], mag[sel]
    xs, ys = downsample(freq, mag, max_points)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, linewidth=0.7)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("magnitude")
    ax.set_title("FFT Spectrum")
    return fig_to_data_uri(fig)

def plot_trend(stamps, values, flagged, title: str = "") -> str:
    values = np.asarray(values, dtype=float); flagged = np.asarray(flagged, dtype=bool)
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, values, "-o", markersize=3, linewidth=0.8, label="value")
    if flagged.any():
        ax.plot(x[flagged], values[flagged], "rx", markersize=9, label="anomaly")
    step = max(1, len(stamps) // 10)
    ax.set_xticks(x[::step]); ax.set_xticklabels([stamps[i] for i in x[::step]],
                                                 rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(title or "value"); ax.set_title(title or "Trend")
    ax.legend(loc="upper left", fontsize=8)
    return fig_to_data_uri(fig)
