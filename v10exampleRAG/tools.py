# tools.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Literal
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d


# ---------- Internal helpers ----------

def _make_output_dir(output_dir: Optional[str]) -> str:
    if output_dir is None:
        output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_fig(fig: plt.Figure, output_dir: Optional[str], prefix: str) -> Tuple[str, str]:
    output_dir = _make_output_dir(output_dir)
    plot_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
    file_path = os.path.join(output_dir, f"{plot_id}.png")
    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return file_path, plot_id


def _ensure_2d(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)  # (1, timepoints)
    if data.ndim != 2:
        raise ValueError("data must be 2D: (n_channels, n_timepoints)")
    return data


def _slice_time_window(
    data: np.ndarray,
    sampling_rate: float,
    channels: Optional[List[int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    downsample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Slice data by channels, time_range, and optional downsampling.

    Returns:
        data_win: shape (len(channels), n_samples_window)
        times: shape (n_samples_window,)
        channels_used: list[int]
    """
    data = _ensure_2d(data)
    n_channels, n_time = data.shape

    # channels
    if channels is None or len(channels) == 0:
        channels = list(range(n_channels))
    channels = [ch for ch in channels if 0 <= ch < n_channels]

    # time window
    total_time = n_time / sampling_rate
    if time_range is None:
        t0, t1 = 0.0, total_time
    else:
        t0, t1 = max(0.0, time_range[0]), min(total_time, time_range[1])

    idx0, idx1 = int(t0 * sampling_rate), int(t1 * sampling_rate)
    idx0, idx1 = max(0, idx0), min(n_time, idx1)
    if idx1 <= idx0:
        raise ValueError("Invalid time_range; resulting window is empty.")

    data_win = data[channels, idx0:idx1]
    times = np.arange(idx0, idx1) / sampling_rate

    # downsample
    if downsample is not None and downsample > 1:
        data_win = data_win[:, ::downsample]
        times = times[::downsample]

    return data_win, times, channels


# ---------- Individual plotting functions ----------

def plot_time_series(
    data: np.ndarray,
    sampling_rate: float,
    *,
    channels: Optional[List[int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    downsample: Optional[int] = None,
    aggregate: Literal["none", "mean", "median"] = "none",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot raw neural time series for one or more channels.
    """
    data_win, times, channels_used = _slice_time_window(
        data, sampling_rate, channels, time_range, downsample
    )
    fig, ax = plt.subplots(figsize=(12, 4))

    if aggregate in ("mean", "median") and len(channels_used) > 1:
        if aggregate == "mean":
            y = data_win.mean(axis=0)
            agg_name = "mean"
        else:
            y = np.median(data_win, axis=0)
            agg_name = "median"
        ax.plot(times, y)
        title = f"Time series ({agg_name} of channels {channels_used})"
        desc = f"Time series {agg_name} across {len(channels_used)} channels."
        stats = f"{agg_name} over time: mean={y.mean():.3f}, std={y.std():.3f}"
    else:
        offset = 0.0
        step = np.nanmax(np.abs(data_win)) * 1.5 if np.any(data_win) else 1.0
        for i, ch in enumerate(channels_used):
            y = data_win[i] + offset
            ax.plot(times, y, label=f"ch{ch}")
            offset += step
        ax.legend(loc="upper right", fontsize=6)
        title = f"Time series for channels {channels_used}"
        desc = f"Stacked time series for {len(channels_used)} channels."
        stats = f"Global mean={data_win.mean():.3f}, global std={data_win.std():.3f}"

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    params = dict(
        channels=channels_used,
        time_range=(float(times[0]), float(times[-1])),
        downsample=downsample,
        aggregate=aggregate,
    )

    file_path, plot_id = _save_fig(fig, output_dir, "time_series")
    return {
        "plot_id": plot_id,
        "plot_type": "plot_time_series",
        "file_path": file_path,
        "description": desc,
        "stats_summary": stats,
        "params": params,
        "errors": None,
    }


def plot_psd(
    data: np.ndarray,
    sampling_rate: float,
    *,
    channels: Optional[List[int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    n_fft: Optional[int] = None,
    aggregate: Literal["none", "mean", "median"] = "none",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot power spectral density for selected channels.
    """
    data_win, _, channels_used = _slice_time_window(
        data, sampling_rate, channels, time_range, downsample=None
    )

    if n_fft is None:
        n_fft = min(2048, data_win.shape[1])

    psds = []
    freqs = None
    for i, _ in enumerate(channels_used):
        f, Pxx = welch(data_win[i], fs=sampling_rate, nperseg=n_fft)
        if freqs is None:
            freqs = f
        psds.append(Pxx)
    psds = np.array(psds)

    if freq_range is not None:
        f0, f1 = freq_range
        mask = (freqs >= f0) & (freqs <= f1)
        freqs = freqs[mask]
        psds = psds[:, mask]

    fig, ax = plt.subplots(figsize=(6, 4))

    if aggregate in ("mean", "median") and len(channels_used) > 1:
        if aggregate == "mean":
            y = psds.mean(axis=0)
            agg_name = "mean"
        else:
            y = np.median(psds, axis=0)
            agg_name = "median"
        ax.semilogy(freqs, y)
        ax.set_title(f"PSD ({agg_name} across channels {channels_used})")
        desc = f"PSD {agg_name} across {len(channels_used)} channels."
        stats = f"Mean PSD in band={y.mean():.3e}"
    else:
        for i, ch in enumerate(channels_used):
            ax.semilogy(freqs, psds[i], label=f"ch{ch}")
        ax.legend(fontsize=6)
        ax.set_title(f"PSD for channels {channels_used}")
        desc = f"Per-channel PSD for channels {channels_used}."
        stats = f"Global mean PSD={psds.mean():.3e}"

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")

    params = dict(
        channels=channels_used,
        freq_range=freq_range,
        n_fft=n_fft,
        aggregate=aggregate,
    )

    file_path, plot_id = _save_fig(fig, output_dir, "psd")
    return {
        "plot_id": plot_id,
        "plot_type": "plot_psd",
        "file_path": file_path,
        "description": desc,
        "stats_summary": stats,
        "params": params,
        "errors": None,
    }


def plot_spectrogram(
    data: np.ndarray,
    sampling_rate: float,
    *,
    channels: Optional[List[int]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    n_fft: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Plot spectrogram (time-frequency) for a single channel (first in list).
    """
    data_win, _, channels_used = _slice_time_window(
        data, sampling_rate, channels, time_range, downsample=None
    )

    ch0 = 0  # first in the selected list
    x = data_win[ch0]

    if n_fft is None:
        n_fft = min(256, len(x))

    f, t, Sxx = spectrogram(x, fs=sampling_rate, nperseg=n_fft)
    if freq_range is not None:
        f0, f1 = freq_range
        mask = (f >= f0) & (f <= f1)
        f = f[mask]
        Sxx = Sxx[mask, :]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram (channel {channels_used[ch0]})")

    desc = (
        f"Spectrogram of channel {channels_used[ch0]} from {f[0]:.1f}–{f[-1]:.1f} Hz."
        if len(f) > 1
        else f"Spectrogram of channel {channels_used[ch0]}."
    )
    stats = f"Mean power (dB)={10 * np.log10(Sxx + 1e-12).mean():.2f}"

    params = dict(
        channels=[channels_used[ch0]],
        freq_range=freq_range,
        n_fft=n_fft,
    )

    file_path, plot_id = _save_fig(fig, output_dir, "spectrogram")
    return {
        "plot_id": plot_id,
        "plot_type": "plot_spectrogram",
        "file_path": file_path,
        "description": desc,
        "stats_summary": stats,
        "params": params,
        "errors": None,
    }


def compute_power_density_matrix_hilbert(
        
    data: np.ndarray,
    sampling_rate: float,
    output_dir: Optional[str] = None,
    freq_bands: Optional[Dict[str, List[float]]] = None,
    clip_seconds: float = 0.5,
    filter_order: int = 4,
    envelope_smoothing_sigma: float = 10.0,
    normalize_mode: str = "global",
    label: str = "Power Density Matrix (Hilbert)",
) -> Dict[str, Any]:
    """
    Compute time–frequency power density matrix using Hilbert envelopes.
    Save ONE standardized plot via _save_fig, and return image info only.

    Parameters
    ----------
    data : np.ndarray
        1D or 2D signal (2D -> first channel auto-selected)
    sampling_rate : float
    output_dir : Optional[str]
        Directory where the figure will be saved. If None, _make_output_dir
        will select a default location.
    freq_bands : dict, optional
    clip_seconds : float
    filter_order : int
    envelope_smoothing_sigma : float
    normalize_mode : {"global","per_band","none"}
    label : str
        Title for the plot.

    Returns
    -------
    dict with keys:
        - plot_id
        - plot_type
        - file_path
        - description
        - stats_summary
        - params
        - errors
    """
    params: Dict[str, Any] = {
        "freq_bands": freq_bands,
        "clip_seconds": clip_seconds,
        "filter_order": filter_order,
        "envelope_smoothing_sigma": envelope_smoothing_sigma,
        "normalize_mode": normalize_mode,
    }

    try:
        # Ensure 1D
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[0]
        data = data.ravel()

        n_samples = len(data)
        time = np.arange(n_samples) / float(sampling_rate)

        # Default frequency bands
        if freq_bands is None:
            nyq = sampling_rate / 2.0
            freq_bands = {
                "delta": [0.5, min(4.0, nyq)],
                "theta": [4.0, min(8.0, nyq)],
                "alpha": [8.0, min(13.0, nyq)],
                "beta": [13.0, min(30.0, nyq)],
                "low_gamma": [30.0, min(55.0, nyq)],
                "powerline_60hz": [57.0, min(63.0, nyq)],
                "mid_gamma": [65.0, min(80.0, nyq)],
                "high_gamma": [80.0, min(150.0, nyq)],
                "ripple": [150.0, min(250.0, nyq)],
                "fast_ripple": [250.0, min(500.0, nyq)],
            }
            params["freq_bands"] = freq_bands

        band_names = list(freq_bands.keys())
        n_bands = len(band_names)
        nyq = sampling_rate / 2.0
        clip_samples = int(sampling_rate * clip_seconds)

        # Time–frequency power matrix
        pdm = np.zeros((n_bands, n_samples), dtype=float)

        for i, (band, (lo, hi)) in enumerate(freq_bands.items()):
            lo_norm = lo / nyq
            hi_norm = hi / nyq
            if lo_norm >= hi_norm:
                continue

            sos = butter(filter_order, [lo_norm, hi_norm], btype="band", output="sos")
            filtered = sosfiltfilt(sos, data)
            envelope = np.abs(hilbert(filtered))

            if envelope_smoothing_sigma > 0:
                envelope = gaussian_filter1d(envelope, sigma=envelope_smoothing_sigma)

            pdm[i] = envelope

        # Clip edges
        if clip_samples > 0 and clip_samples * 2 < n_samples:
            pdm = pdm[:, clip_samples:-clip_samples]
            time = time[clip_samples:-clip_samples]

        # Normalize
        if normalize_mode == "global":
            mn, mx = pdm.min(), pdm.max()
            pdm_norm = (pdm - mn) / (mx - mn + 1e-12)
        elif normalize_mode == "per_band":
            pdm_norm = np.zeros_like(pdm)
            for i in range(n_bands):
                row = pdm[i]
                mn, mx = row.min(), row.max()
                pdm_norm[i] = (row - mn) / (mx - mn + 1e-12)
        else:
            pdm_norm = pdm

        # ---------------------
        # Plot (heatmap + lines)
        # ---------------------
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        im = axes[0].imshow(
            pdm_norm,
            aspect="auto",
            origin="lower",
            extent=[time[0], time[-1], 0, n_bands],
            cmap="hot",
        )
        axes[0].set_yticks(np.arange(n_bands) + 0.5)
        axes[0].set_yticklabels(band_names)
        axes[0].set_title(f"{label} — Heatmap ({normalize_mode})")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Frequency Band")
        plt.colorbar(im, ax=axes[0])

        for i, name in enumerate(band_names):
            axes[1].plot(time, pdm_norm[i], label=name)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].set_title("Temporal Evolution")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Norm Power")

        plt.tight_layout()

        # Use shared helper to save (no finalize_figure)
        file_path, plot_id = _save_fig(fig, output_dir, prefix="power_density_matrix_hilbert")

        return {
            "plot_id": plot_id,
            "plot_type": "power_density_matrix_hilbert",
            "file_path": file_path,
            "description": label,
            "stats_summary": "",
            "params": params,
            "errors": None,
        }

    except Exception as e:
        # In error case, we might not have a plot_id/file_path
        return {
            "plot_id": "power_density_matrix_hilbert_error",
            "plot_type": "power_density_matrix_hilbert",
            "file_path": None,
            "description": label,
            "stats_summary": "",
            "params": params,
            "errors": str(e),
        }


def plot_simple_line_noise(
    data: np.ndarray,
    sampling_rate: float,
    output_dir: Optional[str] = None,
    line_freq: float = 60.0,
    welch_nperseg: Optional[int] = None,
    label: str = "Simple Line Noise Detection",
) -> Dict[str, Any]:
    """
    Very simple 50/60-Hz line-noise visualization.
    Computes PSD via Welch, plots one figure, marks the chosen line frequency.

    Parameters
    ----------
    data : np.ndarray (1D or 2D)
        Neural signal. If 2D, first channel is auto-selected.
    sampling_rate : float
        Sampling rate in Hz.
    output_dir : Optional[str]
        Directory where the figure will be saved.
    line_freq : float
        Either 50.0 or 60.0 Hz.
    welch_nperseg : Optional[int]
        PSD window length; if None uses 2 * sampling_rate.
    label : str
        Title for the figure.

    Returns
    -------
    dict containing only image info.
    """

    params = {
        "line_freq": line_freq,
        "welch_nperseg": welch_nperseg,
        "label": label,
    }

    try:
        # Ensure 1D data
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[0]
        data = data.ravel()

        n_samples = len(data)

        if welch_nperseg is None:
            welch_nperseg = min(n_samples, int(2 * sampling_rate))
            params["welch_nperseg"] = welch_nperseg

        # --- Welch PSD ---
        freqs, psd = welch(
            data,
            fs=sampling_rate,
            nperseg=welch_nperseg,
            noverlap=welch_nperseg // 2,
        )
        psd_db = 10 * np.log10(psd + 1e-20)

        # --- Figure ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, psd_db, linewidth=1)
        ax.set_xlim(0, sampling_rate / 2)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.set_title(f"{label} — {line_freq} Hz Check")
        ax.grid(alpha=0.3)

        # Mark the line noise frequency
        if line_freq < sampling_rate / 2:
            ax.axvline(line_freq, color="red", linestyle="--", alpha=0.9)

        plt.tight_layout()

        # Save file
        file_path, plot_id = _save_fig(fig, output_dir, prefix="simple_line_noise")

        return {
            "plot_id": plot_id,
            "plot_type": "simple_line_noise",
            "file_path": file_path,
            "description": label,
            "stats_summary": "",
            "params": params,
            "errors": None,
        }

    except Exception as e:
        return {
            "plot_id": "simple_line_noise_error",
            "plot_type": "simple_line_noise",
            "file_path": None,
            "description": label,
            "stats_summary": "",
            "params": params,
            "errors": str(e),
        }

# ---------- Registry for dynamic dispatch ----------

TOOL_REGISTRY: Dict[str, Any] = {
    "plot_time_series": plot_time_series,
    "plot_psd": plot_psd,
    "plot_spectrogram": plot_spectrogram,
    "power_density_matrix_hilbert": compute_power_density_matrix_hilbert,
    "line_noise_harmonics": plot_simple_line_noise,
}
