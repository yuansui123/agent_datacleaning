"""
Tools return matplotlib Figure objects.

Single-channel version.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Literal
import os
import uuid

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import gaussian_filter1d


# ============================================================
# HELPER FUNCTIONS FOR PLOTTING
# ============================================================

def _ensure_1d(data: np.ndarray) -> np.ndarray:
    """Ensure data is 1D: (n_timepoints,)."""
    data = np.asarray(data)
    if data.ndim != 1:
        data = data.ravel()
    return data


def _slice_time_window(
    data: np.ndarray,
    sampling_rate: float,
    time_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract time window from 1D data.
    
    Returns:
        data_win: Windowed data
        times: Time axis in seconds
    """
    data = _ensure_1d(data)
    n_samples = len(data)
    
    if time_range is None:
        start_idx = 0
        end_idx = n_samples
    else:
        start_sec, end_sec = time_range
        start_idx = int(start_sec * sampling_rate)
        end_idx = int(end_sec * sampling_rate)
        start_idx = max(0, start_idx)
        end_idx = min(n_samples, end_idx)
    
    data_win = data[start_idx:end_idx]
    times = np.arange(len(data_win)) / sampling_rate + (start_idx / sampling_rate)
    
    return data_win, times


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_time_series(
    data: np.ndarray,
    sampling_rate: float,
    time_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create time series plot for single channel.
    
    Args:
        required:
            data: (n_timepoints,) single channel
            sampling_rate: Hz
        optional:
            time_range: (start_sec, end_sec)
            figsize: Figure size
    
    Returns:
        fig: Matplotlib Figure object
        metadata: Dict with plot metadata (stats, params, etc.)
    """
    data_win, times = _slice_time_window(data, sampling_rate, time_range)

    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(times, data_win, linewidth=0.8, color='C0')

    # Add more x-axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))  # ~10 ticks
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time Series")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Metadata
    metadata = {
        "plot_type": "time_series",
        "sampling_rate": sampling_rate,
        "time_range": (float(times[0]), float(times[-1])),
        "n_samples": len(data_win),
    }
    
    return fig, metadata


def plot_psd(
    data: np.ndarray,
    sampling_rate: float,
    time_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = [5, 200],
    freq_of_interest: Optional[List[float]] = [60.0],
    n_fft: Optional[int] = None,
    use_db: bool = True,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create power spectral density plot for single channel data.
    
    Args:
        required:   
            data: (n_timepoints,) single channel signal
            sampling_rate: Hz
        optional:
            time_range: (start_sec, end_sec) to analyze subset of data
            freq_range: (min_hz, max_hz) to display
            freq_of_interest: List of frequencies (Hz) to mark with vertical dashed lines (default: [60])
            n_fft: FFT window size (default: min(2048, n_timepoints))
            use_db: If True, plot in dB scale; if False, use semilogy with V²/Hz (default: True)
            figsize: Figure size
    
    Returns:
        fig: Matplotlib Figure object
        metadata: Dict with plot metadata
    """
    data = _ensure_1d(data)
    
    # Apply time window if specified
    if time_range is not None:
        data, _ = _slice_time_window(data, sampling_rate, time_range)
    
    n_samples = len(data)
    
    # Set FFT window size
    if n_fft is None:
        n_fft = min(2048, n_samples)
    
    # Compute PSD using Welch's method
    freqs, Pxx = welch(data, fs=sampling_rate, nperseg=n_fft)
    
    # Apply frequency range filtering if specified
    if freq_range is not None:
        f_min, f_max = freq_range
        mask = (freqs >= f_min) & (freqs <= f_max)
        freqs = freqs[mask]
        Pxx = Pxx[mask]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if use_db:
        # Convert to dB scale and use regular plot
        Pxx_dB = 10 * np.log10(Pxx + 1e-20)
        ax.plot(freqs, Pxx_dB, linewidth=1.5)
        ylabel = "Power Spectral Density (dB)"
        scale = "dB"
    else:
        # Use semilogy with linear power units
        ax.semilogy(freqs, Pxx, linewidth=1.5)
        ylabel = "Power Spectral Density (V²/Hz)"
        scale = "linear_log"
    
    # Add vertical lines for frequencies of interest
    for freq in freq_of_interest:
        # Only plot if frequency is within the displayed range
        if freq_range is None or (freq_range[0] <= freq <= freq_range[1]):
            ax.axvline(x=freq, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'{freq} Hz')
    
    # Add legend if there are frequency markers
    if freq_of_interest:
        ax.legend(fontsize=9)
    
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title("Power Spectral Density")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Metadata
    metadata = {
        "plot_type": "psd_single_channel",
        "sampling_rate": sampling_rate,
        "n_samples": n_samples,
        "time_range": time_range,
        "freq_range": (float(freqs[0]), float(freqs[-1])) if len(freqs) > 0 else None,
        "freq_of_interest": freq_of_interest,
        "n_fft": n_fft,
        "time_range": time_range,
        "scale": scale,
    }
    
    return fig, metadata


def plot_spectrogram(
    data: np.ndarray,
    sampling_rate: float,
    time_range: Optional[Tuple[float, float]] = None,
    freq_range: Optional[Tuple[float, float]] = [5, 200],
    n_fft: Optional[int] = None,
    overlap_ratio: float = 0.75,
    window: str = 'hann',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'hot',
    figsize: Tuple[float, float] = (12, 6),
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create spectrogram plot for single channel.
    
    Args:
        required:
            data: (n_timepoints,) single channel
            sampling_rate: Hz
        optional:
            time_range: (start_sec, end_sec)
            freq_range: (min_hz, max_hz)
            n_fft: FFT window size (default: 256 for good time-freq balance)
            overlap_ratio: Overlap between windows (0-1, default: 0.75 for smooth visualization)
            window: Window function ('hann', 'hamming', 'blackman')
            vmin, vmax: Color scale limits in dB (auto if None)
            cmap: Colormap ('hot', 'viridis', 'jet', 'magma')
            figsize: Figure size
    
    Returns:
        fig: Matplotlib Figure object
        metadata: Dict with plot metadata
    """
    data_win, _ = _slice_time_window(data, sampling_rate, time_range)

    # Better default n_fft
    if n_fft is None:
        n_fft = min(512, len(data_win) // 4)
    
    # Calculate overlap in samples
    noverlap = int(n_fft * overlap_ratio)

    # Compute spectrogram with overlap
    f, t, Sxx = spectrogram(
        data_win, 
        fs=sampling_rate, 
        nperseg=n_fft,
        noverlap=noverlap,
        window=window
    )
    
    # Convert to dB before filtering
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    # Frequency range filtering
    if freq_range is not None:
        f0, f1 = freq_range
        mask = (f >= f0) & (f <= f1)
        f = f[mask]
        Sxx_db = Sxx_db[mask, :]

    # Auto-scale color limits if not provided
    if vmin is None:
        vmin = np.percentile(Sxx_db, 5)
    if vmax is None:
        vmax = np.percentile(Sxx_db, 95)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolormesh(
        t, 
        f, 
        Sxx_db, 
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    
    cbar = fig.colorbar(im, ax=ax, label="Power (dB)")

    # Add more x-axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))  # ~10 ticks
    
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title("Spectrogram", fontsize=12, pad=10)
    ax.set_ylim([f[0], f[-1]])
    
    plt.tight_layout()

    # Metadata
    metadata = {
        "plot_type": "spectrogram",
        "freq_range": (float(f[0]), float(f[-1])) if len(f) > 0 else None,
        "time_range": (float(t[0]), float(t[-1])) if len(t) > 0 else None,
        "n_fft": n_fft,
        "noverlap": noverlap,
        "window": window,
        "color_scale": {"vmin": float(vmin), "vmax": float(vmax)},
    }

    return fig, metadata


def plot_power_density_matrix_hilbert(
    data: np.ndarray,
    sampling_rate: float,
    freq_bands: Optional[Dict[str, List[float]]] = None,
    clip_seconds: float = 0.1,
    filter_order: int = 4,
    envelope_smoothing_sigma: float = 10.0,
    normalize_mode: Literal["global", "per_band", "none"] = "per_band",
    figsize: Tuple[float, float] = (12, 4),
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Create power density matrix using Hilbert transform.
    
    Args:
        required:
            data: (n_timepoints,) single channel
            sampling_rate: Hz
        optional:
            freq_bands: Dict mapping band names to [low, high] Hz
            clip_seconds: Edge clipping duration
            filter_order: Butterworth filter order
            envelope_smoothing_sigma: Gaussian smoothing sigma
            normalize_mode: Normalization strategy
            figsize: Figure size
    
    Returns:
        fig: Matplotlib Figure object
        metadata: Dict with plot metadata
    """
    data = _ensure_1d(data)
    n_samples = len(data)
    time = np.arange(n_samples) / float(sampling_rate)

    # Default frequency bands
    if freq_bands is None:
        freq_bands = {
            "5-20 Hz": [5.0, 20.0],
            "20-35 Hz": [20.0, 35.0],
            "35-50 Hz": [35.0, 50.0],
            "50-65 Hz": [50.0, 65.0],
            "65-80 Hz": [65.0, 80.0],
            "80-95 Hz": [80.0, 95.0],
            "95-110 Hz": [95.0, 110.0],
            "110-125 Hz": [110.0, 125.0],
            "125-140 Hz": [125.0, 140.0],
            "140-155 Hz": [140.0, 155.0],
            "155-170 Hz": [155.0, 170.0],
            "170-185 Hz": [170.0, 185.0],
            "185-200 Hz": [185.0, 200.0],
        }

    band_names = list(freq_bands.keys())
    n_bands = len(band_names)
    nyq = sampling_rate / 2.0
    clip_samples = int(sampling_rate * clip_seconds)

    # Compute power density matrix
    pdm = np.zeros((n_bands, n_samples), dtype=float)

    for i, (band, (lo, hi)) in enumerate(freq_bands.items()):
        lo_norm = lo / nyq
        hi_norm = hi / nyq
        if lo_norm >= hi_norm:
            continue

        # Bandpass filter
        sos = butter(filter_order, [lo_norm, hi_norm], btype="band", output="sos")
        filtered = sosfiltfilt(sos, data)
        
        # Hilbert envelope
        envelope = np.abs(hilbert(filtered))

        # Smooth envelope
        if envelope_smoothing_sigma > 0:
            envelope = gaussian_filter1d(envelope, sigma=envelope_smoothing_sigma)

        pdm[i] = envelope

    # Clip edges
    if clip_samples > 0 and clip_samples * 2 < n_samples:
        pdm = pdm[:, clip_samples:-clip_samples]
        time = time[clip_samples:-clip_samples]
    else:
        raise ValueError("clip_seconds too large; no data left after clipping.")

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

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        pdm_norm,
        aspect="auto",
        origin="lower",
        extent=[time[0], time[-1], 0, n_bands],
        cmap="hot",
    )

    # Add more x-axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=30))  
    
    ax.set_yticks(np.arange(n_bands) + 0.5)
    ax.set_yticklabels(band_names, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency Band")
    ax.set_title(f"Power Density Matrix (Hilbert) - {normalize_mode} normalization")
    
    fig.colorbar(im, ax=ax, label="Normalized Power")
    plt.tight_layout()

    # Metadata
    metadata = {
        "plot_type": "power_density_matrix_hilbert",
        "freq_bands": freq_bands,
        "normalize_mode": normalize_mode,
        "clip_seconds": clip_seconds,
        "filter_order": filter_order,
    }

    return fig, metadata


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY = {
    "plot_time_series": plot_time_series,
    "plot_psd": plot_psd,
    "plot_spectrogram": plot_spectrogram,
    "plot_power_density_matrix_hilbert": plot_power_density_matrix_hilbert,
}