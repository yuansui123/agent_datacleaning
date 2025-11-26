# tools.py
import os
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, iirnotch, filtfilt


# Ensure plot directory exists
DEFAULT_PLOT_DIR = "inspection_plots"
os.makedirs(DEFAULT_PLOT_DIR, exist_ok=True)


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def compute_fft(
    data: np.ndarray,
    sampling_rate: float,
    max_freq: Optional[float] = None,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Fourier Transform",
    plot_window: list = [0, 1000]
) -> Dict[str, Any]:
    """
    Compute the magnitude FFT of a 1D signal.

    Parameters
    ----------
    data : np.ndarray
        1D signal array.
    sampling_rate : float
        Sampling rate in Hz.
    max_freq : float, optional
        If provided, crop frequency axis to [0, max_freq] in the plot.
    make_plot : bool
        Whether to save a plot of the FFT.
    plot_dir : str
        Directory to store plots.
    label : str
        Plot title.
    plot_window : list
        x-axis limits for the plot.

    Returns
    -------
    dict with keys:
        numeric_results: {
            "fourier": {"freqs": [...], "fft": [...]}
        }
        image_paths: [path_to_fft_plot]  (may be empty if make_plot=False)
    """
    freqs = np.fft.rfftfreq(len(data), 1.0 / sampling_rate)
    fft_vals = np.abs(np.fft.rfft(data))

    numeric_results = {
        "fourier": {
            "freqs": freqs.tolist(),
            "fft": fft_vals.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        plt.plot(freqs, fft_vals)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(label)
        if max_freq is not None:
            plt.xlim(0, max_freq)
        if plot_window is not None:
            plt.xlim(plot_window)
        path = os.path.join(plot_dir, "fourier.png")
        _ensure_dir(path)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        image_paths.append(path)

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def compute_psd(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: int = 2048,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Power Spectral Density",
) -> Dict[str, Any]:
    """
    Compute Welch PSD estimate of the signal.

    Returns
    -------
    {
      "numeric_results": {
         "psd": {"freqs": [...], "psd": [...]}
      },
      "image_paths": [path_to_plot]
    }
    """
    f, Pxx = welch(data, fs=sampling_rate, nperseg=nperseg)

    numeric_results = {
        "psd": {
            "freqs": f.tolist(),
            "psd": Pxx.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        plt.semilogy(f, Pxx)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(label)
        path = os.path.join(plot_dir, "psd.png")
        _ensure_dir(path)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        image_paths.append(path)

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def compute_spectrogram_tool(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Spectrogram (dB)",
) -> Dict[str, Any]:
    """
    Compute a spectrogram of the signal.

    Returns
    -------
    {
      "numeric_results": {
         "spectrogram": {"f": [...], "t": [...], "sxx": [[...], ...]}
      },
      "image_paths": [path_to_plot]
    }
    """
    f, t, Sxx = spectrogram(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)

    numeric_results = {
        "spectrogram": {
            "f": f.tolist(),
            "t": t.tolist(),
            "sxx": Sxx.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title(label)
        path = os.path.join(plot_dir, "spectrogram.png")
        _ensure_dir(path)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        image_paths.append(path)

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def compute_variance_tool(
    data: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute the variance of the signal.

    Returns
    -------
    {
      "numeric_results": {"variance": float},
      "image_paths": []
    }
    """
    var = float(np.var(data))
    return {
        "numeric_results": {"variance": var},
        "image_paths": [],
    }


def compute_snr_tool(
    data: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute a simple SNR estimate: mean(signal^2) / var(signal - mean(signal)).

    Returns
    -------
    {
      "numeric_results": {"snr": float},
      "image_paths": []
    }
    """
    signal_power = float(np.mean(data**2))
    noise_power = float(np.var(data - np.mean(data)))
    snr = signal_power / (noise_power + 1e-12)
    return {
        "numeric_results": {"snr": snr},
        "image_paths": [],
    }


def compute_channel_stats_tool(
    data: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute mean and standard deviation.

    Returns
    -------
    {
      "numeric_results": {
        "channel_mean": float,
        "channel_std": float
      },
      "image_paths": []
    }
    """
    return {
        "numeric_results": {
            "channel_mean": float(np.mean(data)),
            "channel_std": float(np.std(data)),
        },
        "image_paths": [],
    }


def metadata_snapshot_tool(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return metadata so that interpreters can reason about it numerically.

    Returns
    -------
    {
      "numeric_results": {"metadata_snapshot": metadata_dict},
      "image_paths": []
    }
    """
    return {
        "numeric_results": {"metadata_snapshot": metadata},
        "image_paths": [],
    }


def apply_notch_filter_tool(
    data: np.ndarray,
    sampling_rate: float,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0,
    make_plot: bool = False,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Notch-filtered Signal (time domain)",
) -> Dict[str, Any]:
    """
    Apply a notch filter at a given frequency (e.g., 50/60 Hz).

    Returns
    -------
    {
      "numeric_results": {"filtered_signal": [...], "notch_freq": float},
      "image_paths": [optional_plot]
    }
    """
    w0 = notch_freq / (sampling_rate / 2.0)
    b, a = iirnotch(w0, quality_factor)
    filtered = filtfilt(b, a, data)

    numeric_results = {
        "filtered_signal": filtered.tolist(),
        "notch_freq": notch_freq,
    }

    image_paths: List[str] = []
    if make_plot:
        t = np.arange(len(data)) / sampling_rate
        plt.figure()
        plt.plot(t, data, alpha=0.5, label="raw")
        plt.plot(t, filtered, alpha=0.8, label="filtered")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(label)
        plt.legend()
        path = os.path.join(plot_dir, "notch_filtered_signal.png")
        _ensure_dir(path)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        image_paths.append(path)

    return {"numeric_results": numeric_results, "image_paths": image_paths}
