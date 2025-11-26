# tools/statistics.py
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from ._utils import DEFAULT_PLOT_DIR, finalize_figure


def basic_stats_tool(
    data: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute basic descriptive statistics of the signal.

    Returns
    -------
    {
      "numeric_results": {
        "mean": float,
        "std": float,
        "var": float,
        "min": float,
        "max": float,
        "skew": float,
        "kurtosis": float
      },
      "image_paths": []
    }
    """
    data = np.asarray(data).ravel()
    stats = {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "var": float(np.var(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "skew": float(skew(data)),
        "kurtosis": float(kurtosis(data)),
    }
    return {"numeric_results": {"basic_stats": stats}, "image_paths": []}


def compute_variance_tool(
    data: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute the variance of the signal.
    """
    data = np.asarray(data).ravel()
    var = float(np.var(data))
    return {"numeric_results": {"variance": var}, "image_paths": []}


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
    data = np.asarray(data).ravel()
    signal_power = float(np.mean(data**2))
    noise_power = float(np.var(data - np.mean(data)))
    snr = signal_power / (noise_power + 1e-12)
    return {"numeric_results": {"snr": snr}, "image_paths": []}


def amplitude_histogram_tool(
    data: np.ndarray,
    bins: int = 100,
    range: Optional[Sequence[float]] = None,
    density: bool = True,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Amplitude distribution",
) -> Dict[str, Any]:
    """
    Compute and optionally plot the amplitude distribution of the signal.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    bins : int
        Number of histogram bins.
    range : [float, float], optional
        Amplitude range for the histogram. If None, use data min/max.
    density : bool
        If True, histogram is normalized to form a probability density.
    make_plot : bool
        If True, save a histogram plot.
    plot_dir : str
        Directory for plots.
    label : str
        Plot title.
    """
    data = np.asarray(data).ravel()
    counts, bin_edges = np.histogram(data, bins=bins, range=range, density=density)

    numeric_results = {
        "amplitude_histogram": {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
            "density": density,
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        plt.hist(data, bins=bins, range=range, density=density)
        plt.xlabel("Amplitude")
        plt.ylabel("Probability density" if density else "Count")
        plt.title(label)
        path = f"{plot_dir}/amplitude_histogram.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def autocorrelation_tool(
    data: np.ndarray,
    max_lag_seconds: float,
    sampling_rate: float,
    normalize: bool = True,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Autocorrelation",
) -> Dict[str, Any]:
    """
    Compute autocorrelation of the signal up to a maximum lag.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    max_lag_seconds : float
        Maximum lag in seconds to compute.
    sampling_rate : float
        Sampling rate in Hz.
    normalize : bool
        If True, divide by zero-lag value to get correlation coefficient.
    make_plot : bool
        If True, save a plot of autocorrelation vs lag.
    plot_dir : str
        Directory for plots.
    label : str
        Plot title.
    """
    data = np.asarray(data).ravel()
    max_lag_samples = int(max_lag_seconds * sampling_rate)
    corr_full = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
    mid = len(corr_full) // 2
    corr = corr_full[mid : mid + max_lag_samples + 1]

    if normalize and corr[0] != 0:
        corr = corr / corr[0]

    lags = np.arange(0, max_lag_samples + 1) / sampling_rate

    numeric_results = {
        "autocorrelation": {
            "lags": lags.tolist(),
            "values": corr.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        plt.plot(lags, corr)
        plt.xlabel("Lag (s)")
        plt.ylabel("Autocorrelation")
        plt.title(label)
        path = f"{plot_dir}/autocorrelation.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def numeric_summary_tool(
    data: np.ndarray,
    sampling_rate: float,
) -> Dict[str, Any]:
    """
    Composite tool: run basic_stats_tool, compute_snr_tool, and autocorrelation (short lag).

    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    Combined numeric results from multiple simple statistics tools.
    """
    numeric_results: Dict[str, Any] = {}
    image_paths: List[str] = []

    out_stats = basic_stats_tool(data)
    numeric_results.update(out_stats["numeric_results"])

    out_snr = compute_snr_tool(data)
    numeric_results.update(out_snr["numeric_results"])

    out_auto = autocorrelation_tool(
        data=data,
        max_lag_seconds=0.05,
        sampling_rate=sampling_rate,
        make_plot=False,
    )
    numeric_results.update(out_auto["numeric_results"])

    return {"numeric_results": numeric_results, "image_paths": image_paths}
