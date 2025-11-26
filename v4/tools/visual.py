# tools/visual.py
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ._utils import DEFAULT_PLOT_DIR, finalize_figure


def plot_raw_signal_tool(
    data: np.ndarray,
    sampling_rate: float,
    t_max: Optional[float] = None,
    ylim: Optional[Sequence[float]] = None,
    label: str = "Raw signal",
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Plot the raw time-domain signal (no filtering or modification).

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    t_max : float, optional
        Maximum time (seconds) to display. If None, plot entire signal.
    ylim : [float, float], optional
        y-axis limits for visualization.
    label : str
        Plot title.
    plot_dir : str
        Directory for plots.

    Returns
    -------
    {
      "numeric_results": {},
      "image_paths": [path_to_plot]
    }
    """
    data = np.asarray(data).ravel()
    t = np.arange(len(data)) / sampling_rate
    if t_max is not None:
        mask = t <= t_max
        t = t[mask]
        data = data[mask]

    plt.figure()
    plt.plot(t, data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(label)
    if ylim is not None and len(ylim) == 2:
        plt.ylim(ylim[0], ylim[1])
    path = f"{plot_dir}/raw_signal.png"
    img_path = finalize_figure(path)

    return {"numeric_results": {}, "image_paths": [img_path]}


def plot_zoomed_window_tool(
    data: np.ndarray,
    sampling_rate: float,
    t_start: float,
    t_end: float,
    ylim: Optional[Sequence[float]] = None,
    label: str = "Zoomed window",
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Plot a zoomed time window of the raw signal.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    t_start : float
        Start time in seconds.
    t_end : float
        End time in seconds.
    ylim : [float, float], optional
        y-axis limits for visualization.
    label : str
        Plot title.
    plot_dir : str
        Directory for plots.
    """
    data = np.asarray(data).ravel()
    t = np.arange(len(data)) / sampling_rate
    mask = (t >= t_start) & (t <= t_end)
    t_win = t[mask]
    d_win = data[mask]

    plt.figure()
    plt.plot(t_win, d_win)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(label + f" [{t_start:.3f}, {t_end:.3f}] s")
    if ylim is not None and len(ylim) == 2:
        plt.ylim(ylim[0], ylim[1])
    path = f"{plot_dir}/raw_zoom_{t_start:.3f}_{t_end:.3f}.png"
    img_path = finalize_figure(path)

    return {"numeric_results": {}, "image_paths": [img_path]}


def raster_plot_tool(
    data: np.ndarray,
    sampling_rate: float,
    label: str = "Raster-like plot",
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Visualize multi-trial or multi-channel activity as a raster-like image.

    Parameters
    ----------
    data : np.ndarray
        2D array with shape (n_traces, n_samples). Each row is a trial or channel.
        This tool does not threshold or bin; it simply visualizes intensity.
    sampling_rate : float
        Sampling rate in Hz (used only for x-axis).
    label : str
        Plot title.
    plot_dir : str
        Directory for plots.

    Returns
    -------
    {
      "numeric_results": {},
      "image_paths": [path_to_plot]
    }
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[None, :]

    n_samples = data.shape[1]
    t = np.arange(n_samples) / sampling_rate

    plt.figure()
    plt.imshow(
        data,
        aspect="auto",
        extent=[t[0], t[-1], 0, data.shape[0]],
        origin="lower",
        interpolation="nearest",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Trace index")
    plt.title(label)
    path = f"{plot_dir}/raster_like.png"
    img_path = finalize_figure(path)

    return {"numeric_results": {}, "image_paths": [img_path]}
