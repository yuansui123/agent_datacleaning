# tools/spike.py
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ._utils import DEFAULT_PLOT_DIR, finalize_figure


def spike_rate_stats_tool(
    spike_times: Sequence[float],
    recording_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute simple spike rate statistics.

    Parameters
    ----------
    spike_times : sequence of float
        Spike times in seconds.
    recording_duration : float, optional
        Total recording duration in seconds. If None, inferred from max(spike_times).

    Returns
    -------
    {
      "numeric_results": {
         "spike_rate": float,
         "n_spikes": int,
         "recording_duration": float
      },
      "image_paths": []
    }
    """
    spike_times = np.asarray(spike_times, dtype=float)
    if spike_times.size == 0:
        stats = {"spike_rate": 0.0, "n_spikes": 0, "recording_duration": float(recording_duration or 0.0)}
        return {"numeric_results": {"spike_rate_stats": stats}, "image_paths": []}

    if recording_duration is None:
        recording_duration = float(spike_times.max())

    rate = spike_times.size / (recording_duration + 1e-12)
    stats = {
        "spike_rate": float(rate),
        "n_spikes": int(spike_times.size),
        "recording_duration": float(recording_duration),
    }
    return {"numeric_results": {"spike_rate_stats": stats}, "image_paths": []}


def isi_histogram_tool(
    spike_times: Sequence[float],
    bins: int = 50,
    max_isi: Optional[float] = None,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Inter-spike interval (ISI) histogram",
) -> Dict[str, Any]:
    """
    Compute and optionally plot the distribution of inter-spike intervals.

    Parameters
    ----------
    spike_times : sequence of float
        Spike times in seconds.
    bins : int
        Number of histogram bins.
    max_isi : float, optional
        If provided, discard ISIs above this value for numeric results and plotting.
    make_plot : bool
        If True, save histogram plot.
    plot_dir : str
        Directory for plots.
    label : str
        Plot title.
    """
    spike_times = np.sort(np.asarray(spike_times, dtype=float))
    if spike_times.size < 2:
        return {
            "numeric_results": {"isi_histogram": {"bin_edges": [], "counts": []}},
            "image_paths": [],
        }

    isis = np.diff(spike_times)
    if max_isi is not None:
        isis = isis[isis <= max_isi]

    counts, bin_edges = np.histogram(isis, bins=bins)

    numeric_results = {
        "isi_histogram": {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot and isis.size > 0:
        plt.figure()
        plt.hist(isis, bins=bins)
        plt.xlabel("ISI (s)")
        plt.ylabel("Count")
        plt.title(label)
        path = f"{plot_dir}/isi_histogram.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def spike_waveform_summary_tool(
    spike_waveforms: np.ndarray,
    sampling_rate: float,
    make_plot: bool = True,
    n_examples: int = 100,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Spike waveforms",
) -> Dict[str, Any]:
    """
    Summarize spike waveforms (already extracted elsewhere).

    Parameters
    ----------
    spike_waveforms : np.ndarray
        2D array of shape (n_spikes, n_samples_per_spike).
    sampling_rate : float
        Sampling rate in Hz (for x-axis in plots).
    make_plot : bool
        If True, plot a subset of waveforms and their mean ± std.
    n_examples : int
        Maximum number of individual waveforms to overlay.
    plot_dir : str
        Directory for plots.
    label : str
        Plot title.
    """
    wf = np.asarray(spike_waveforms)
    if wf.ndim != 2 or wf.size == 0:
        return {
            "numeric_results": {"spike_waveform_summary": {}},
            "image_paths": [],
        }

    n_spikes, n_samples = wf.shape
    mean_waveform = wf.mean(axis=0)
    std_waveform = wf.std(axis=0)
    t = np.arange(n_samples) / sampling_rate * 1000.0  # ms

    numeric_results = {
        "spike_waveform_summary": {
            "n_spikes": int(n_spikes),
            "n_samples": int(n_samples),
            "mean_waveform": mean_waveform.tolist(),
            "std_waveform": std_waveform.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        idx = np.linspace(0, n_spikes - 1, min(n_examples, n_spikes)).astype(int)
        for i in idx:
            plt.plot(t, wf[i], alpha=0.3)
        plt.plot(t, mean_waveform, color="k", linewidth=2, label="Mean")
        plt.fill_between(
            t,
            mean_waveform - std_waveform,
            mean_waveform + std_waveform,
            alpha=0.3,
            label="Mean ± std",
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.title(label)
        plt.legend()
        path = f"{plot_dir}/spike_waveforms.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def spike_quality_metrics_tool(
    spike_times: Sequence[float],
    spike_waveforms: Optional[np.ndarray] = None,
    recording_duration: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Composite tool to summarize spike quality.

    Parameters
    ----------
    spike_times : sequence of float
        Spike times in seconds.
    spike_waveforms : np.ndarray, optional
        2D array (n_spikes, n_samples_per_spike).
    recording_duration : float, optional
        Total recording duration in seconds.

    Returns
    -------
    {
      "numeric_results": { ... spike_rate_stats, isi summary, waveform summary ... },
      "image_paths": []
    }
    """
    numeric_results: Dict[str, Any] = {}
    image_paths: List[str] = []

    rate_out = spike_rate_stats_tool(spike_times, recording_duration)
    numeric_results.update(rate_out["numeric_results"])

    isi_out = isi_histogram_tool(spike_times, make_plot=False)
    numeric_results.update(isi_out["numeric_results"])

    if spike_waveforms is not None:
        wf_out = spike_waveform_summary_tool(
            spike_waveforms=spike_waveforms,
            sampling_rate=1.0,  # time scale not needed numerically
            make_plot=False,
        )
        numeric_results.update(wf_out["numeric_results"])

    return {"numeric_results": numeric_results, "image_paths": image_paths}
