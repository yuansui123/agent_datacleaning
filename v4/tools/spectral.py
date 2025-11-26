# tools/spectral.py
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

from ._utils import DEFAULT_PLOT_DIR, finalize_figure


def compute_fft(
    data: np.ndarray,
    sampling_rate: float,
    max_freq: Optional[float] = None,
    plot_window: Optional[Sequence[float]] = None,
    db_scale: bool = False,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Fourier spectrum",
) -> Dict[str, Any]:
    """
    Compute the magnitude FFT of a 1D signal.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    max_freq : float, optional
        If provided, discard frequency components above this value in the PLOT ONLY.
    plot_window : [float, float], optional
        Explicit x-axis limits [f_min, f_max] in Hz. Overrides max_freq if given.
    db_scale : bool
        If True, plot 20*log10(|FFT| + eps) instead of linear magnitude.
    make_plot : bool
        If True, save a plot to disk.
    plot_dir : str
        Directory to store plots.
    label : str
        Plot title.

    Returns
    -------
    dict with keys:
        numeric_results: {
            "fourier": {"freqs": [...], "fft": [...]}
        }
        image_paths: [path_to_fft_plot]  (may be empty if make_plot=False)
    """
    data = np.asarray(data).ravel()
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
        y = 20 * np.log10(fft_vals + 1e-12) if db_scale else fft_vals
        plt.plot(freqs, y)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)" if db_scale else "Magnitude")
        plt.title(label)

        if plot_window is not None and len(plot_window) == 2:
            plt.xlim(plot_window[0], plot_window[1])
        elif max_freq is not None:
            plt.xlim(0.0, max_freq)

        path = f"{plot_dir}/fft_spectrum.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def compute_psd(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: int = 2048,
    noverlap: Optional[int] = None,
    detrend: str = "constant",
    scaling: str = "density",
    max_freq: Optional[float] = None,
    make_plot: bool = True,
    semilogy: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Power Spectral Density",
) -> Dict[str, Any]:
    """
    Compute Welch PSD estimate of the signal.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    nperseg : int
        Length of each Welch segment (in samples).
    noverlap : int, optional
        Number of samples to overlap between segments. If None, defaults to nperseg // 2.
    detrend : {"constant", "linear", None}
        Detrending method passed to scipy.signal.welch.
    scaling : {"density", "spectrum"}
        Scaling passed to scipy.signal.welch.
    max_freq : float, optional
        If provided, restrict the plot to [0, max_freq] Hz.
    make_plot : bool
        If True, save a plot of the PSD.
    semilogy : bool
        If True, plot with logarithmic y-axis.
    plot_dir : str
        Directory to store plots.
    label : str
        Plot title.

    Returns
    -------
    {
      "numeric_results": {
         "psd": {"freqs": [...], "psd": [...]}
      },
      "image_paths": [path_to_plot]
    }
    """
    data = np.asarray(data).ravel()
    f, Pxx = welch(
        data,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling=scaling,
    )

    numeric_results = {
        "psd": {
            "freqs": f.tolist(),
            "psd": Pxx.tolist(),
        }
    }

    image_paths: List[str] = []
    if make_plot:
        plt.figure()
        if semilogy:
            plt.semilogy(f, Pxx)
        else:
            plt.plot(f, Pxx)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(label)
        if max_freq is not None:
            plt.xlim(0.0, max_freq)
        path = f"{plot_dir}/psd.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def compute_spectrogram_tool(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    mode: str = "psd",
    freq_max: Optional[float] = None,
    time_max: Optional[float] = None,
    db_scale: bool = True,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Spectrogram",
) -> Dict[str, Any]:
    """
    Compute a spectrogram of the signal.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    nperseg : int
        Window length in samples.
    noverlap : int, optional
        Number of samples to overlap between windows.
    mode : str
        Mode passed to scipy.signal.spectrogram (e.g., "psd", "magnitude").
    freq_max : float, optional
        If provided, crop frequencies above this value in the plot.
    time_max : float, optional
        If provided, crop times above this value in the plot.
    db_scale : bool
        If True, convert power to dB scale for plotting.
    make_plot : bool
        If True, save a spectrogram plot.
    plot_dir : str
        Directory where the plot will be stored.
    label : str
        Plot title.

    Returns
    -------
    {
      "numeric_results": {
         "spectrogram": {"f": [...], "t": [...], "sxx": [[...], ...]}
      },
      "image_paths": [path_to_plot]
    }
    """
    data = np.asarray(data).ravel()
    f, t, Sxx = spectrogram(
        data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, mode=mode
    )

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
        plot_sxx = Sxx
        if db_scale:
            plot_sxx = 10 * np.log10(Sxx + 1e-12)
        plt.pcolormesh(t, f, plot_sxx, shading="auto")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title(label)
        if freq_max is not None:
            plt.ylim(0.0, freq_max)
        if time_max is not None:
            plt.xlim(0.0, time_max)
        path = f"{plot_dir}/spectrogram.png"
        image_paths.append(finalize_figure(path))

    return {"numeric_results": numeric_results, "image_paths": image_paths}


def bandpower_summary_tool(
    data: np.ndarray,
    sampling_rate: float,
    bands: Optional[Dict[str, List[float]]] = None,
    nperseg: int = 2048,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Compute power in specified frequency bands using Welch PSD.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    bands : dict[str, [float, float]], optional
        Mapping from band name to [f_low, f_high] in Hz.
        If None, defaults to generic bands:
        {"low": [0, 100], "mid": [100, 1000], "high": [1000, Nyquist]}.
    nperseg : int
        Segment length for Welch PSD.
    normalize : bool
        If True, band powers are divided by total power.

    Returns
    -------
    {
      "numeric_results": {
          "bandpower": {
              "band_name": {"power": float, "rel_power": float}
          }
      },
      "image_paths": []
    }
    """
    data = np.asarray(data).ravel()
    f, Pxx = welch(data, fs=sampling_rate, nperseg=nperseg)
    total_power = np.trapz(Pxx, f) + 1e-12

    if bands is None:
        nyq = sampling_rate / 2.0
        bands = {
            "low": [0.0, min(100.0, nyq)],
            "mid": [min(100.0, nyq), min(1000.0, nyq)],
            "high": [min(1000.0, nyq), nyq],
        }

    bp_results: Dict[str, Dict[str, float]] = {}
    for name, (f_lo, f_hi) in bands.items():
        mask = (f >= f_lo) & (f <= f_hi)
        power = float(np.trapz(Pxx[mask], f[mask]))
        if normalize:
            rel_power = power / total_power
        else:
            rel_power = power
        bp_results[name] = {"power": power, "rel_power": rel_power}

    return {
        "numeric_results": {"bandpower": bp_results},
        "image_paths": [],
    }


def spectral_signature_tool(
    data: np.ndarray,
    sampling_rate: float,
    include_fft: bool = True,
    include_psd: bool = True,
    include_spectrogram: bool = True,
    include_bandpower: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Composite tool that computes a "spectral signature" of the signal by
    combining FFT, PSD, spectrogram, and bandpower summaries.

    This is purely diagnostic and does not modify the original data.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    include_fft, include_psd, include_spectrogram, include_bandpower : bool
        Control which sub-analyses are performed.
    plot_dir : str
        Directory for any generated plots.

    Returns
    -------
    {
      "numeric_results": { ... merged keys ... },
      "image_paths": [ ... all generated plots ... ]
    }
    """
    numeric_results: Dict[str, Any] = {}
    image_paths: List[str] = []

    if include_fft:
        out = compute_fft(
            data=data,
            sampling_rate=sampling_rate,
            make_plot=True,
            plot_dir=plot_dir,
        )
        numeric_results.update(out["numeric_results"])
        image_paths.extend(out["image_paths"])

    if include_psd:
        out = compute_psd(
            data=data,
            sampling_rate=sampling_rate,
            make_plot=True,
            plot_dir=plot_dir,
        )
        numeric_results.update(out["numeric_results"])
        image_paths.extend(out["image_paths"])

    if include_spectrogram:
        out = compute_spectrogram_tool(
            data=data,
            sampling_rate=sampling_rate,
            make_plot=True,
            plot_dir=plot_dir,
        )
        numeric_results.update(out["numeric_results"])
        image_paths.extend(out["image_paths"])

    if include_bandpower:
        out = bandpower_summary_tool(
            data=data,
            sampling_rate=sampling_rate,
        )
        numeric_results.update(out["numeric_results"])

    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths,
    }
