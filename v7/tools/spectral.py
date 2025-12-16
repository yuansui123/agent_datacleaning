# tools/spectral.py
from typing import Dict, Any, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from scipy.signal import butter, sosfiltfilt
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

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

    data = np.asarray(data).ravel()
    freqs = np.fft.rfftfreq(len(data), 1.0 / sampling_rate)
    fft_vals = np.abs(np.fft.rfft(data))

    idx_peak = np.argmax(fft_vals)
    peak_freq = float(freqs[idx_peak])
    peak_amp = float(fft_vals[idx_peak])
    mean_amp = float(np.mean(fft_vals))

    numeric_results = {
        "fft_peak_freq": peak_freq,
        "fft_peak_amp": peak_amp,
        "fft_mean_amp": mean_amp,
    }

    image_paths = []
    if make_plot:
        plt.figure()
        y = 20 * np.log10(fft_vals + 1e-12) if db_scale else fft_vals
        plt.plot(freqs, y)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)" if db_scale else "Magnitude")
        plt.title(label)
        if plot_window is not None:
            plt.xlim(*plot_window)
        elif max_freq is not None:
            plt.xlim(0, max_freq)

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

    data = np.asarray(data).ravel()
    f, Pxx = welch(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap,
                   detrend=detrend, scaling=scaling)

    idx_peak = np.argmax(Pxx)
    peak_freq = float(f[idx_peak])
    peak_power = float(Pxx[idx_peak])
    total_power = float(np.trapz(Pxx, f))

    numeric_results = {
        "psd_peak_freq": peak_freq,
        "psd_peak_power": peak_power,
        "psd_total_power": total_power,
    }

    image_paths = []
    if make_plot:
        plt.figure()
        plt.semilogy(f, Pxx) if semilogy else plt.plot(f, Pxx)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(label)
        if max_freq is not None:
            plt.xlim(0, max_freq)

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
    integrate_bands: bool = False,
    freq_bands: Optional[Dict[str, List[float]]] = None,
    normalize_bands: bool = False,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Spectrogram",
) -> Dict[str, Any]:
    """
    Compute spectrogram with optional frequency band integration.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    nperseg : int, default=256
        Length of each segment.
    noverlap : int, optional
        Number of points to overlap. If None, defaults to nperseg // 2.
    mode : str, default="psd"
        Mode for spectrogram computation ('psd', 'magnitude', 'angle', 'phase').
    freq_max : float, optional
        Maximum frequency to display in plot.
    time_max : float, optional
        Maximum time to display in plot.
    db_scale : bool, default=True
        If True, convert to dB scale for plotting.
    integrate_bands : bool, default=False
        If True, integrate power across specified frequency bands (Power Density Matrix).
    freq_bands : dict[str, [float, float]], optional
        Frequency bands for integration. Only used if integrate_bands=True.
        If None and integrate_bands=True, uses default EEG bands.
    normalize_bands : bool, default=False
        If True, normalize power at each time point to sum to 1.
        Only used if integrate_bands=True.
    make_plot : bool, default=True
        Whether to generate plots.
    plot_dir : str
        Directory to save plots.
    label : str
        Title for the plot.
    
    Returns
    -------
    dict with keys:
        "numeric_results": {
            "spec_max_power": float,
            "spec_dominant_freq": float,
            # If integrate_bands=True, also includes:
            "power_matrix": 2D array (n_bands, n_time_points),
            "freq_bands": dict of band definitions,
            "band_names": list of band names,
            "mean_band_power": dict of mean power per band,
            "peak_time_per_band": dict of peak times per band
        },
        "image_paths": list of plot paths
    """
    data = np.asarray(data).ravel()
    f, t, Sxx = spectrogram(data, fs=sampling_rate, nperseg=nperseg,
                           noverlap=noverlap, mode=mode)

    # Scalar summaries from full spectrogram
    overall_max = float(Sxx.max())
    dominant_freq_bin = int(np.argmax(Sxx.mean(axis=1)))
    dominant_freq = float(f[dominant_freq_bin])

    numeric_results = {
        "spec_max_power": overall_max,
        "spec_dominant_freq": dominant_freq,
    }

    image_paths = []
    
    # ========================================================================
    # OPTIONAL: Integrate power across frequency bands (Power Density Matrix)
    # ========================================================================
    if integrate_bands:
        # Define default frequency bands if not provided
        if freq_bands is None:
            nyq = sampling_rate / 2.0
            freq_bands = {
                "delta": [0.5, min(4.0, nyq)],
                "theta": [4.0, min(8.0, nyq)],
                "alpha": [8.0, min(13.0, nyq)],
                "beta": [13.0, min(30.0, nyq)],
                "gamma": [30.0, min(100.0, nyq)],
            }
        
        band_names = list(freq_bands.keys())
        n_bands = len(band_names)
        n_times = len(t)
        power_matrix = np.zeros((n_bands, n_times))
        
        # Integrate power within each band
        for i, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
            freq_mask = (f >= f_low) & (f <= f_high)
            # Integrate power across frequencies for each time point
            power_matrix[i, :] = np.trapz(Sxx[freq_mask, :], f[freq_mask], axis=0)
        
        # Optional normalization
        if normalize_bands:
            col_sums = power_matrix.sum(axis=0, keepdims=True) + 1e-12
            power_matrix = power_matrix / col_sums
        
        # Compute band statistics
        mean_band_power = {
            band_names[i]: float(power_matrix[i, :].mean()) 
            for i in range(n_bands)
        }
        
        peak_time_per_band = {
            band_names[i]: float(t[np.argmax(power_matrix[i, :])]) 
            for i in range(n_bands)
        }
        
        # Add to numeric results
        numeric_results.update({
            "freq_bands": freq_bands,
            "band_names": band_names,
            "mean_band_power": mean_band_power,
            "peak_time_per_band": peak_time_per_band,
        })

    # ========================================================================
    # PLOTTING
    # ========================================================================
    if make_plot:
        if integrate_bands:
            # Plot both spectrogram and integrated bands
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Full spectrogram
            plot_sxx = 10 * np.log10(Sxx + 1e-12) if db_scale else Sxx
            im1 = axes[0].pcolormesh(t, f, plot_sxx, shading="auto", cmap='viridis')
            axes[0].set_ylabel("Frequency (Hz)")
            axes[0].set_xlabel("Time (s)")
            axes[0].set_title(f"{label} - Full Spectrogram")
            if freq_max is not None:
                axes[0].set_ylim(0, freq_max)
            if time_max is not None:
                axes[0].set_xlim(0, time_max)
            plt.colorbar(im1, ax=axes[0], label='Power (dB)' if db_scale else 'Power')
            
            # Plot 2: Integrated bands (Power Density Matrix)
            im2 = axes[1].imshow(
                power_matrix, 
                aspect='auto', 
                origin='lower',
                extent=[t[0], t[-1], 0, n_bands],
                cmap='viridis'
            )
            axes[1].set_yticks(np.arange(n_bands) + 0.5)
            axes[1].set_yticklabels(band_names)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Frequency Band')
            axes[1].set_title(f'{label} - Integrated Frequency Bands')
            cbar_label = 'Normalized Power' if normalize_bands else 'Integrated Power'
            plt.colorbar(im2, ax=axes[1], label=cbar_label)
            
            plt.tight_layout()
            path = f"{plot_dir}/spectrogram_with_bands.png"
            image_paths.append(finalize_figure(path))
            
        else:
            # Plot only standard spectrogram
            plt.figure()
            plot_sxx = 10 * np.log10(Sxx + 1e-12) if db_scale else Sxx
            plt.pcolormesh(t, f, plot_sxx, shading="auto", cmap='viridis')
            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Time (s)")
            plt.title(label)
            if freq_max is not None:
                plt.ylim(0, freq_max)
            if time_max is not None:
                plt.xlim(0, time_max)
            plt.colorbar(label='Power (dB)' if db_scale else 'Power')

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


def compute_power_density_matrix_hilbert(
    data: np.ndarray,
    sampling_rate: float,
    freq_bands: Optional[Dict[str, List[float]]] = None,
    clip_seconds: float = 0.5,
    filter_order: int = 4,
    envelope_smoothing_sigma: float = 10.0,
    normalize_mode: str = "global",
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Power Density Matrix (Hilbert)",
) -> Dict[str, Any]:
    """
    Compute time-frequency power density matrix using bandpass filtering 
    and Hilbert transform for instantaneous power envelope.
    
    This method provides sample-by-sample temporal resolution, making it ideal
    for detecting transient events and artifacts like powerline noise.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    freq_bands : dict[str, [float, float]], optional
        Frequency bands to analyze. If None, uses extended bands including
        powerline artifact detection (60Hz).
    clip_seconds : float, default=0.5
        Seconds to clip from each edge to remove filter artifacts.
    filter_order : int, default=4
        Order of Butterworth bandpass filter.
    envelope_smoothing_sigma : float, default=10.0
        Gaussian smoothing sigma for envelope (in samples).
    normalize_mode : str, default="global"
        Normalization mode:
        - "global": Normalize all bands to same scale (shows relative power)
        - "per_band": Normalize each band independently (shows dynamics)
        - "none": No normalization
    make_plot : bool, default=True
        Whether to generate visualization plots.
    plot_dir : str
        Directory to save plots.
    label : str
        Title for the plot.
    
    Returns
    -------
    dict with keys:
        "numeric_results": {
            "freq_bands": dict of band definitions,
            "band_names": list of band names,
            "mean_band_power": dict of mean power per band,
            "max_band_power": dict of max power per band,
            "peak_time_per_band": dict of time points with maximum power per band,
            "powerline_contamination": float (if 60Hz band present)
        },
        "image_paths": list of generated plot paths
    """
    
    data = np.asarray(data).ravel()
    n_samples = len(data)
    time = np.arange(n_samples) / sampling_rate
    
    # Define default frequency bands with powerline artifact detection
    if freq_bands is None:
        nyq = sampling_rate / 2.0
        freq_bands = {
            "delta": [0.5, min(4.0, nyq)],
            "theta": [4.0, min(8.0, nyq)],
            "alpha": [8.0, min(13.0, nyq)],
            "beta": [13.0, min(30.0, nyq)],
            "low_gamma": [30.0, min(55.0, nyq)],
            "powerline_60hz": [57.0, min(63.0, nyq)],  # Powerline artifact
            "mid_gamma": [65.0, min(80.0, nyq)],
            "high_gamma": [80.0, min(150.0, nyq)],
            "ripple": [150.0, min(250.0, nyq)],
            "fast_ripple": [250.0, min(500.0, nyq)],
        }
    
    band_names = list(freq_bands.keys())
    n_bands = len(band_names)
    nyquist = sampling_rate / 2.0
    
    # Calculate samples to clip
    clip_samples = int(clip_seconds * sampling_rate)
    
    # Initialize power matrix
    pdm = np.zeros((n_bands, n_samples))
    
    # Compute power envelope for each band
    for i, (band_name, (low_freq, high_freq)) in enumerate(freq_bands.items()):
        # Normalize frequencies for filter design
        low = max(low_freq / nyquist, 0.001)
        high = min(high_freq / nyquist, 0.999)
        
        if low >= high:
            continue
        
        # Design Butterworth bandpass filter (SOS for stability)
        sos = butter(filter_order, [low, high], btype='band', output='sos')
        
        # Apply zero-phase filtering
        filtered_signal = sosfiltfilt(sos, data)
        
        # Compute analytic signal and instantaneous envelope
        analytical_signal = hilbert(filtered_signal)
        envelope = np.abs(analytical_signal)
        
        # Smooth the envelope to reduce noise
        if envelope_smoothing_sigma > 0:
            envelope = gaussian_filter1d(envelope, sigma=envelope_smoothing_sigma)
        
        pdm[i, :] = envelope
    
    # Clip edges to remove filter artifacts
    if clip_samples > 0:
        pdm_clipped = pdm[:, clip_samples:-clip_samples]
        time_clipped = time[clip_samples:-clip_samples]
    else:
        pdm_clipped = pdm
        time_clipped = time
    
    # Apply normalization
    if normalize_mode == "global":
        # Normalize all bands to same scale (shows relative power across bands)
        global_min = pdm_clipped.min()
        global_max = pdm_clipped.max()
        pdm_normalized = (pdm_clipped - global_min) / (global_max - global_min + 1e-12)
    elif normalize_mode == "per_band":
        # Normalize each band independently (shows temporal dynamics)
        pdm_normalized = np.zeros_like(pdm_clipped)
        for i in range(n_bands):
            row_min = pdm_clipped[i, :].min()
            row_max = pdm_clipped[i, :].max()
            pdm_normalized[i, :] = (pdm_clipped[i, :] - row_min) / (row_max - row_min + 1e-12)
    else:  # "none"
        pdm_normalized = pdm_clipped
    
    # Compute summary statistics
    mean_band_power = {
        band_names[i]: float(pdm_clipped[i, :].mean()) 
        for i in range(n_bands)
    }
    
    max_band_power = {
        band_names[i]: float(pdm_clipped[i, :].max()) 
        for i in range(n_bands)
    }
    
    peak_time_per_band = {
        band_names[i]: float(time_clipped[np.argmax(pdm_clipped[i, :])]) 
        for i in range(n_bands)
    }
    
    # Special check for powerline contamination
    powerline_contamination = None
    if "powerline_60hz" in band_names:
        idx = band_names.index("powerline_60hz")
        total_power = sum(mean_band_power.values())
        powerline_contamination = float(mean_band_power["powerline_60hz"] / total_power)
    
    numeric_results = {
        "freq_bands": freq_bands,
        "band_names": band_names,
        "mean_band_power": mean_band_power,
        "max_band_power": max_band_power,
        "peak_time_per_band": peak_time_per_band,
        "normalize_mode": normalize_mode,
    }
    
    if powerline_contamination is not None:
        numeric_results["powerline_contamination"] = powerline_contamination
    
    # Generate plots
    image_paths = []
    if make_plot:
        # Plot 1: Heatmap
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Heatmap
        im = axes[0].imshow(
            pdm_normalized, 
            aspect='auto', 
            origin='lower',
            extent=[time_clipped[0], time_clipped[-1], 0, n_bands],
            cmap='hot',
            interpolation='bilinear'
        )
        axes[0].set_yticks(np.arange(n_bands) + 0.5)
        axes[0].set_yticklabels(band_names)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency Band')
        axes[0].set_title(f'{label} - Heatmap ({normalize_mode} normalization)')
        
        cbar_label = 'Normalized Power Envelope'
        if normalize_mode == "none":
            cbar_label = 'Power Envelope (arbitrary units)'
        plt.colorbar(im, ax=axes[0], label=cbar_label)
        
        # Plot 2: Line plot of each band over time
        for i, band_name in enumerate(band_names):
            alpha = 0.5 if band_name == "powerline_60hz" else 0.7
            linestyle = '--' if band_name == "powerline_60hz" else '-'
            axes[1].plot(time_clipped, pdm_normalized[i, :], 
                        label=band_name, alpha=alpha, linestyle=linestyle)
        
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized Power')
        axes[1].set_title(f'{label} - Temporal Evolution')
        axes[1].legend(loc='best', ncol=2, fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = f"{plot_dir}/power_density_matrix_hilbert.png"
        image_paths.append(finalize_figure(path))
        
        # Plot 3: Bar chart of mean power per band
        plt.figure(figsize=(12, 6))
        colors = ['red' if name == "powerline_60hz" else 'steelblue' 
                 for name in band_names]
        plt.bar(band_names, [mean_band_power[b] for b in band_names], 
               color=colors, alpha=0.7)
        plt.xlabel('Frequency Band')
        plt.ylabel('Mean Power (original scale)')
        plt.title(f'{label} - Mean Band Power')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        path = f"{plot_dir}/mean_band_power_hilbert.png"
        image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def compute_power_density_matrix(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    freq_bands: Optional[Dict[str, List[float]]] = None,
    normalize: bool = True,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
    label: str = "Power Density Matrix",
) -> Dict[str, Any]:
    """
    Compute time-frequency power density matrix using spectrogram.
    
    This tool computes a 2D matrix showing how power is distributed across
    frequency bands over time, useful for detecting temporal changes in
    spectral content.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    nperseg : int, default=256
        Length of each segment for spectrogram computation.
    noverlap : int, optional
        Number of points to overlap between segments. 
        If None, defaults to nperseg // 2.
    freq_bands : dict[str, [float, float]], optional
        Frequency bands to analyze. Keys are band names, values are [f_low, f_high].
        If None, uses default EEG bands:
        {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 13], 
         "beta": [13, 30], "gamma": [30, 100]}
    normalize : bool, default=True
        If True, normalize power at each time point to sum to 1.
    make_plot : bool, default=True
        Whether to generate visualization plots.
    plot_dir : str
        Directory to save plots.
    label : str
        Title for the plot.
    
    Returns
    -------
    dict with keys:
        "numeric_results": {
            "power_matrix": 2D array of shape (n_bands, n_time_points),
            "freq_bands": dict of band definitions,
            "band_names": list of band names,
            "mean_band_power": dict of mean power per band over time,
            "temporal_variance": dict of variance per band over time,
            "peak_time_per_band": dict of time points with maximum power per band
        },
        "image_paths": list of generated plot paths
    """
    data = np.asarray(data).ravel()
    
    # Compute spectrogram
    if noverlap is None:
        noverlap = nperseg // 2
    
    f, t, Sxx = spectrogram(
        data, 
        fs=sampling_rate, 
        nperseg=nperseg,
        noverlap=noverlap, 
        mode='psd'
    )
    
    # Define default frequency bands (EEG-style)
    if freq_bands is None:
        nyq = sampling_rate / 2.0
        freq_bands = {
            "delta": [0.5, min(4.0, nyq)],
            "theta": [4.0, min(8.0, nyq)],
            "alpha": [8.0, min(13.0, nyq)],
            "beta": [13.0, min(30.0, nyq)],
            "gamma": [30.0, min(100.0, nyq)],
        }
    
    # Compute power in each band over time
    band_names = list(freq_bands.keys())
    n_bands = len(band_names)
    n_times = len(t)
    power_matrix = np.zeros((n_bands, n_times))
    
    for i, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
        # Find frequency bins in this band
        freq_mask = (f >= f_low) & (f <= f_high)
        # Integrate power across frequencies for each time point
        power_matrix[i, :] = np.trapz(Sxx[freq_mask, :], f[freq_mask], axis=0)
    
    # Normalize if requested
    if normalize:
        # Normalize each time column to sum to 1
        col_sums = power_matrix.sum(axis=0, keepdims=True) + 1e-12
        power_matrix = power_matrix / col_sums
    
    # Compute summary statistics
    mean_band_power = {
        band_names[i]: float(power_matrix[i, :].mean()) 
        for i in range(n_bands)
    }
    
    temporal_variance = {
        band_names[i]: float(power_matrix[i, :].var()) 
        for i in range(n_bands)
    }
    
    peak_time_per_band = {
        band_names[i]: float(t[np.argmax(power_matrix[i, :])]) 
        for i in range(n_bands)
    }
    
    numeric_results = {
        "freq_bands": freq_bands,
        "band_names": band_names,
        "mean_band_power": mean_band_power,
        "temporal_variance": temporal_variance,
        "peak_time_per_band": peak_time_per_band,
    }
    
    # Generate plots
    image_paths = []
    if make_plot:
        # Plot 1: Heatmap of power density matrix
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Heatmap
        im = axes[0].imshow(
            power_matrix, 
            aspect='auto', 
            origin='lower',
            extent=[t[0], t[-1], 0, n_bands],
            cmap='viridis'
        )
        axes[0].set_yticks(np.arange(n_bands) + 0.5)
        axes[0].set_yticklabels(band_names)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency Band')
        axes[0].set_title(f'{label} - Heatmap')
        plt.colorbar(im, ax=axes[0], label='Normalized Power' if normalize else 'Power')
        
        # Plot 2: Line plot of each band over time
        for i, band_name in enumerate(band_names):
            axes[1].plot(t, power_matrix[i, :], label=band_name, alpha=0.7)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Normalized Power' if normalize else 'Power')
        axes[1].set_title(f'{label} - Temporal Evolution')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = f"{plot_dir}/power_density_matrix.png"
        image_paths.append(finalize_figure(path))
        
        # Plot 3: Bar chart of mean power per band
        plt.figure(figsize=(10, 6))
        plt.bar(band_names, [mean_band_power[b] for b in band_names], alpha=0.7)
        plt.xlabel('Frequency Band')
        plt.ylabel('Mean Power')
        plt.title(f'{label} - Mean Band Power')
        plt.grid(True, alpha=0.3, axis='y')
        
        path = f"{plot_dir}/mean_band_power.png"
        image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def spectral_signature_tool(
    data: np.ndarray,
    sampling_rate: float,
    include_fft: bool = True,
    include_psd: bool = True,
    include_spectrogram: bool = True,
    include_bandpower: bool = True,
    include_power_density_matrix: bool = False,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Composite tool that computes a "spectral signature" of the signal by
    combining FFT, PSD, spectrogram, bandpower summaries, and optionally
    power density matrix.

    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    include_fft, include_psd, include_spectrogram, include_bandpower : bool
        Control which sub-analyses are performed.
    include_power_density_matrix : bool
        Whether to include power density matrix analysis.
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

    if include_power_density_matrix:
        out = compute_power_density_matrix(
            data=data,
            sampling_rate=sampling_rate,
            make_plot=True,
            plot_dir=plot_dir,
        )
        numeric_results.update(out["numeric_results"])
        image_paths.extend(out["image_paths"])

    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths,
    }