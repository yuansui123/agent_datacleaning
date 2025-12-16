# tools/neuro_features.py
"""
Common neurophysiology feature extraction tools.

Standard features used across the field for signal characterization,
quality assessment, and classification.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew

from ._utils import DEFAULT_PLOT_DIR, finalize_figure


def line_length_tool(
    data: np.ndarray,
    sampling_rate: float,
    window_size_sec: Optional[float] = None,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Compute line length - sum of absolute differences between consecutive points.
    
    Line length is a simple but effective measure of signal complexity and
    activity level. High line length = high frequency content or sharp transitions.
    Widely used in seizure detection and signal quality assessment.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    window_size_sec : float, optional
        If provided, compute line length in sliding windows.
    make_plot : bool
        Generate plots.
    plot_dir : str
        Plot directory.
    
    Returns
    -------
    dict with:
        "numeric_results": {
            "total_line_length": float,
            "mean_line_length": float (if windowed),
            "line_length_timeseries": array (if windowed)
        }
    """
    data = np.asarray(data).ravel()
    
    # Compute total line length
    diffs = np.abs(np.diff(data))
    total_line_length = float(np.sum(diffs))
    
    numeric_results = {
        "total_line_length": total_line_length,
        "normalized_line_length": total_line_length / len(data),
    }
    
    image_paths = []
    
    # Windowed line length
    if window_size_sec is not None:
        window_samples = int(window_size_sec * sampling_rate)
        n_windows = len(data) // window_samples
        
        line_lengths = []
        window_times = []
        
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window_data = data[start:end]
            ll = np.sum(np.abs(np.diff(window_data)))
            line_lengths.append(ll)
            window_times.append((start + end) / 2 / sampling_rate)
        
        numeric_results["windowed_line_length"] = line_lengths
        numeric_results["window_times"] = window_times
        numeric_results["mean_line_length"] = float(np.mean(line_lengths))
        numeric_results["std_line_length"] = float(np.std(line_lengths))
        
        if make_plot:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Original signal
            time = np.arange(len(data)) / sampling_rate
            axes[0].plot(time, data, linewidth=0.5, alpha=0.7)
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Amplitude')
            axes[0].set_title('Original Signal')
            axes[0].grid(True, alpha=0.3)
            
            # Line length evolution
            axes[1].plot(window_times, line_lengths, linewidth=2, marker='o', markersize=3)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Line Length')
            axes[1].set_title(f'Line Length Evolution (window={window_size_sec}s)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            path = f"{plot_dir}/line_length.png"
            image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def hjorth_parameters_tool(
    data: np.ndarray,
    sampling_rate: float,
) -> Dict[str, Any]:
    """
    Compute Hjorth parameters: Activity, Mobility, and Complexity.
    
    Classic EEG/neurophysiology features introduced by Hjorth (1970):
    - Activity: Variance of the signal (power)
    - Mobility: Estimate of mean frequency
    - Complexity: Deviation from sinusoidal shape
    
    Widely used in EEG analysis, epilepsy detection, and sleep staging.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    
    Returns
    -------
    dict with:
        "numeric_results": {
            "activity": float,
            "mobility": float,
            "complexity": float
        }
    """
    data = np.asarray(data).ravel()
    
    # First derivative
    d1 = np.diff(data)
    # Second derivative
    d2 = np.diff(d1)
    
    # Activity: variance of signal
    activity = np.var(data)
    
    # Mobility: sqrt(variance of first derivative / variance of signal)
    mobility = np.sqrt(np.var(d1) / (activity + 1e-12))
    
    # Complexity: (mobility of derivative) / (mobility of signal)
    var_d2 = np.var(d2)
    var_d1 = np.var(d1)
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
    complexity = mobility_d1 / (mobility + 1e-12)
    
    numeric_results = {
        "activity": float(activity),
        "mobility": float(mobility),
        "complexity": float(complexity),
    }
    
    return {
        "numeric_results": numeric_results,
        "image_paths": []
    }


def zero_crossing_rate_tool(
    data: np.ndarray,
    sampling_rate: float,
    window_size_ms: float = 100.0,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Compute zero-crossing rate over time.
    
    Zero-crossing rate indicates dominant frequency content.
    High ZCR = high frequency, Low ZCR = low frequency.
    Simple but effective feature for signal classification.
    
    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sampling_rate : float
        Sampling rate in Hz.
    window_size_ms : float
        Window size in milliseconds for computing ZCR.
    make_plot : bool
        Generate plots.
    plot_dir : str
        Plot directory.
    
    Returns
    -------
    dict with ZCR statistics and time evolution
    """
    data = np.asarray(data).ravel()
    
    # Remove DC offset
    data = data - np.mean(data)
    
    # Compute zero crossings
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    total_zcr = len(zero_crossings) / (len(data) / sampling_rate)  # crossings per second
    
    # Windowed ZCR
    window_samples = int(window_size_ms * sampling_rate / 1000)
    n_windows = len(data) // window_samples
    
    zcr_timeseries = []
    window_times = []
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        window_data = data[start:end] - np.mean(data[start:end])
        
        zc = len(np.where(np.diff(np.sign(window_data)))[0])
        zcr = zc / (window_samples / sampling_rate)  # crossings per second
        
        zcr_timeseries.append(zcr)
        window_times.append((start + end) / 2 / sampling_rate)
    
    numeric_results = {
        "total_zcr": float(total_zcr),
        "mean_zcr": float(np.mean(zcr_timeseries)),
        "std_zcr": float(np.std(zcr_timeseries)),
        "zcr_timeseries": zcr_timeseries,
        "window_times": window_times,
    }
    
    image_paths = []
    if make_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Signal with zero crossings marked
        time = np.arange(len(data)) / sampling_rate
        axes[0].plot(time, data, linewidth=0.5, alpha=0.7)
        axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[0].scatter(zero_crossings / sampling_rate, 
                       np.zeros(len(zero_crossings)), 
                       color='red', s=10, alpha=0.5, label='Zero crossings')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Signal with Zero Crossings (Total ZCR: {total_zcr:.1f} Hz)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, min(1.0, time[-1]))  # Show first second
        
        # ZCR evolution
        axes[1].plot(window_times, zcr_timeseries, linewidth=2, marker='o', markersize=3)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Zero-Crossing Rate (Hz)')
        axes[1].set_title(f'ZCR Evolution (window={window_size_ms}ms)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = f"{plot_dir}/zero_crossing_rate.png"
        image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def rms_energy_tool(
    data: np.ndarray,
    sampling_rate: float,
    window_size_ms: float = 100.0,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Compute root-mean-square (RMS) energy over time.
    
    RMS energy quantifies signal power/amplitude.
    Useful for:
    - Activity detection
    - Signal quality assessment
    - Comparing signal strength across channels/sessions
    
    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sampling_rate : float
        Sampling rate in Hz.
    window_size_ms : float
        Window size in milliseconds.
    make_plot : bool
        Generate plots.
    plot_dir : str
        Plot directory.
    
    Returns
    -------
    dict with RMS statistics
    """
    data = np.asarray(data).ravel()
    
    # Total RMS
    total_rms = float(np.sqrt(np.mean(data**2)))
    
    # Windowed RMS
    window_samples = int(window_size_ms * sampling_rate / 1000)
    n_windows = len(data) // window_samples
    
    rms_timeseries = []
    window_times = []
    
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        window_data = data[start:end]
        
        rms = np.sqrt(np.mean(window_data**2))
        rms_timeseries.append(rms)
        window_times.append((start + end) / 2 / sampling_rate)
    
    numeric_results = {
        "total_rms": total_rms,
        "mean_rms": float(np.mean(rms_timeseries)),
        "std_rms": float(np.std(rms_timeseries)),
        "max_rms": float(np.max(rms_timeseries)),
        "min_rms": float(np.min(rms_timeseries)),
        "rms_timeseries": rms_timeseries,
        "window_times": window_times,
    }
    
    image_paths = []
    if make_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original signal
        time = np.arange(len(data)) / sampling_rate
        axes[0].plot(time, data, linewidth=0.5, alpha=0.7)
        axes[0].axhline(total_rms, color='red', linestyle='--', alpha=0.7, label=f'Total RMS: {total_rms:.4f}')
        axes[0].axhline(-total_rms, color='red', linestyle='--', alpha=0.7)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Signal with RMS Level')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RMS evolution
        axes[1].plot(window_times, rms_timeseries, linewidth=2, marker='o', markersize=3, color='orange')
        axes[1].fill_between(window_times, 0, rms_timeseries, alpha=0.3, color='orange')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('RMS Energy')
        axes[1].set_title(f'RMS Energy Evolution (window={window_size_ms}ms)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = f"{plot_dir}/rms_energy.png"
        image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def kurtosis_tool(
    data: np.ndarray,
    sampling_rate: float,
    window_size_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute kurtosis - measure of "tailedness" of distribution.
    
    Kurtosis quantifies outliers/spikes:
    - Normal distribution: kurtosis ≈ 3
    - High kurtosis: heavy tails, many outliers (spiky signal)
    - Low kurtosis: light tails, few outliers (smooth signal)
    
    Useful for artifact detection and signal characterization.
    
    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sampling_rate : float
        Sampling rate in Hz.
    window_size_sec : float, optional
        Window size for time-resolved kurtosis.
    
    Returns
    -------
    dict with kurtosis statistics
    """
    data = np.asarray(data).ravel()
    
    # Overall kurtosis
    kurt = float(kurtosis(data, fisher=True))  # Fisher=True means excess kurtosis (subtract 3)
    skewness = float(skew(data))
    
    numeric_results = {
        "kurtosis": kurt,
        "skewness": skewness,
        "interpretation": "normal" if abs(kurt) < 1 else ("heavy_tailed" if kurt > 1 else "light_tailed"),
    }
    
    # Windowed kurtosis
    if window_size_sec is not None:
        window_samples = int(window_size_sec * sampling_rate)
        n_windows = len(data) // window_samples
        
        kurt_timeseries = []
        window_times = []
        
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            window_data = data[start:end]
            
            k = kurtosis(window_data, fisher=True)
            kurt_timeseries.append(k)
            window_times.append((start + end) / 2 / sampling_rate)
        
        numeric_results["windowed_kurtosis"] = kurt_timeseries
        numeric_results["window_times"] = window_times
        numeric_results["mean_kurtosis"] = float(np.mean(kurt_timeseries))
        numeric_results["std_kurtosis"] = float(np.std(kurt_timeseries))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": []
    }


def shannon_entropy_tool(
    data: np.ndarray,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """
    Compute Shannon entropy of signal amplitude distribution.
    
    Entropy quantifies randomness/predictability:
    - High entropy: random, unpredictable, complex
    - Low entropy: regular, predictable, simple
    
    Useful for:
    - Distinguishing noise from neural activity
    - Assessing signal complexity
    - Quality control
    
    Parameters
    ----------
    data : np.ndarray
        1D signal.
    n_bins : int
        Number of bins for histogram.
    
    Returns
    -------
    dict with entropy value
    """
    data = np.asarray(data).ravel()
    
    # Compute histogram
    counts, _ = np.histogram(data, bins=n_bins)
    
    # Normalize to get probabilities
    probs = counts / np.sum(counts)
    
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    # Shannon entropy
    entropy = float(-np.sum(probs * np.log2(probs)))
    
    # Normalized entropy (0 to 1)
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy / max_entropy
    
    numeric_results = {
        "shannon_entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "max_possible_entropy": float(max_entropy),
    }
    
    return {
        "numeric_results": numeric_results,
        "image_paths": []
    }