# tools/artifact_detection.py
"""
Artifact detection tools for neurophysiology data.

Common artifacts in neural recordings:
- Powerline noise (50/60 Hz)
- Saturation/clipping
- Electrode disconnection spikes
- Baseline drift
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import zscore

from ._utils import DEFAULT_PLOT_DIR, finalize_figure


def detect_powerline_noise(
    data: np.ndarray,
    sampling_rate: float,
    target_freq: float = 60.0,
    bandwidth: float = 2.0,
    harmonics: List[int] = [1, 2, 3],
    snr_threshold: float = 3.0,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Detect and quantify powerline noise contamination.
    
    Powerline noise appears as narrow-band peaks at 50/60 Hz and harmonics.
    Common in clinical and lab recordings due to electrical interference.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    target_freq : float, default=60.0
        Powerline frequency (60 Hz in US, 50 Hz in EU/Asia).
    bandwidth : float, default=2.0
        Frequency bandwidth (±Hz) around target for detection.
    harmonics : List[int], default=[1, 2, 3]
        Which harmonics to check (1=60Hz, 2=120Hz, 3=180Hz).
    snr_threshold : float, default=3.0
        SNR threshold above which contamination is flagged.
    make_plot : bool, default=True
        Generate diagnostic plots.
    plot_dir : str
        Directory for plots.
    
    Returns
    -------
    dict with keys:
        "numeric_results": {
            "contaminated": bool,
            "contamination_score": float (0-1),
            "snr_at_harmonics": dict {harmonic_freq: snr},
            "peak_frequencies": list of detected peak frequencies,
            "recommendation": str ("clean" / "mild" / "severe"),
            "needs_notch_filter": bool
        },
        "image_paths": list of plot paths
    """
    data = np.asarray(data).ravel()
    
    # Compute power spectral density
    freqs, psd = scipy_signal.welch(data, fs=sampling_rate, nperseg=min(2048, len(data)//4))
    
    # Check each harmonic
    snr_results = {}
    contaminated_harmonics = []
    
    for harmonic_num in harmonics:
        harmonic_freq = target_freq * harmonic_num
        
        if harmonic_freq > sampling_rate / 2:
            continue
        
        # Get power at harmonic
        harmonic_mask = (freqs >= harmonic_freq - bandwidth) & (freqs <= harmonic_freq + bandwidth)
        power_at_harmonic = np.mean(psd[harmonic_mask])
        
        # Get background power (excluding harmonic regions)
        background_mask = np.ones_like(freqs, dtype=bool)
        for h in harmonics:
            h_freq = target_freq * h
            background_mask &= ~((freqs >= h_freq - bandwidth) & (freqs <= h_freq + bandwidth))
        
        # Only consider nearby frequencies for background
        nearby_mask = (freqs >= harmonic_freq - 20) & (freqs <= harmonic_freq + 20)
        background_power = np.mean(psd[background_mask & nearby_mask])
        
        # Compute SNR
        snr = power_at_harmonic / (background_power + 1e-12)
        snr_results[f"{int(harmonic_freq)}Hz"] = float(snr)
        
        if snr > snr_threshold:
            contaminated_harmonics.append(harmonic_freq)
    
    # Overall assessment
    is_contaminated = len(contaminated_harmonics) > 0
    max_snr = max(snr_results.values()) if snr_results else 0.0
    
    # Contamination score (0-1)
    contamination_score = min(1.0, (max_snr - 1.0) / 10.0)
    
    # Recommendation
    if max_snr < snr_threshold:
        recommendation = "clean"
        needs_notch = False
    elif max_snr < 6.0:
        recommendation = "mild_contamination"
        needs_notch = True
    else:
        recommendation = "severe_contamination"
        needs_notch = True
    
    # Find all spectral peaks
    peak_indices, _ = scipy_signal.find_peaks(psd, height=np.percentile(psd, 95))
    peak_frequencies = freqs[peak_indices].tolist()
    
    numeric_results = {
        "contaminated": bool(is_contaminated),
        "contamination_score": float(contamination_score),
        "snr_at_harmonics": snr_results,
        "contaminated_harmonics_hz": [float(f) for f in contaminated_harmonics],
        "peak_frequencies": peak_frequencies[:10],  # Top 10 peaks
        "recommendation": recommendation,
        "needs_notch_filter": bool(needs_notch),
    }
    
    # Generate plots
    image_paths = []
    if make_plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Full spectrum
        axes[0].semilogy(freqs, psd, linewidth=1, alpha=0.7)
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power Spectral Density')
        axes[0].set_title('Power Spectrum - Powerline Noise Detection')
        axes[0].grid(True, alpha=0.3)
        
        # Mark harmonic locations
        for harmonic_num in harmonics:
            harmonic_freq = target_freq * harmonic_num
            if harmonic_freq < sampling_rate / 2:
                snr_val = snr_results.get(f"{int(harmonic_freq)}Hz", 0)
                color = 'red' if harmonic_freq in contaminated_harmonics else 'orange'
                axes[0].axvline(harmonic_freq, color=color, linestyle='--', alpha=0.5, 
                               label=f'{int(harmonic_freq)}Hz (SNR={snr_val:.1f})')
        
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(0, min(300, sampling_rate/2))
        
        # Plot 2: Zoomed view around fundamental frequency
        zoom_width = 20
        zoom_mask = (freqs >= target_freq - zoom_width) & (freqs <= target_freq + zoom_width)
        axes[1].plot(freqs[zoom_mask], psd[zoom_mask], linewidth=2)
        axes[1].axvline(target_freq, color='red', linestyle='--', alpha=0.7, 
                       label=f'{target_freq}Hz powerline')
        axes[1].axvspan(target_freq - bandwidth, target_freq + bandwidth, 
                       alpha=0.2, color='red', label='Detection band')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power Spectral Density')
        axes[1].set_title(f'Zoom: {target_freq}Hz Region')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        path = f"{plot_dir}/powerline_detection.png"
        image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def detect_saturation(
    data: np.ndarray,
    sampling_rate: float,
    threshold_method: str = "percentile",
    threshold_value: float = 99.5,
    min_duration_ms: float = 5.0,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Detect signal saturation/clipping events.
    
    Saturation occurs when signal amplitude exceeds amplifier range,
    causing flat-topping. Common with improper gain settings.
    
    Parameters
    ----------
    data : np.ndarray
        1D time-domain signal.
    sampling_rate : float
        Sampling rate in Hz.
    threshold_method : str, default="percentile"
        Method to determine saturation threshold:
        - "percentile": Use percentile of absolute values
        - "absolute": Use fixed threshold value
    threshold_value : float, default=99.5
        If percentile: percentile value (e.g., 99.5)
        If absolute: absolute threshold
    min_duration_ms : float, default=5.0
        Minimum duration (ms) to count as saturation event.
    make_plot : bool
        Generate diagnostic plots.
    plot_dir : str
        Directory for plots.
    
    Returns
    -------
    dict with keys:
        "numeric_results": {
            "saturated": bool,
            "n_saturation_events": int,
            "total_saturated_duration_ms": float,
            "percent_saturated": float,
            "saturation_times": list of (start, end) tuples in seconds,
            "max_amplitude": float,
            "recommendation": str
        },
        "image_paths": list
    """
    data = np.asarray(data).ravel()
    n_samples = len(data)
    
    # Determine saturation threshold
    if threshold_method == "percentile":
        threshold = np.percentile(np.abs(data), threshold_value)
    else:
        threshold = threshold_value
    
    # Find saturated samples
    is_saturated = np.abs(data) >= threshold
    
    # Find continuous saturated regions
    min_samples = int(min_duration_ms * sampling_rate / 1000.0)
    
    # Label connected components
    from scipy.ndimage import label
    labeled_array, n_events = label(is_saturated)
    
    saturation_events = []
    total_saturated_samples = 0
    
    for event_id in range(1, n_events + 1):
        event_mask = labeled_array == event_id
        event_samples = np.sum(event_mask)
        
        if event_samples >= min_samples:
            event_indices = np.where(event_mask)[0]
            start_idx = event_indices[0]
            end_idx = event_indices[-1]
            
            start_time = start_idx / sampling_rate
            end_time = end_idx / sampling_rate
            
            saturation_events.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration_ms": float((end_idx - start_idx) / sampling_rate * 1000),
            })
            
            total_saturated_samples += event_samples
    
    # Calculate statistics
    percent_saturated = (total_saturated_samples / n_samples) * 100
    total_duration_ms = (total_saturated_samples / sampling_rate) * 1000
    
    # Recommendation
    if len(saturation_events) == 0:
        recommendation = "clean"
    elif percent_saturated < 1.0:
        recommendation = "minor_saturation"
    elif percent_saturated < 5.0:
        recommendation = "moderate_saturation"
    else:
        recommendation = "severe_saturation"
    
    numeric_results = {
        "saturated": bool(len(saturation_events) > 0),
        "n_saturation_events": len(saturation_events),
        "total_saturated_duration_ms": float(total_duration_ms),
        "percent_saturated": float(percent_saturated),
        "saturation_events": saturation_events[:20],  # Limit to first 20
        "threshold_used": float(threshold),
        "max_amplitude": float(np.max(np.abs(data))),
        "recommendation": recommendation,
    }
    
    # Generate plots
    image_paths = []
    if make_plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        time = np.arange(n_samples) / sampling_rate
        
        # Plot 1: Full signal with saturation markers
        axes[0].plot(time, data, linewidth=0.5, alpha=0.7)
        axes[0].axhline(threshold, color='red', linestyle='--', alpha=0.5, label='Saturation threshold')
        axes[0].axhline(-threshold, color='red', linestyle='--', alpha=0.5)
        
        # Mark saturation events
        for event in saturation_events[:10]:  # Show first 10
            axes[0].axvspan(event['start_time'], event['end_time'], 
                           alpha=0.3, color='red')
        
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Signal with Saturation Detection ({len(saturation_events)} events, {percent_saturated:.2f}% saturated)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        axes[1].hist(data, bins=100, alpha=0.7, edgecolor='black')
        axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label='Saturation threshold')
        axes[1].axvline(-threshold, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Amplitude')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Amplitude Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        path = f"{plot_dir}/saturation_detection.png"
        image_paths.append(finalize_figure(path))
    
    return {
        "numeric_results": numeric_results,
        "image_paths": image_paths
    }


def detect_line_noise_all_harmonics(
    data: np.ndarray,
    sampling_rate: float,
    line_freq: float = 60.0,
    max_harmonic: Optional[int] = None,
    make_plot: bool = True,
    plot_dir: str = DEFAULT_PLOT_DIR,
) -> Dict[str, Any]:
    """
    Comprehensive line noise detection across all harmonics up to Nyquist.
    
    This is a more thorough version that checks ALL harmonics automatically.
    
    Parameters
    ----------
    data : np.ndarray
        1D signal.
    sampling_rate : float
        Sampling rate in Hz.
    line_freq : float, default=60.0
        Powerline frequency.
    max_harmonic : int, optional
        Maximum harmonic to check. If None, checks up to Nyquist.
    make_plot : bool
        Generate plots.
    plot_dir : str
        Plot directory.
    
    Returns
    -------
    dict with comprehensive harmonic analysis
    """
    data = np.asarray(data).ravel()
    nyquist = sampling_rate / 2.0
    
    # Determine max harmonic
    if max_harmonic is None:
        max_harmonic = int(nyquist / line_freq)
    
    harmonics = list(range(1, max_harmonic + 1))
    
    # Use the main detection function
    result = detect_powerline_noise(
        data=data,
        sampling_rate=sampling_rate,
        target_freq=line_freq,
        harmonics=harmonics,
        make_plot=make_plot,
        plot_dir=plot_dir,
    )
    
    # Add total harmonic distortion estimate
    snr_values = list(result['numeric_results']['snr_at_harmonics'].values())
    total_harmonic_distortion = float(np.sum(snr_values))
    
    result['numeric_results']['total_harmonic_distortion'] = total_harmonic_distortion
    result['numeric_results']['n_harmonics_checked'] = len(harmonics)
    
    return result