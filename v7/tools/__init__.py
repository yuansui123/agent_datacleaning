# tools/__init__.py
"""
Unified import surface for analysis tools.

All functions here are INSPECTION ONLY (no data modification).
"""

from ._utils import DEFAULT_PLOT_DIR

# Spectral / frequency-domain tools
from .spectral import (
    compute_fft,
    compute_psd,
    compute_spectrogram_tool,
    bandpower_summary_tool,
    compute_power_density_matrix,
    compute_power_density_matrix_hilbert,
    spectral_signature_tool,
)

# Pure visualization tools
from .visual import (
    plot_raw_signal_tool,
    plot_zoomed_window_tool,
    raster_plot_tool,
)

# Statistical tools
from .statistics import (
    basic_stats_tool,
    compute_variance_tool,
    compute_snr_tool,
    amplitude_histogram_tool,
    autocorrelation_tool,
    numeric_summary_tool,
)

# Spike-related tools
from .spike import (
    spike_rate_stats_tool,
    isi_histogram_tool,
    spike_waveform_summary_tool,
    spike_quality_metrics_tool,
)

# Metadata tools
from .metadata_tools import (
    metadata_snapshot_tool,
    metadata_completeness_tool,
)

# Artifact detection tools
from .artifact_detection import (
    detect_powerline_noise,
    detect_saturation,
    detect_line_noise_all_harmonics,
)

# Neurophysiology feature tools
from .neuro_features import (
    line_length_tool,
    hjorth_parameters_tool,
    zero_crossing_rate_tool,
    rms_energy_tool,
    kurtosis_tool,
    shannon_entropy_tool,
)

__all__ = [
    # spectral
    "compute_fft",
    "compute_psd",
    "compute_spectrogram_tool",
    "bandpower_summary_tool",
    "compute_power_density_matrix",
    "compute_power_density_matrix_hilbert",
    "spectral_signature_tool",
    # visual
    "plot_raw_signal_tool",
    "plot_zoomed_window_tool",
    "raster_plot_tool",
    # statistics
    "basic_stats_tool",
    "compute_variance_tool",
    "compute_snr_tool",
    "amplitude_histogram_tool",
    "autocorrelation_tool",
    "numeric_summary_tool",
    # spike
    "spike_rate_stats_tool",
    "isi_histogram_tool",
    "spike_waveform_summary_tool",
    "spike_quality_metrics_tool",
    # metadata
    "metadata_snapshot_tool",
    "metadata_completeness_tool",
    # artifact detection
    "detect_powerline_noise",
    "detect_saturation",
    "detect_line_noise_all_harmonics",
    # neurophysiology features
    "line_length_tool",
    "hjorth_parameters_tool",
    "zero_crossing_rate_tool",
    "rms_energy_tool",
    "kurtosis_tool",
    "shannon_entropy_tool",
    # misc
    "DEFAULT_PLOT_DIR",
]