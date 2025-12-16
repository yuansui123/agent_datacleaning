# tools_schema.py
from typing import List, Dict, Any

TOOLS_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "compute_fft",
        "description": (
            "Compute the FFT of a 1D signal and optionally plot it. "
            "This tool now returns ONLY scalar summary values (peak frequency, peak amplitude, etc.) "
            "and image paths, never raw FFT arrays."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate of the signal in Hz. Used to convert indices to frequencies.",
                },
                "max_freq": {
                    "type": ["number", "null"],
                    "description": "Optional maximum frequency (Hz) to display in the FFT plot. If null, show full spectrum.",
                    "default": None,
                },
                "plot_window": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": "Optional [f_min, f_max] in Hz for x-axis limits; overrides max_freq if provided.",
                    "default": None,
                },
                "db_scale": {
                    "type": "boolean",
                    "description": "If true, FFT magnitude is plotted in decibels (20*log10(|FFT| + eps)).",
                    "default": False,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save a PNG of the FFT magnitude spectrum.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the FFT plot should be saved.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title to display on the FFT plot.",
                    "default": "Fourier spectrum",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "compute_psd",
        "description": (
            "Compute the Welch PSD estimate and optionally plot it. "
            "This tool now returns ONLY scalar PSD summary metrics "
            "(peak frequency, peak power, total power) and image paths, not full PSD arrays."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate of the signal in Hz.",
                },
                "nperseg": {
                    "type": "integer",
                    "description": "Segment length (in samples) used in the Welch PSD computation.",
                    "default": 2048,
                },
                "noverlap": {
                    "type": ["integer", "null"],
                    "description": "Number of overlapping samples between segments. If null, defaults to nperseg // 2.",
                    "default": None,
                },
                "detrend": {
                    "type": ["string", "null"],
                    "description": "Detrending method passed to scipy.signal.welch (e.g., 'constant', 'linear', or null).",
                    "default": "constant",
                },
                "scaling": {
                    "type": "string",
                    "description": "Scaling parameter for welch ('density' or 'spectrum').",
                    "default": "density",
                },
                "max_freq": {
                    "type": ["number", "null"],
                    "description": "Optional maximum frequency (Hz) to display in the PSD plot.",
                    "default": None,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save a PNG of the PSD.",
                    "default": True,
                },
                "semilogy": {
                    "type": "boolean",
                    "description": "If true, plot PSD with logarithmic y-axis.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the PSD plot should be saved.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title of the PSD plot.",
                    "default": "Power Spectral Density",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "compute_spectrogram_tool",
        "description": (
            "Compute a time-frequency spectrogram with OPTIONAL frequency band integration. "
            "When integrate_bands=False (default), returns standard spectrogram summary. "
            "When integrate_bands=True, also integrates power across specified frequency bands "
            "(Power Density Matrix), providing band-level summaries useful for neurophysiology analysis. "
            "This enables both detailed frequency analysis and physiologically meaningful band summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate of the signal in Hz.",
                },
                "nperseg": {
                    "type": "integer",
                    "description": "Window length in samples for spectrogram computation.",
                    "default": 256,
                },
                "noverlap": {
                    "type": ["integer", "null"],
                    "description": "Number of overlapping samples between successive windows.",
                    "default": None,
                },
                "mode": {
                    "type": "string",
                    "description": "Mode parameter passed to scipy.signal.spectrogram (e.g., 'psd', 'magnitude').",
                    "default": "psd",
                },
                "freq_max": {
                    "type": ["number", "null"],
                    "description": "Optional maximum frequency (Hz) to display in the spectrogram plot.",
                    "default": None,
                },
                "time_max": {
                    "type": ["number", "null"],
                    "description": "Optional maximum time (seconds) to display in the spectrogram plot.",
                    "default": None,
                },
                "db_scale": {
                    "type": "boolean",
                    "description": "If true, convert power to dB scale for plotting.",
                    "default": True,
                },
                "integrate_bands": {
                    "type": "boolean",
                    "description": (
                        "If true, integrate spectrogram power across frequency bands to create Power Density Matrix. "
                        "This reduces hundreds of frequency bins to a few physiologically meaningful bands "
                        "(e.g., delta, theta, alpha, beta, gamma), making it easier to interpret temporal dynamics "
                        "of different frequency components."
                    ),
                    "default": False,
                },
                "freq_bands": {
                    "type": ["object", "null"],
                    "description": (
                        "Frequency bands for integration (only used if integrate_bands=True). "
                        "Mapping from band name to [f_low, f_high] in Hz. "
                        "If null, uses default EEG bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), "
                        "beta (13-30 Hz), gamma (30-100 Hz)."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "default": None,
                },
                "normalize_bands": {
                    "type": "boolean",
                    "description": (
                        "If true (and integrate_bands=True), normalize power at each time point to sum to 1. "
                        "This shows relative power distribution across bands rather than absolute power."
                    ),
                    "default": False,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save PNG image(s). With integrate_bands=True, creates dual plot showing both full spectrogram and integrated bands.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the spectrogram plot is saved.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title for the spectrogram plot.",
                    "default": "Spectrogram",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "bandpower_summary_tool",
        "description": "Compute power within named frequency bands using the Welch PSD.",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate of the signal in Hz.",
                },
                "bands": {
                    "type": ["object", "null"],
                    "description": (
                        "Optional mapping from band name to [f_low, f_high] in Hz. "
                        "If null, generic bands (low/mid/high) are used."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "default": None,
                },
                "nperseg": {
                    "type": "integer",
                    "description": "Segment length in samples for PSD estimation.",
                    "default": 2048,
                },
                "normalize": {
                    "type": "boolean",
                    "description": (
                        "If true, band powers are divided by total power to yield relative power fractions."
                    ),
                    "default": True,
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "spectral_signature_tool",
        "description": "Composite tool: compute FFT, PSD, spectrogram, and bandpower summary for a signal.",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "include_fft": {
                    "type": "boolean",
                    "description": "If true, include FFT computation and FFT plot.",
                    "default": True,
                },
                "include_psd": {
                    "type": "boolean",
                    "description": "If true, include PSD computation and PSD plot.",
                    "default": True,
                },
                "include_spectrogram": {
                    "type": "boolean",
                    "description": "If true, include spectrogram computation and plot.",
                    "default": True,
                },
                "include_bandpower": {
                    "type": "boolean",
                    "description": "If true, include bandpower summary metrics.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where all generated plots are saved.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "plot_raw_signal_tool",
        "description": "Plot the raw time-domain signal without modifying it.",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate of the signal in Hz.",
                },
                "t_max": {
                    "type": ["number", "null"],
                    "description": "Optional maximum time (seconds) to display. If null, show full recording.",
                    "default": None,
                },
                "ylim": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": "Optional [y_min, y_max] amplitude limits for visualization.",
                    "default": None,
                },
                "label": {
                    "type": "string",
                    "description": "Title displayed on the raw signal plot.",
                    "default": "Raw signal",
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the raw signal plot is stored.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "plot_zoomed_window_tool",
        "description": "Plot a zoomed time window of the raw signal.",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "t_start": {
                    "type": "number",
                    "description": "Start time of the window in seconds.",
                },
                "t_end": {
                    "type": "number",
                    "description": "End time of the window in seconds (must be > t_start).",
                },
                "ylim": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": "Optional [y_min, y_max] for amplitude limits.",
                    "default": None,
                },
                "label": {
                    "type": "string",
                    "description": "Title shown on the zoomed plot.",
                    "default": "Zoomed window",
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for saving the plot.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate", "t_start", "t_end"],
        },
    },
    {
        "name": "raster_plot_tool",
        "description": "Visualize multi-trial or multi-channel activity as a raster-like intensity image.",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz used to label the time axis.",
                },
                "label": {
                    "type": "string",
                    "description": "Title for the raster-like plot.",
                    "default": "Raster-like plot",
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the raster plot image is stored.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "basic_stats_tool",
        "description": "Compute basic descriptive statistics: mean, std, variance, min, max, skew, kurtosis.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_variance_tool",
        "description": "Compute the variance of the signal.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_snr_tool",
        "description": "Compute a simple SNR estimate as mean(signal^2) / var(signal - mean(signal)).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "amplitude_histogram_tool",
        "description": (
            "Compute and optionally plot an amplitude histogram. "
            "This tool now returns ONLY scalar descriptors of the histogram "
            "(peak bin center, mean amplitude, standard deviation) and image paths. "
            "It no longer returns bin_edges or counts arrays."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bins": {
                    "type": "integer",
                    "description": "Number of histogram bins for amplitude distribution.",
                    "default": 100,
                },
                "range": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "description": "Optional [low, high] amplitude range. If null, use data min and max.",
                    "default": None,
                },
                "density": {
                    "type": "boolean",
                    "description": "If true, normalize histogram to probability density.",
                    "default": True,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save a PNG of the amplitude histogram.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for the histogram plot.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title of the amplitude histogram plot.",
                    "default": "Amplitude distribution",
                },
            },
            "required": [],
        },
    },
    {
        "name": "autocorrelation_tool",
        "description": (
            "Compute autocorrelation of the signal and optionally plot it. "
            "This tool now returns ONLY scalar autocorrelation metrics "
            "(zero-lag value, lag-1 value, decay time) and image paths. "
            "It no longer returns full autocorrelation traces or lag arrays."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "max_lag_seconds": {
                    "type": "number",
                    "description": "Maximum lag in seconds for which autocorrelation is computed.",
                },
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "normalize": {
                    "type": "boolean",
                    "description": "If true, normalize by zero-lag so autocorrelation at lag 0 is 1.",
                    "default": True,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save a PNG of autocorrelation vs lag.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the autocorrelation plot is stored.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title for the autocorrelation plot.",
                    "default": "Autocorrelation",
                },
            },
            "required": ["max_lag_seconds", "sampling_rate"],
        },
    },
    {
        "name": "numeric_summary_tool",
        "description": "Composite tool: basic stats, SNR, and short-lag autocorrelation (no plots).",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz (used for autocorrelation lag calculation).",
                }
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "spike_rate_stats_tool",
        "description": "Compute spike rate statistics from spike times.",
        "parameters": {
            "type": "object",
            "properties": {
                "spike_times": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of spike times in seconds.",
                },
                "recording_duration": {
                    "type": ["number", "null"],
                    "description": "Optional recording duration in seconds. If null, inferred from max spike time.",
                    "default": None,
                },
            },
            "required": ["spike_times"],
        },
    },
    {
        "name": "isi_histogram_tool",
        "description": "Compute and optionally plot the distribution of inter-spike intervals (ISI).",
        "parameters": {
            "type": "object",
            "properties": {
                "spike_times": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Spike times in seconds, not necessarily sorted.",
                },
                "bins": {
                    "type": "integer",
                    "description": "Number of bins in ISI histogram.",
                    "default": 50,
                },
                "max_isi": {
                    "type": ["number", "null"],
                    "description": "Optional maximum ISI (s); ISIs above this are discarded.",
                    "default": None,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save a PNG ISI histogram.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for the ISI histogram plot.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title of the ISI histogram plot.",
                    "default": "Inter-spike interval (ISI) histogram",
                },
            },
            "required": ["spike_times"],
        },
    },
    {
        "name": "spike_waveform_summary_tool",
        "description": "Summarize spike waveforms (mean, std, and optionally plot example waveforms).",
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz for converting samples to milliseconds on the x-axis.",
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, plot a subset of waveforms and the mean ± std.",
                    "default": True,
                },
                "n_examples": {
                    "type": "integer",
                    "description": "Maximum number of individual waveforms to overlay in the plot.",
                    "default": 100,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory where the spike waveform plot is saved.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title of the spike waveform plot.",
                    "default": "Spike waveforms",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "spike_quality_metrics_tool",
        "description": "Composite tool combining spike rate, ISI statistics, and optional waveform summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "spike_times": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Spike times in seconds.",
                },
                "recording_duration": {
                    "type": ["number", "null"],
                    "description": "Optional total recording duration in seconds.",
                    "default": None,
                },
                "include_waveforms": {
                    "type": "boolean",
                    "description": "If true, the agent should also call spike_waveform_summary_tool with matching waveforms.",
                    "default": False,
                },
            },
            "required": ["spike_times"],
        },
    },
    {
        "name": "metadata_snapshot_tool",
        "description": "Return the full metadata dictionary as a numeric result for interpreters.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "metadata_completeness_tool",
        "description": "Compute a simple completeness score for metadata given required fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "required_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metadata keys that are expected to be present.",
                }
            },
            "required": ["required_fields"],
        },
    },
    {
        "name": "compute_power_density_matrix",
        "description": (
            "Compute time-frequency Power Density Matrix using STFT-based spectrogram with frequency band integration. "
            "This collapses full spectrogram into predefined frequency bands (e.g., delta, theta, alpha, beta, gamma), "
            "showing how power in each band evolves over time. More efficient than full spectrogram for long signals "
            "and provides physiologically interpretable summaries. Good for standard spectral analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "nperseg": {
                    "type": "integer",
                    "description": "Length of each segment for spectrogram computation (affects frequency resolution).",
                    "default": 256,
                },
                "noverlap": {
                    "type": ["integer", "null"],
                    "description": "Number of points to overlap between segments. If None, defaults to nperseg // 2.",
                    "default": None,
                },
                "freq_bands": {
                    "type": ["object", "null"],
                    "description": (
                        "Frequency bands to analyze as dict mapping band names to [f_low, f_high] in Hz. "
                        "If None, uses default EEG bands: delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-100)."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "default": None,
                },
                "normalize": {
                    "type": "boolean",
                    "description": "If true, normalize power at each time point so bands sum to 1.",
                    "default": True,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Whether to generate visualization plots (heatmap, temporal evolution, and bar chart).",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory to save plots.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title for the plot.",
                    "default": "Power Density Matrix",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "compute_power_density_matrix_hilbert",
        "description": (
            "Compute time-frequency Power Density Matrix using bandpass filtering + Hilbert transform for instantaneous envelope. "
            "Provides SAMPLE-BY-SAMPLE temporal resolution (much higher than STFT), making it ideal for detecting transient events, "
            "sharp artifacts, and powerline noise. Includes explicit 60Hz powerline detection band by default. "
            "Uses zero-phase Butterworth filtering with edge clipping to avoid artifacts. "
            "Best for: powerline artifact detection, transient event detection, high temporal precision tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "freq_bands": {
                    "type": ["object", "null"],
                    "description": (
                        "Frequency bands to analyze. If None, uses extended bands including powerline artifact detection: "
                        "delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), low_gamma (30-55), "
                        "powerline_60hz (57-63), mid_gamma (65-80), high_gamma (80-150), ripple (150-250), fast_ripple (250-500)."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "default": None,
                },
                "clip_seconds": {
                    "type": "number",
                    "description": "Seconds to clip from each edge to remove filter artifacts.",
                    "default": 0.5,
                },
                "filter_order": {
                    "type": "integer",
                    "description": "Order of Butterworth bandpass filter.",
                    "default": 4,
                },
                "envelope_smoothing_sigma": {
                    "type": "number",
                    "description": "Gaussian smoothing sigma for envelope (in samples). Set to 0 for no smoothing.",
                    "default": 10.0,
                },
                "normalize_mode": {
                    "type": "string",
                    "description": (
                        "Normalization mode: 'global' (all bands same scale, shows relative power), "
                        "'per_band' (each band normalized independently, shows dynamics), or 'none' (raw values)."
                    ),
                    "default": "global",
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Whether to generate visualization plots.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory to save plots.",
                    "default": "inspection_plots",
                },
                "label": {
                    "type": "string",
                    "description": "Title for the plot.",
                    "default": "Power Density Matrix (Hilbert)",
                },
                "file_name": {
                    "type": "string",
                    "description": "Base filename for saved plots. if call multiple times, use unique names.",
                    "default": "power_density_matrix",
                },
            },
            "required": ["sampling_rate", "file_name"],
        },
    },
    {
        "name": "detect_powerline_noise",
        "description": (
            "Detect and quantify powerline noise contamination (50/60 Hz and harmonics). "
            "Powerline noise appears as narrow-band peaks at the fundamental frequency and its harmonics, "
            "common in clinical and laboratory recordings due to electrical interference. "
            "Returns contamination score, SNR at each harmonic, and recommendation for notch filtering. "
            "Essential for data quality assessment before analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "target_freq": {
                    "type": "number",
                    "description": "Powerline frequency: 60 Hz in US, 50 Hz in EU/Asia/most other regions.",
                    "default": 60.0,
                },
                "bandwidth": {
                    "type": "number",
                    "description": "Frequency bandwidth (±Hz) around target frequency for detection window.",
                    "default": 2.0,
                },
                "harmonics": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Which harmonics to check (e.g., [1, 2, 3] checks 60Hz, 120Hz, 180Hz).",
                    "default": [1, 2, 3],
                },
                "snr_threshold": {
                    "type": "number",
                    "description": "SNR threshold above which contamination is flagged.",
                    "default": 3.0,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Generate diagnostic plots showing power spectrum with harmonic markers.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for plots.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "detect_saturation",
        "description": (
            "Detect signal saturation/clipping events where amplitude exceeds amplifier range. "
            "Saturation causes flat-topping and is common with improper gain settings in neural recordings. "
            "Returns number of saturation events, total duration, percentage of data affected, and timestamps. "
            "Critical for identifying unusable data segments in clinical and experimental recordings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "threshold_method": {
                    "type": "string",
                    "description": "Method to determine saturation threshold: 'percentile' or 'absolute'.",
                    "default": "percentile",
                },
                "threshold_value": {
                    "type": "number",
                    "description": "If percentile method: percentile value (e.g., 99.5). If absolute: fixed threshold value.",
                    "default": 99.5,
                },
                "min_duration_ms": {
                    "type": "number",
                    "description": "Minimum duration (ms) for an event to count as saturation.",
                    "default": 5.0,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Generate diagnostic plots showing signal with saturation events marked.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for plots.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "detect_line_noise_all_harmonics",
        "description": (
            "Comprehensive powerline noise detection checking ALL harmonics up to Nyquist frequency. "
            "More thorough than basic powerline detection - automatically determines maximum harmonic based on sampling rate. "
            "Includes total harmonic distortion estimate. Use this for comprehensive powerline assessment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "line_freq": {
                    "type": "number",
                    "description": "Powerline frequency (60 Hz in US, 50 Hz elsewhere).",
                    "default": 60.0,
                },
                "max_harmonic": {
                    "type": ["integer", "null"],
                    "description": "Maximum harmonic number to check. If None, checks all harmonics up to Nyquist.",
                    "default": None,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Generate diagnostic plots.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for plots.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "line_length_tool",
        "description": (
            "Compute line length - sum of absolute differences between consecutive samples. "
            "Line length is a simple but effective measure of signal complexity and activity level widely used in neuroscience. "
            "High line length indicates high frequency content or sharp transitions. "
            "Applications: seizure detection, signal quality assessment, activity quantification. "
            "Can compute windowed line length to track temporal changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "window_size_sec": {
                    "type": ["number", "null"],
                    "description": "If provided, compute line length in sliding windows of this duration (seconds).",
                    "default": None,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Generate plots showing signal and line length evolution.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for plots.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "hjorth_parameters_tool",
        "description": (
            "Compute Hjorth parameters (Hjorth, 1970) - classic EEG features widely used in neurophysiology: "
            "Activity = variance of signal (power), "
            "Mobility = estimate of mean frequency (sqrt of variance of first derivative / variance of signal), "
            "Complexity = deviation from sinusoidal shape (mobility of derivative / mobility of signal). "
            "Applications: EEG analysis, epilepsy detection, sleep staging, signal characterization. "
            "These are fundamental features in clinical neurophysiology and brain-computer interfaces."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "zero_crossing_rate_tool",
        "description": (
            "Compute zero-crossing rate (ZCR) - number of times signal crosses zero per second. "
            "ZCR is a simple but effective indicator of dominant frequency content: "
            "high ZCR = high frequency, low ZCR = low frequency. "
            "Can be computed in sliding windows to track temporal changes. "
            "Applications: signal classification, frequency estimation, quality assessment. "
            "Widely used in speech processing and adapted for neural signal analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "window_size_ms": {
                    "type": "number",
                    "description": "Window size in milliseconds for computing windowed ZCR.",
                    "default": 100.0,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Generate plots showing signal with zero crossings and ZCR evolution.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for plots.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "rms_energy_tool",
        "description": (
            "Compute root-mean-square (RMS) energy over time - quantifies signal power/amplitude. "
            "RMS energy is fundamental for: activity detection, signal quality assessment, "
            "comparing signal strength across channels or recording sessions. "
            "Computed in sliding windows to show temporal evolution of signal energy. "
            "Essential metric in neurophysiology, audio processing, and signal quality control."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "window_size_ms": {
                    "type": "number",
                    "description": "Window size in milliseconds for windowed RMS computation.",
                    "default": 100.0,
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Generate plots showing signal and RMS energy evolution.",
                    "default": True,
                },
                "plot_dir": {
                    "type": "string",
                    "description": "Directory for plots.",
                    "default": "inspection_plots",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "kurtosis_tool",
        "description": (
            "Compute kurtosis - fourth moment measuring 'tailedness' of amplitude distribution. "
            "Quantifies presence of outliers/extreme values: "
            "kurtosis ≈ 0 (normal distribution), >0 (heavy tails, many spikes/outliers), <0 (light tails, smooth signal). "
            "Can compute windowed kurtosis to detect transient artifacts. "
            "Applications: artifact detection, signal characterization, quality control. "
            "High kurtosis often indicates noise, artifacts, or pathological activity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz.",
                },
                "window_size_sec": {
                    "type": ["number", "null"],
                    "description": "If provided, compute kurtosis in sliding windows of this duration (seconds).",
                    "default": None,
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "shannon_entropy_tool",
        "description": (
            "Compute Shannon entropy of signal amplitude distribution - quantifies randomness/complexity. "
            "Entropy measures: high entropy = random, unpredictable, complex signal; "
            "low entropy = regular, predictable, simple signal. "
            "Applications: distinguishing noise from neural activity, assessing signal complexity, "
            "quality control, comparing recording conditions. "
            "Fundamental information-theoretic measure adapted from Claude Shannon's work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n_bins": {
                    "type": "integer",
                    "description": "Number of bins for amplitude histogram used in entropy calculation.",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
]
