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
            "Compute a time-frequency spectrogram and optionally plot it. "
            "This tool now returns ONLY scalar spectrogram summaries "
            "(max power, dominant frequency) and image paths, not the spectrogram matrix."
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
                "make_plot": {
                    "type": "boolean",
                    "description": "If true, save a PNG image of the spectrogram.",
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
]
