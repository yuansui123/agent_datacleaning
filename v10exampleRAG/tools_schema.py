# tools_schema.py

from __future__ import annotations

# One schema per plotting function. These names must match tools.TOOL_REGISTRY keys.

PLOT_TIME_SERIES_SCHEMA = {
    "name": "plot_time_series",
    "description": (
        "Plot raw neural time series for one or more channels. "
        "Use this when you need to inspect the waveform over time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channels": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "List of 0-based channel indices to plot. "
                    "If omitted, uses all channels."
                ),
            },
            "time_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Time window [start_sec, end_sec] within the recording. "
                    "If omitted, uses full recording."
                ),
            },
            "downsample": {
                "type": "integer",
                "description": (
                    "Downsample factor (e.g., 10 keeps every 10th sample) "
                    "to make plots less dense."
                ),
            },
            "aggregate": {
                "type": "string",
                "enum": ["none", "mean", "median"],
                "description": (
                    "If plotting multiple channels, you can aggregate them "
                    "by mean or median instead of stacking."
                ),
                "default": "none",
            },
            "output_dir": {
                "type": "string",
                "description": (
                    "Directory to save the generated PNG. "
                    "If omitted, defaults to './plots'."
                ),
            },
        },
        "required": [],
    },
}

PLOT_PSD_SCHEMA = {
    "name": "plot_psd",
    "description": (
        "Plot the power spectral density (PSD) of selected channels. "
        "Use this when you care about oscillations / frequency content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channels": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Channels to compute PSD for.",
            },
            "time_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Time window [start_sec, end_sec] in seconds.",
            },
            "freq_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Frequency range [f_low, f_high] in Hz to keep. "
                    "If omitted, uses the full computed band."
                ),
            },
            "n_fft": {
                "type": "integer",
                "description": "FFT length / segment size for Welch's method.",
            },
            "aggregate": {
                "type": "string",
                "enum": ["none", "mean", "median"],
                "description": (
                    "If multiple channels are selected, you can aggregate "
                    "their PSDs by mean or median."
                ),
                "default": "none",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save the PSD PNG.",
            },
        },
        "required": [],
    },
}

PLOT_SPECTROGRAM_SCHEMA = {
    "name": "plot_spectrogram",
    "description": (
        "Plot a spectrogram (time-frequency representation) for a single channel. "
        "Use this when you need how frequency content evolves over time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "channels": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Channel indices; only the first one in the list will be used."
                ),
            },
            "time_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Time window [start_sec, end_sec] in seconds.",
            },
            "freq_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Frequency range [f_low, f_high] in Hz.",
            },
            "n_fft": {
                "type": "integer",
                "description": "FFT window length for the spectrogram.",
            },
            "output_dir": {
                "type": "string",
                "description": "Directory to save the spectrogram PNG.",
            },
        },
        "required": [],
    },
}

PLOT_POWER_DENSITY_MATRIX_HILBERT = {
    "name": "power_density_matrix_hilbert",
    "description": (
        "Compute a time–frequency power density matrix using bandpass filtering "
        "and Hilbert envelope extraction, then visualize it as a heatmap with "
        "band-wise temporal traces. Useful for detecting transient bursts, "
        "oscillations, and powerline artifacts around 60 Hz. Operates on the "
        "neural data and sampling_rate automatically provided to the tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "output_dir": {
                "type": "string",
                "description": (
                    "Optional directory where the figure will be saved. "
                    "If omitted, a default directory will be created automatically. "
                    "The final filename is generated via the shared _save_fig helper."
                ),
            },
            "freq_bands": {
                "type": "object",
                "description": (
                    "Optional dictionary mapping band names to [low_freq, high_freq] "
                    "in Hz. If omitted, a comprehensive default set of bands is used "
                    "(delta, theta, alpha, beta, gamma, 60 Hz powerline, ripple, etc.)."
                ),
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                }
            },
            "clip_seconds": {
                "type": "number",
                "default": 0.5,
                "description": (
                    "Seconds to remove from each edge before analysis to eliminate "
                    "filter transients and ringing artifacts."
                ),
            },
            "filter_order": {
                "type": "integer",
                "default": 4,
                "description": (
                    "Order of the Butterworth bandpass filter used prior to Hilbert envelope extraction."
                ),
            },
            "envelope_smoothing_sigma": {
                "type": "number",
                "default": 10.0,
                "description": (
                    "Gaussian smoothing (sigma in samples) applied to the envelope "
                    "to reduce noise and improve clarity of slow trends."
                ),
            },
            "normalize_mode": {
                "type": "string",
                "enum": ["global", "per_band", "none"],
                "default": "global",
                "description": (
                    "Normalization strategy for the power density matrix:\n"
                    "- 'global': Normalize all bands together.\n"
                    "- 'per_band': Normalize each band independently.\n"
                    "- 'none': Use raw, unnormalized envelope magnitudes."
                ),
            },
            "label": {
                "type": "string",
                "default": "Power Density Matrix (Hilbert)",
                "description": "Title label applied to the resulting plot."
            }
        },
        "required": []   # all optional, data+sampling_rate injected automatically
    }
}

PLOT_SIMPLE_LINE_NOISE = {
    "name": "plot_simple_line_noise",
    "description": (
        "Compute a basic Welch PSD and visualize the presence of line noise at "
        "50 Hz or 60 Hz by drawing a red vertical line. Only one figure is produced."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "output_dir": {
                "type": "string",
                "description": (
                    "Optional directory where the PNG figure will be saved. "
                    "If omitted, the tool uses a default directory."
                ),
            },
            "line_freq": {
                "type": "number",
                "default": 60.0,
                "description": "Line noise frequency in Hz (common values: 50 or 60).",
            },
            "welch_nperseg": {
                "type": "integer",
                "description": (
                    "Optional Welch window size. If omitted, defaults to "
                    "min(len(data), 2 * sampling_rate)."
                ),
            },
            "label": {
                "type": "string",
                "default": "Simple Line Noise Detection",
                "description": "Title/label for the plot.",
            },
        },
        "required": [],
    },
}

TOOLS_SCHEMAS = [
    PLOT_TIME_SERIES_SCHEMA,
    PLOT_PSD_SCHEMA,
    PLOT_SPECTROGRAM_SCHEMA,
    PLOT_POWER_DENSITY_MATRIX_HILBERT,
    PLOT_SIMPLE_LINE_NOISE,
]
