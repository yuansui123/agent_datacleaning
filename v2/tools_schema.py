# tool_schemas.py or inside your main module

TOOLS_SCHEMAS = [
    {
        "name": "compute_fft",
        "description": (
            "Compute the magnitude FFT of a 1D neural signal. "
            "Use this to detect line noise, dominant frequencies, or broadband activity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {
                    "type": "number",
                    "description": "Sampling rate in Hz (e.g., 30000)."
                },
                "max_freq": {
                    "type": "number",
                    "description": "Optional maximum frequency to show on the plot (Hz).",
                },
                "make_plot": {
                    "type": "boolean",
                    "description": "Whether to generate and save a plot.",
                    "default": True,
                },
                "label": {
                    "type": "string",
                    "description": "Title for the FFT plot.",
                    "default": "Fourier Transform",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "compute_psd",
        "description": (
            "Compute the power spectral density (Welch) of the signal to inspect "
            "power across frequencies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {"type": "number"},
                "nperseg": {
                    "type": "integer",
                    "description": "Segment length for Welch PSD.",
                    "default": 2048,
                },
                "make_plot": {
                    "type": "boolean",
                    "default": True,
                },
                "label": {
                    "type": "string",
                    "default": "Power Spectral Density",
                },
            },
            "required": ["sampling_rate"],
        },
    },
    {
        "name": "compute_spectrogram_tool",
        "description": (
            "Compute a time-frequency spectrogram of the signal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {"type": "number"},
                "nperseg": {
                    "type": "integer",
                    "default": 256,
                    "description": "Window length for spectrogram."
                },
                "noverlap": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Number of overlapping samples between segments."
                },
                "make_plot": {
                    "type": "boolean",
                    "default": True,
                },
                "label": {
                    "type": "string",
                    "default": "Spectrogram (dB)",
                },
            },
            "required": ["sampling_rate"],
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
        "description": (
            "Compute a simple SNR estimate of the signal: mean(signal^2)/var(signal-mean(signal))."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "compute_channel_stats_tool",
        "description": "Compute mean and standard deviation of the signal.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "metadata_snapshot_tool",
        "description": (
            "Return the recording metadata so numeric/domain interpreters can reason about it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # The LLM does not pass metadata; executor injects it.
            },
            "required": [],
        },
    },
    {
        "name": "apply_notch_filter_tool",
        "description": (
            "Apply a notch filter at a given frequency (e.g., 50/60 Hz) to remove line noise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sampling_rate": {"type": "number"},
                "notch_freq": {
                    "type": "number",
                    "default": 60.0,
                    "description": "Notch frequency in Hz."
                },
                "quality_factor": {
                    "type": "number",
                    "default": 30.0,
                    "description": "Quality factor of the notch filter."
                },
                "make_plot": {
                    "type": "boolean",
                    "default": False,
                },
                "label": {
                    "type": "string",
                    "default": "Notch-filtered Signal (time domain)",
                },
            },
            "required": ["sampling_rate"],
        },
    },
]
