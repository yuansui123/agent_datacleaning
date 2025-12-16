"""
synthetic_neural_data.py

Generate realistic synthetic neural data for testing inspection pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Any, Optional, List, Tuple


# ============================================================
# 1. Main SEEG Data Generator
# ============================================================

def generate_seeg_data(
    duration: float = 5.0,
    sampling_rate: float = 2048.0,
    seed: Optional[int] = 42,
    include_ecg: bool = True,
    include_sharp_wave: Optional[List[float]] = None,
    include_movement: Optional[List[float]] = None,
    include_electrode_pop: Optional[List[float]] = None,
    movement_amplitude_uv: float = 400.0,
    movement_duration_sec: float = 0.5,
    electrode_pop_amplitude_uv: float = 600.0,
    electrode_pop_duration_sec: float = 0.01,
    line_noise_amplitude: float = 8.0
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Generate realistic SEEG (stereoelectroencephalography) data.

    Returns
    -------
    raw : np.ndarray
        Simulated SEEG signal in µV.
    metadata : dict
        Recording metadata.
    ground_truth : dict
        Ground truth structure containing physiological components,
        artifacts, and noise sources.
    """

    if seed is not None:
        np.random.seed(seed)

    t = np.arange(0, duration, 1.0 / sampling_rate)
    n_samples = len(t)

    # ============================================================
    # 1. PHYSIOLOGICAL BANDS (REAL µV AMPLITUDES)
    # ============================================================

    def band(freqs: List[float], base_amp_uv: float) -> np.ndarray:
        """Build a band as a sum of sinusoids with small amplitude jitter."""
        return sum(
            (base_amp_uv * (1 + 0.1 * np.random.randn())) *
            np.sin(2 * np.pi * f * t + 2 * np.pi * np.random.rand())
            for f in freqs
        )

    # Approximate physiological amplitudes (peak) in µV
    delta = band([1, 2, 3], 80.0)   # 0.5–4 Hz
    theta = band([5, 6, 7], 40.0)   # 4–8 Hz
    alpha = band([9, 10, 11], 30.0) # 8–13 Hz
    beta  = band([15, 20, 25], 15.0) # 13–30 Hz
    gamma = band([40, 60, 70], 5.0)  # 30–80 Hz

    # Broadband neural noise (band-limited white)
    white_noise = np.random.randn(n_samples) * 10.0  # RMS ~10 µV
    sos = signal.butter(4, [0.5, 250], btype="band", fs=sampling_rate, output="sos")
    broadband = signal.sosfilt(sos, white_noise)

    neural_signal = delta + theta + alpha + beta + gamma + broadband

    # ============================================================
    # 2. PATHOLOGICAL ACTIVITY: SHARP WAVES
    # ============================================================

    sharp_wave = np.zeros_like(t)
    sharp_wave_time: Optional[float] = None

    for sw_start in include_sharp_wave or []:
        sw_idx = int(sw_start * sampling_rate)
        sw_samples = int(0.07 * sampling_rate)  # 70 ms
        if sw_idx + sw_samples < n_samples:
            sw_time = np.linspace(0, 0.07, sw_samples)
            sharp_wave[sw_idx:sw_idx + sw_samples] = (
                -180.0 * np.exp(-sw_time * 25.0) * (1.0 - np.exp(-sw_time * 80.0))
            )
            sharp_wave_time = sw_start

    # ============================================================
    # 3. NOISE SOURCES
    # ============================================================

    # 60 Hz + harmonics at 120 and 180 Hz
    line_noise = line_noise_amplitude * np.sin(2 * np.pi * 60 * t + 0.3)
    line_noise += (line_noise_amplitude / 2.0) * np.sin(2 * np.pi * 120 * t + 0.1)
    line_noise += (line_noise_amplitude / 4.0) * np.sin(2 * np.pi * 180 * t + 0.5)

    # Electrode impedance noise (broadband)
    impedance_noise = np.random.randn(n_samples) * 5.0  # ~5 µV

    # ============================================================
    # 4. ARTIFACTS
    # ============================================================

    artifacts = np.zeros_like(t)
    artifact_times: Dict[str, Any] = {}

    # ---- ECG artifact (periodic) ----
    if include_ecg:
        ecg_period = 0.8  # ≈75 bpm
        ecg_times: List[float] = []
        for hb_time in np.arange(ecg_period, duration, ecg_period):
            hb_idx = int(hb_time * sampling_rate)
            hb_samples = int(0.15 * sampling_rate)  # 150 ms
            if hb_idx + hb_samples < n_samples:
                hb_t = np.linspace(0, 0.15, hb_samples)
                # QRS-like negative spike
                artifacts[hb_idx:hb_idx + hb_samples] += (
                    -120.0 * np.exp(-((hb_t - 0.05) ** 2) / 0.001)
                )
                ecg_times.append(hb_time)
        artifact_times["ecg"] = ecg_times

    # ---- Movement Artifact (configurable) ----
    if include_movement:
        movement_times: List[float] = []
        mov_samples = int(movement_duration_sec * sampling_rate)
        for mov_start in include_movement:
            mov_idx = int(mov_start * sampling_rate)
            if mov_idx + mov_samples < n_samples:
                mov_t = np.linspace(0, movement_duration_sec, mov_samples)
                artifacts[mov_idx:mov_idx + mov_samples] += (
                    -movement_amplitude_uv *
                    np.exp(-mov_t * 6.0) *
                    np.sin(2 * np.pi * 3.0 * mov_t)
                )
                movement_times.append(mov_start)
        artifact_times["movement"] = movement_times

    # ---- Electrode Pop (configurable) ----
    if include_electrode_pop:
        pop_times: List[float] = []
        pop_samples = int(electrode_pop_duration_sec * sampling_rate)
        for pop_start in include_electrode_pop:
            pop_idx = int(pop_start * sampling_rate)
            if pop_idx + pop_samples < n_samples:
                artifacts[pop_idx:pop_idx + pop_samples] = electrode_pop_amplitude_uv
                pop_times.append(pop_start)
        artifact_times["electrode_pop"] = pop_times

    # ---- Slow baseline drift ----
    baseline_drift = (
        30.0 * np.sin(2 * np.pi * 0.08 * t) +
        15.0 * np.sin(2 * np.pi * 0.03 * t)
    )

    # ============================================================
    # 5. FINAL RAW SIGNAL
    # ============================================================

    raw = (
        baseline_drift
        + neural_signal
        + sharp_wave
        + line_noise
        + impedance_noise
        + artifacts
    )

    # ============================================================
    # 6. METADATA AND GROUND TRUTH
    # ============================================================

    metadata: Dict[str, Any] = {
        "sampling_rate": sampling_rate,
        "units": "µV",
        "electrode_type": "depth electrode",
        "recording_type": "SEEG",
        "duration_sec": duration,
        "filter_applied": "none (raw)",
    }

    ground_truth: Dict[str, Any] = {
        "physiological_components": {
            "delta": {"freq_range": [0.5, 4], "amplitude_uv": 80},
            "theta": {"freq_range": [4, 8], "amplitude_uv": 40},
            "alpha": {"freq_range": [8, 13], "amplitude_uv": 30},
            "beta":  {"freq_range": [13, 30], "amplitude_uv": 15},
            "gamma": {"freq_range": [30, 80], "amplitude_uv": 5},
            "broadband_noise": {"amplitude_rms_uv": 10},
        },
        "pathological_activity": {},
        "artifacts": artifact_times,
        "noise_sources": {
            "line_noise_60hz": line_noise_amplitude,
            "line_noise_harmonics": [120, 180],
            "baseline_drift": {"freq_range": [0.03, 0.08], "amplitude_uv": 30},
            "impedance_noise": {"amplitude_uv": 5},
        }
    }

    if include_sharp_wave and sharp_wave_time is not None:
        ground_truth["pathological_activity"]["sharp_wave"] = {
            "time_sec": sharp_wave_time,
            "amplitude_uv": -180,
            "duration_ms": 70,
            "description": "interictal epileptiform discharge",
        }

    return raw, metadata, ground_truth


# ============================================================
# 2. SEPARATE PLOTTING FUNCTION
# ============================================================

def plot_seeg(
    raw: np.ndarray,
    t: np.ndarray,
    artifact_times: Dict[str, Any],
    sharp_wave_time: Optional[float] = None,
    movement_duration_sec: float = 0.5,
    show_artifact_lines: bool = True,
    show_legend: bool = True,
    save_plot: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot SEEG signal with optional artifact markers and zoom windows.

    Parameters
    ----------
    raw : np.ndarray
        SEEG waveform in µV.
    t : np.ndarray
        Time vector (seconds).
    artifact_times : dict
        'artifacts' field from ground_truth.
    sharp_wave_time : float or None
        Time of sharp wave (seconds), if any.
    movement_duration_sec : float
        Duration of movement artifact for zoom window selection.
    show_artifact_lines : bool
        Whether to overlay colored vertical lines for artifacts.
    show_legend : bool
        Whether to show a legend for artifact types.
    save_plot : str or None
        If provided, saves the figure to this path.
    show : bool
        If True, calls plt.show(); otherwise, closes the figure.
    """

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # -------- Main signal plot --------
    axes[0].plot(t, raw, "k", linewidth=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].set_title(f"Full {t[-1]:.2f}- SEEG Recording")
    axes[0].grid(True, alpha=0.3)

    legend_handles = []

    if show_artifact_lines:
        # Sharp wave
        if sharp_wave_time is not None:
            h = axes[0].axvline(
                sharp_wave_time, color="red", linestyle="--",
                alpha=0.7, label="Sharp Wave"
            )
            legend_handles.append(h)

        # Movement artifacts
        for mt in artifact_times.get("movement", []):
            h = axes[0].axvline(
                mt, color="orange", linestyle="--",
                alpha=0.7, label="Movement Artifact"
            )
            legend_handles.append(h)

        # Electrode pops
        for pt in artifact_times.get("electrode_pop", []):
            h = axes[0].axvline(
                pt, color="purple", linestyle="--",
                alpha=0.7, label="Electrode Pop"
            )
            legend_handles.append(h)

        # ECG artifacts (first few beats only)
        for hb in artifact_times.get("ecg", [])[:5]:
            h = axes[0].axvline(
                hb, color="blue", linestyle=":",
                alpha=0.4, label="ECG Artifact"
            )
            legend_handles.append(h)

    if show_legend and legend_handles:
        unique = {h.get_label(): h for h in legend_handles}
        axes[0].legend(
            unique.values(), unique.keys(),
            loc="upper right", frameon=True
        )

    # -------- Zoom Window 1: sharp wave or first second --------
    if sharp_wave_time is not None:
        mask1 = (t >= sharp_wave_time - 0.2) & (t <= sharp_wave_time + 0.3)
    else:
        mask1 = (t >= 0) & (t <= 1.0)

    axes[1].plot(t[mask1], raw[mask1], "k", linewidth=0.8)
    axes[1].set_title("Zoom Window 1 (Sharp Wave or First Second)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].grid(True, alpha=0.3)

    if show_artifact_lines and sharp_wave_time is not None:
        axes[1].axvline(
            sharp_wave_time, color="red", linestyle="--", alpha=0.7
        )

    # -------- Zoom Window 2: movement artifact or last second --------
    movement_list = artifact_times.get("movement", [])
    if movement_list:
        mov_t = movement_list[0]
        mask2 = (t >= mov_t - 0.2) & (t <= mov_t + movement_duration_sec + 0.2)
    else:
        mask2 = (t >= t[-1] - 1.0)

    axes[2].plot(t[mask2], raw[mask2], "k", linewidth=0.8)
    axes[2].set_title("Zoom Window 2 (Movement or Last Second)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude (µV)")
    axes[2].grid(True, alpha=0.3)

    if show_artifact_lines and movement_list:
        axes[2].axvline(
            mov_t, color="orange", linestyle="--", alpha=0.7
        )

    plt.tight_layout()

    if save_plot is not None:
        plt.savefig(save_plot, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 3. GROUND TRUTH PRINTING
# ============================================================

def print_ground_truth(ground_truth: Dict[str, Any]) -> None:
    """
    Pretty-print the ground truth information.
    """
    print("\n" + "=" * 60)
    print("GROUND TRUTH DESCRIPTION")
    print("=" * 60)

    print("\n📊 PHYSIOLOGICAL COMPONENTS:")
    for band, info in ground_truth["physiological_components"].items():
        if "freq_range" in info:
            print(
                f"  • {band.upper()}: {info['freq_range'][0]}–{info['freq_range'][1]} Hz, "
                f"~{info.get('amplitude_uv', info.get('amplitude_rms_uv', 'N/A'))} µV"
            )
        else:
            print(f"  • {band}: {info}")

    if ground_truth["pathological_activity"]:
        print("\n⚠️  PATHOLOGICAL ACTIVITY:")
        for event, info in ground_truth["pathological_activity"].items():
            print(
                f"  • {event}: t={info['time_sec']}s, {info['amplitude_uv']} µV, "
                f"{info['duration_ms']} ms - {info['description']}"
            )

    if ground_truth["artifacts"]:
        print("\n🚨 ARTIFACTS:")
        for artifact_type, times in ground_truth["artifacts"].items():
            if isinstance(times, list):
                preview = times[:5]
                print(f"  • {artifact_type}: at times {preview} ... "
                      f"({len(times)} occurrences)")
            else:
                print(f"  • {artifact_type}: t={times}s")

    print("\n📡 NOISE SOURCES:")
    noise = ground_truth["noise_sources"]
    print(f"  • 60 Hz line noise amplitude: {noise['line_noise_60hz']} µV")
    print(f"  • Harmonics: {noise['line_noise_harmonics']} Hz")
    print(f"  • Baseline drift amplitude: {noise['baseline_drift']['amplitude_uv']} µV")
    print(f"  • Impedance noise amplitude: {noise['impedance_noise']['amplitude_uv']} µV")

    print("=" * 60 + "\n")
