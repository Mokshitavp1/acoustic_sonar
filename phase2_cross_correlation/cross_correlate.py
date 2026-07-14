"""
cross_correlate.py

Phase 2 core algorithm: finds the time delay of the strongest echo in a
recorded signal by cross-correlating it against the known emitted chirp,
ignoring the direct speaker-to-mic bleed at the very start of the
recording.
"""

import os

import numpy as np
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt


def find_echo_delay(
    emitted: np.ndarray,
    recorded: np.ndarray,
    sample_rate: int,
    min_delay_sec: float,
    debug_plot: bool = True,
) -> float:
    """
    Cross-correlates `recorded` against `emitted` to find the time delay
    of the strongest echo peak, ignoring any peak occurring before
    `min_delay_sec` (the direct speaker-to-mic bleed path, which is not
    a real reflection).

    Args:
        emitted: 1D numpy array of the known emitted chirp waveform.
        recorded: 1D numpy array of the recorded audio (same sample rate).
        sample_rate: Sample rate in Hz, shared by both signals.
        min_delay_sec: Minimum time delay (seconds) to consider — any
            correlation peak occurring before this is assumed to be
            direct bleed rather than a real echo, and is ignored.
        debug_plot: If True, shows a plot of the cross-correlation output
            with the detected peak marked.

    Returns:
        Estimated time delay, in seconds, of the strongest echo peak.

    Raises:
        ValueError: if no correlation peak is found at or beyond
            `min_delay_sec` (e.g. min_delay_sec is longer than the
            recording itself).
    """
    emitted = np.asarray(emitted, dtype=np.float64)
    recorded = np.asarray(recorded, dtype=np.float64)

    # Full cross-correlation: for each possible lag, how well does a
    # shifted copy of `emitted` line up with `recorded`.
    correlation = correlate(recorded, emitted, mode="full")

    # correlation_lags gives the lag (in samples) corresponding to each
    # entry in `correlation`. A positive lag means `recorded` is delayed
    # relative to `emitted` — i.e. it's an echo that arrived later.
    lags = correlation_lags(recorded.shape[0], emitted.shape[0], mode="full")
    lag_times_sec = lags / sample_rate

    # Use magnitude so we're robust to any polarity flip in the echo.
    correlation_mag = np.abs(correlation)

    # Mask out anything before min_delay_sec (direct bleed) and anything
    # at negative delay (not physically meaningful for an echo).
    valid_mask = lag_times_sec >= min_delay_sec
    if not np.any(valid_mask):
        raise ValueError(
            f"No correlation samples found at or beyond min_delay_sec={min_delay_sec}s. "
            "Check that the recording is long enough and min_delay_sec isn't too large."
        )

    valid_indices = np.where(valid_mask)[0]
    peak_index_within_valid = np.argmax(correlation_mag[valid_indices])
    peak_index = valid_indices[peak_index_within_valid]
    peak_delay_sec = lag_times_sec[peak_index]

    if debug_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(lag_times_sec * 1000, correlation_mag, label="Cross-correlation magnitude")
        plt.axvline(
            peak_delay_sec * 1000,
            color="red",
            linestyle="--",
            label=f"Detected peak: {peak_delay_sec*1000:.2f}ms",
        )
        plt.axvline(
            min_delay_sec * 1000,
            color="gray",
            linestyle=":",
            label=f"min_delay_sec cutoff: {min_delay_sec*1000:.2f}ms",
        )
        plt.title("Cross-correlation: recorded vs. emitted chirp")
        plt.xlabel("Lag (ms)")
        plt.ylabel("Correlation magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return peak_delay_sec


if __name__ == "__main__":
    # Quick self-test with synthetic data: build a fake "recording" that's
    # silence, direct bleed, then a delayed+attenuated copy of the chirp
    # (simulating a real echo), and confirm find_echo_delay recovers the
    # correct delay while ignoring the bleed.
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "phase1_chirp_hardware"))
    from chirp_generator import generate_chirp  # noqa: E402

    SAMPLE_RATE = 48000
    chirp_signal = generate_chirp(18000, 22000, 0.015, SAMPLE_RATE)

    bleed_delay_samples = int(0.001 * SAMPLE_RATE)   # 1ms direct bleed
    echo_delay_samples = int(0.006 * SAMPLE_RATE)     # 6ms simulated echo
    total_len = echo_delay_samples + chirp_signal.shape[0] + 500

    fake_recording = np.zeros(total_len)
    fake_recording[bleed_delay_samples: bleed_delay_samples + chirp_signal.shape[0]] += 0.3 * chirp_signal
    fake_recording[echo_delay_samples: echo_delay_samples + chirp_signal.shape[0]] += 0.6 * chirp_signal

    detected_delay = find_echo_delay(
        chirp_signal, fake_recording, SAMPLE_RATE, min_delay_sec=0.003
    )
    print(f"Expected echo delay: 6.00ms, Detected: {detected_delay*1000:.2f}ms")
