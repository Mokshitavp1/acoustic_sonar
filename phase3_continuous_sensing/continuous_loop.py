"""
continuous_loop.py

Phase 3: runs a continuous chirp-emit-and-record cycle (default every
300ms), keeping a rolling buffer of cross-correlation profiles and
comparing each new cycle against the buffer's smoothed average to flag
motion. When no motion is detected, prints the current estimated
distance instead.

Run with Ctrl+C to stop.
"""

import sys
import os
import time

import numpy as np
from scipy.signal import correlate, correlation_lags

# Allow importing from utils/, phase1_chirp_hardware/, phase2_cross_correlation/,
# and this phase's own modules.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "phase1_chirp_hardware"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "phase2_cross_correlation"))
sys.path.append(os.path.dirname(__file__))

import audio_io  # noqa: E402
from chirp_generator import generate_chirp  # noqa: E402
from distance_calc import delay_to_distance  # noqa: E402
from rolling_buffer import RollingBuffer  # noqa: E402
from motion_detector import compare_profiles  # noqa: E402


def compute_correlation_profile(
    emitted: np.ndarray, recorded: np.ndarray, sample_rate: int, min_delay_sec: float
):
    """
    Computes the full cross-correlation magnitude profile between
    `recorded` and `emitted`, restricted to lags at or beyond
    `min_delay_sec` (to exclude direct speaker-to-mic bleed).

    This is the same underlying computation as find_echo_delay() in
    cross_correlate.py, but returns the whole profile (and its
    corresponding lag times) instead of just the strongest peak — so it
    can be stored in a RollingBuffer and compared frame-to-frame for
    motion detection.

    Args:
        emitted: 1D numpy array of the known emitted chirp waveform.
        recorded: 1D numpy array of the recorded audio (same sample rate).
        sample_rate: Sample rate in Hz, shared by both signals.
        min_delay_sec: Minimum time delay (seconds) to include — anything
            before this is direct bleed, not a real reflection.

    Returns:
        Tuple of (lag_times_sec, correlation_mag), both 1D numpy arrays
        of equal length, restricted to lags >= min_delay_sec.
    """
    emitted = np.asarray(emitted, dtype=np.float64)
    recorded = np.asarray(recorded, dtype=np.float64)

    correlation = correlate(recorded, emitted, mode="full")
    lags = correlation_lags(recorded.shape[0], emitted.shape[0], mode="full")
    lag_times_sec = lags / sample_rate
    correlation_mag = np.abs(correlation)

    valid_mask = lag_times_sec >= min_delay_sec
    return lag_times_sec[valid_mask], correlation_mag[valid_mask]


if __name__ == "__main__":
    # --- Parameters ---
    F_START = 18000              # Hz
    F_END = 22000                 # Hz
    CHIRP_DURATION_SEC = 0.015    # 15ms
    RECORD_DURATION_SEC = 0.05    # 50ms per cycle
    SAMPLE_RATE = 48000            # Hz
    MIN_DELAY_SEC = 0.003          # ~51cm minimum range, filters direct bleed
    CYCLE_INTERVAL_SEC = 0.3       # run a sensing cycle every 300ms
    BUFFER_SIZE = 5                # rolling buffer of last 5 cycles
    MOTION_THRESHOLD = 0.15        # tune this based on real-world testing

    emitted = generate_chirp(F_START, F_END, CHIRP_DURATION_SEC, SAMPLE_RATE)
    buffer = RollingBuffer(max_size=BUFFER_SIZE)

    print("Starting continuous sensing loop. Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle_start = time.time()

            recorded = audio_io.play_and_record(emitted, SAMPLE_RATE, RECORD_DURATION_SEC)
            lag_times_sec, profile = compute_correlation_profile(
                emitted, recorded, SAMPLE_RATE, MIN_DELAY_SEC
            )

            if len(buffer) > 0:
                reference_profile = buffer.average()
                motion_detected, difference_score = compare_profiles(
                    reference_profile, profile, MOTION_THRESHOLD
                )

                if motion_detected:
                    print(f"Motion detected! (difference_score={difference_score:.3f})")
                else:
                    peak_index = np.argmax(profile)
                    distance_m = delay_to_distance(lag_times_sec[peak_index])
                    print(f"No motion. Distance: {distance_m*100:.1f} cm (difference_score={difference_score:.3f})")
            else:
                print("Warming up rolling buffer...")

            buffer.add(profile)

            # Sleep for whatever's left of the cycle interval, accounting
            # for however long the sensing + comparison work just took.
            elapsed = time.time() - cycle_start
            sleep_time = max(0.0, CYCLE_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped.")
