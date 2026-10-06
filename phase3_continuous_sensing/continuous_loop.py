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
from chirp_generator import generate_chirp, generate_varied_chirp  # noqa: E402
from config_loader import load_sonar_config  # noqa: E402
from distance_calc import delay_to_distance  # noqa: E402
from rolling_buffer import RollingBuffer  # noqa: E402
from motion_detector import compare_profiles  # noqa: E402


def compute_correlation_profile(
    emitted: np.ndarray,
    recorded: np.ndarray,
    sample_rate: int,
    min_delay_sec: float,
    latency_s: float = 0.0,
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
        latency_s: System audio latency in seconds to subtract from lags.

    Returns:
        Tuple of (lag_times_sec, correlation_mag), both 1D numpy arrays
        of equal length, restricted to lags >= min_delay_sec.
    """
    emitted = np.asarray(emitted, dtype=np.float64)
    recorded = np.asarray(recorded, dtype=np.float64)

    correlation = correlate(recorded, emitted, mode="full")
    lags = correlation_lags(recorded.shape[0], emitted.shape[0], mode="full")
    lag_times_sec = (lags / sample_rate) - latency_s
    correlation_mag = np.abs(correlation)

    valid_mask = lag_times_sec >= min_delay_sec
    return lag_times_sec[valid_mask], correlation_mag[valid_mask]


if __name__ == "__main__":
    # --- Load calibrated band (falls back to 4-8 kHz if not yet calibrated) ---
    _cfg = load_sonar_config()
    F_START             = _cfg["f_start"]   # Hz
    F_END               = _cfg["f_end"]     # Hz
    SAMPLE_RATE         = _cfg["sample_rate"]
    LATENCY_S           = _cfg.get("latency_s", 0.0)

    # A 75 ms chirp sweeping 4 kHz gives time-bandwidth product ~300,
    # yielding sub-centimetre range resolution via matched-filter correlation.
    CHIRP_DURATION_SEC  = 0.075   # 75 ms  (was 15 ms at ultrasonic freqs)

    # Record long enough to catch echoes from objects up to ~8 m away
    # (round-trip at 343 m/s ≈ 47 ms) with comfortable headroom.
    RECORD_DURATION_SEC = 0.4     # 400 ms

    # Min delay ignores direct speaker->mic bleed. 3 ms ≈ 51 cm min range.
    MIN_DELAY_SEC       = 0.003

    # Cycle at least as long as the record window plus a gap for processing.
    CYCLE_INTERVAL_SEC  = 0.6     # 600 ms
    BUFFER_SIZE         = 5
    MOTION_THRESHOLD    = 0.15

    print(f"Chirp band : {F_START/1000:.1f}-{F_END/1000:.1f} kHz  "
          f"| duration: {CHIRP_DURATION_SEC*1000:.0f} ms  "
          f"| record: {RECORD_DURATION_SEC*1000:.0f} ms")

    buffer = RollingBuffer(max_size=BUFFER_SIZE)
    ping_index = 0
    motion_history = []

    print("Starting continuous sensing loop. Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle_start = time.time()

            # Regenerate chirp each ping — small variation defeats OS AEC.
            emitted = generate_varied_chirp(
                F_START, F_END, CHIRP_DURATION_SEC, SAMPLE_RATE, ping_index
            )
            ping_index += 1

            recorded = audio_io.play_and_record(emitted, SAMPLE_RATE, RECORD_DURATION_SEC)
            
            distance_m = None
            # 1. Recording gate — refuse to process silence
            rms = np.sqrt(np.mean(recorded**2))
            if rms < 1e-4:
                print(f"[sonar] no mic signal (rms={rms:.2e}) — check input device/mic permission")
            else:
                lag_times_sec, profile = compute_correlation_profile(
                    emitted, recorded, SAMPLE_RATE, MIN_DELAY_SEC, LATENCY_S
                )

                # 2. Peak quality gate — only trust a correlation peak that stands out
                peak_index = int(np.argmax(profile))
                peak = profile[peak_index]
                mask = np.ones_like(profile, bool)
                mask[max(0, peak_index-50):peak_index+50] = False
                floor = np.median(profile[mask]) + 1e-12
                
                if peak / floor >= 8.0:
                    # 3. Range clamp — physics limit for the room
                    dist = delay_to_distance(lag_times_sec[peak_index])
                    if dist <= 5.0:
                        distance_m = dist

            raw_motion = False
            difference_score = 0.0
            if distance_m is not None:
                if len(buffer) > 0:
                    reference_profile = buffer.average()
                    raw_motion, difference_score = compare_profiles(reference_profile, profile, MOTION_THRESHOLD)
                buffer.add(profile)

            motion_history.append(raw_motion)
            if len(motion_history) > 5:
                motion_history.pop(0)

            # Require 3 of 5 valid motion triggers
            motion_detected = sum(motion_history) >= 3

            if distance_m is None:
                print("No target detected (weak echo or out of range).")
            elif motion_detected:
                print(f"Motion detected! (difference_score={difference_score:.3f})")
            else:
                print(f"No motion. Distance: {distance_m*100:.1f} cm (difference_score={difference_score:.3f})")

            # Sleep for whatever's left of the cycle interval, accounting
            # for however long the sensing + comparison work just took.
            elapsed = time.time() - cycle_start
            sleep_time = max(0.0, CYCLE_INTERVAL_SEC - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped.")
