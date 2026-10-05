"""
validation_test.py

Phase 2 validation: emits a chirp, records the echo off whatever object
is in front of the laptop, estimates the distance to it, and compares
that estimate against a tape-measure reading you enter manually.
"""

import sys
import os

# Allow importing from utils/, phase1_chirp_hardware/, and this phase's
# own modules.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "phase1_chirp_hardware"))
sys.path.append(os.path.dirname(__file__))

import audio_io  # noqa: E402
from chirp_generator import generate_chirp  # noqa: E402
from config_loader import load_sonar_config  # noqa: E402
from cross_correlate import find_echo_delay  # noqa: E402
from distance_calc import delay_to_distance  # noqa: E402


if __name__ == "__main__":
    # --- Load calibrated band (falls back to 4-8 kHz if not yet calibrated) ---
    _cfg = load_sonar_config()
    F_START             = _cfg["f_start"]    # Hz
    F_END               = _cfg["f_end"]      # Hz
    SAMPLE_RATE         = _cfg["sample_rate"]

    # 75 ms chirp: long enough for a high time-bandwidth product (good SNR)
    # while still being a quick, barely-audible whoosh.
    CHIRP_DURATION_SEC  = 0.075   # 75 ms

    # 400 ms captures echoes from objects up to ~68 m away at 343 m/s.
    RECORD_DURATION_SEC = 0.4     # 400 ms

    # Ignore the first 3 ms (direct speaker-to-mic bleed, ~51 cm bleed zone).
    MIN_DELAY_SEC       = 0.003

    print(f"Using chirp: {F_START/1000:.1f}-{F_END/1000:.1f} kHz  "
          f"{CHIRP_DURATION_SEC*1000:.0f} ms  ({_cfg.get('note', '')})")

    print("Point your laptop at a wall or object you can measure with a tape measure.")
    input("Press Enter when ready to emit the chirp and record the echo...")

    # --- Emit chirp and record echo ---
    emitted = generate_chirp(F_START, F_END, CHIRP_DURATION_SEC, SAMPLE_RATE)
    recorded = audio_io.play_and_record(emitted, SAMPLE_RATE, RECORD_DURATION_SEC)

    # --- Estimate distance ---
    echo_delay_sec = find_echo_delay(
        emitted, recorded, SAMPLE_RATE, min_delay_sec=MIN_DELAY_SEC, debug_plot=True
    )
    estimated_distance_m = delay_to_distance(echo_delay_sec)

    print(f"\nEstimated distance: {estimated_distance_m:.3f} m ({estimated_distance_m*100:.1f} cm)")

    # --- Compare against ground truth ---
    while True:
        raw_input_value = input(
            "Enter the distance you measured with a tape measure, in cm: "
        ).strip()
        try:
            measured_distance_cm = float(raw_input_value)
            break
        except ValueError:
            print("Please enter a number, e.g. 55.5")

    measured_distance_m = measured_distance_cm / 100
    error_m = estimated_distance_m - measured_distance_m

    print("\n--- Validation Result ---")
    print(f"Sonar estimate:   {estimated_distance_m*100:.1f} cm")
    print(f"Tape measurement: {measured_distance_cm:.1f} cm")
    print(f"Error:            {error_m*100:+.1f} cm")
