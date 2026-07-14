"""
record_playback_test.py

Phase 1 hardware validation: generates a chirp, plays it through the
speaker while simultaneously recording from the mic, saves both signals
as .wav files, and plots them stacked for visual comparison.
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# Allow importing from utils/ and this phase's own chirp_generator.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.dirname(__file__))
import audio_io  # noqa: E402
from chirp_generator import generate_chirp  # noqa: E402


if __name__ == "__main__":
    # --- Parameters ---
    F_START = 18000        # Hz
    F_END = 22000          # Hz
    CHIRP_DURATION_SEC = 0.015   # 15ms
    RECORD_DURATION_SEC = 0.2    # 200ms
    SAMPLE_RATE = 48000     # Hz

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    # --- Generate the chirp ---
    emitted = generate_chirp(F_START, F_END, CHIRP_DURATION_SEC, SAMPLE_RATE)

    # --- Play it and record simultaneously ---
    print("Playing chirp and recording... stay quiet for a clean test.")
    recorded = audio_io.play_and_record(emitted, SAMPLE_RATE, RECORD_DURATION_SEC)
    print("Done.")

    # --- Save both signals ---
    emitted_path = os.path.join(data_dir, "emitted_chirp.wav")
    recorded_path = os.path.join(data_dir, "recorded_echo.wav")
    audio_io.save_wav(emitted_path, emitted, SAMPLE_RATE)
    audio_io.save_wav(recorded_path, recorded, SAMPLE_RATE)
    print(f"Saved emitted chirp to {emitted_path}")
    print(f"Saved recorded audio to {recorded_path}")

    # --- Plot both waveforms stacked for comparison ---
    t_emitted = np.linspace(0, CHIRP_DURATION_SEC, emitted.shape[0], endpoint=False) * 1000
    t_recorded = np.linspace(0, RECORD_DURATION_SEC, recorded.shape[0], endpoint=False) * 1000

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axes[0].plot(t_emitted, emitted, color="tab:blue")
    axes[0].set_title(f"Emitted chirp ({F_START/1000:.0f}\u2013{F_END/1000:.0f}kHz, {CHIRP_DURATION_SEC*1000:.0f}ms)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    axes[1].plot(t_recorded, recorded, color="tab:orange")
    axes[1].set_title(f"Recorded audio ({RECORD_DURATION_SEC*1000:.0f}ms window)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
