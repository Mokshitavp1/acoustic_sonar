"""
freq_response_test.py

Phase 1 hardware validation: sweeps chirp start frequency from 15kHz to
22kHz in five steps, records the mic's response to each, and plots the
FFT magnitude spectra together so you can see which frequency range your
laptop's mic actually picks up cleanly (vs. rolling off).
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


def compute_fft_magnitude(signal: np.ndarray, sample_rate: int):
    """
    Computes the single-sided FFT magnitude spectrum of a real-valued signal.

    Returns:
        freqs: 1D array of frequency bins (Hz), 0 to sample_rate/2.
        magnitude: 1D array of magnitude values for each frequency bin.
    """
    n = signal.shape[0]
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitude = np.abs(fft_vals)
    return freqs, magnitude


if __name__ == "__main__":
    # --- Sweep parameters ---
    START_FREQS = np.linspace(15000, 22000, 5)  # 15kHz, 16.75kHz, 18.5kHz, 20.25kHz, 22kHz
    SWEEP_WIDTH = 2000               # each chirp sweeps start_freq -> start_freq + 2kHz
    CHIRP_DURATION_SEC = 0.015       # 15ms
    RECORD_DURATION_SEC = 0.2        # 200ms
    SAMPLE_RATE = 48000               # Hz

    plt.figure(figsize=(10, 6))

    for f_start in START_FREQS:
        f_end = f_start + SWEEP_WIDTH
        print(f"Testing chirp {f_start/1000:.1f}kHz -> {f_end/1000:.1f}kHz ...")

        chirp_signal = generate_chirp(f_start, f_end, CHIRP_DURATION_SEC, SAMPLE_RATE)
        recorded = audio_io.play_and_record(chirp_signal, SAMPLE_RATE, RECORD_DURATION_SEC)

        freqs, magnitude = compute_fft_magnitude(recorded, SAMPLE_RATE)

        # Only plot up to 24kHz (just above our highest test frequency) to
        # keep the graph readable.
        mask = freqs <= 24000
        plt.plot(
            freqs[mask] / 1000,
            magnitude[mask],
            label=f"{f_start/1000:.1f}\u2013{f_end/1000:.1f}kHz chirp",
        )

    print("Done. Close the plot window when finished inspecting.")

    plt.title("Mic FFT response across chirp frequency sweep")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
