"""
chirp_generator.py

Generates a linear frequency-swept sine chirp for the acoustic sonar
system's transmit signal.
"""

import sys
import os

import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt

# Allow importing utils/audio_io.py when this script is run directly
# from within phase1_chirp_hardware/.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
import audio_io  # noqa: E402


def generate_chirp(f_start: float, f_end: float, duration_sec: float, sample_rate: int) -> np.ndarray:
    """
    Generates a linear frequency-swept sine chirp.

    Args:
        f_start: Starting frequency of the sweep, in Hz.
        f_end: Ending frequency of the sweep, in Hz.
        duration_sec: Duration of the chirp, in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        1D numpy array (float32) containing the chirp waveform,
        with amplitude normalized to the range -1.0 to 1.0.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # scipy.signal.chirp generates a swept-frequency cosine. method='linear'
    # means frequency increases linearly from f_start to f_end over t.
    waveform = chirp(t, f0=f_start, f1=f_end, t1=duration_sec, method="linear")

    return waveform.astype(np.float32)


if __name__ == "__main__":
    # --- Chirp parameters ---
    F_START = 18000       # Hz
    F_END = 22000         # Hz
    DURATION_SEC = 0.015  # 15ms
    SAMPLE_RATE = 48000   # Hz

    # NOTE ON SAMPLE RATE / NYQUIST:
    # The Nyquist–Shannon sampling theorem says a sample rate must be at
    # least 2x the highest frequency present in a signal to represent it
    # without aliasing (i.e. without higher frequencies being misread as
    # lower "ghost" frequencies). Our chirp goes up to 22kHz, so the sample
    # rate must be >= 44100 Hz. We use 48000 Hz here, which clears that
    # bar with margin and is also a standard, widely-supported audio
    # sample rate for consumer sound cards.

    chirp_signal = generate_chirp(F_START, F_END, DURATION_SEC, SAMPLE_RATE)

    output_path = os.path.join(os.path.dirname(__file__), "data", "chirp.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_io.save_wav(output_path, chirp_signal, SAMPLE_RATE)
    print(f"Saved chirp to {output_path}")

    # --- Plot the waveform ---
    t = np.linspace(0, DURATION_SEC, chirp_signal.shape[0], endpoint=False)
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1000, chirp_signal)  # time axis in milliseconds
    plt.title(f"Chirp waveform: {F_START/1000:.0f}kHz \u2192 {F_END/1000:.0f}kHz over {DURATION_SEC*1000:.0f}ms")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
