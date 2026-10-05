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


# Pre-computed variation table used by generate_varied_chirp.
# Each entry is (freq_offset_hz, initial_phase_rad).
# The offsets are small enough to stay within any calibrated band
# (±60 Hz on a 4 kHz bandwidth is <2 % shift), but large enough that
# a statistical echo-canceller cannot lock onto a repeating pattern.
_VARIATION_TABLE: list[tuple[float, float]] = [
    (   0.0,  0.000),   # ping 0  — baseline
    ( +55.0,  0.785),   # ping 1  — +55 Hz, π/4 phase
    ( -55.0,  1.571),   # ping 2  — -55 Hz, π/2 phase
    ( +30.0,  3.142),   # ping 3  — +30 Hz, π   phase
    ( -30.0,  2.356),   # ping 4  — -30 Hz, 3π/4 phase
    ( +60.0,  4.712),   # ping 5  — +60 Hz, 3π/2 phase
    ( -60.0,  0.524),   # ping 6  — -60 Hz, π/6 phase
    ( +15.0,  5.497),   # ping 7  — +15 Hz, 7π/4 phase
]


def generate_varied_chirp(
    f_start: float,
    f_end: float,
    duration_sec: float,
    sample_rate: int,
    ping_index: int,
) -> np.ndarray:
    """
    Generates a slightly varied chirp for each ping to defeat OS echo
    cancellers.

    How it works
    ------------
    OS acoustic echo cancellation (AEC) builds a statistical model of the
    loudspeaker signal and subtracts it from the mic input.  If every ping
    is *identical*, the AEC can lock on and cancel the chirp itself (and
    therefore any echoes of it), leaving the sonar blind.

    By rotating through a look-up table of small (±60 Hz, varied phase)
    variations, successive pings look different to the AEC's statistical
    model while remaining essentially identical for ranging purposes:

    * The frequency shift is < 2 % of the chirp bandwidth — the
      cross-correlation peak moves by < 1 sample, so range accuracy is
      unaffected.
    * The phase offset has *zero* effect on cross-correlation magnitude;
      it only changes the waveform's appearance in time domain.
    * The caller MUST use the returned array as both the emitted signal
      (played through the speaker) and the reference template (passed to
      compute_correlation_profile).  Never pre-compute one chirp and reuse
      it while playing a different variant.

    Parameters
    ----------
    f_start, f_end, duration_sec, sample_rate : same as generate_chirp.
    ping_index : monotonically increasing counter (0, 1, 2, …).
        Wraps automatically within the variation table.

    Returns
    -------
    1D float32 numpy array — the chirp waveform for this ping.
    """
    freq_offset, phase_offset = _VARIATION_TABLE[ping_index % len(_VARIATION_TABLE)]

    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Apply the frequency offset to both endpoints so the sweep rate
    # (bandwidth) stays constant — only the absolute position shifts.
    waveform = chirp(
        t,
        f0=f_start + freq_offset,
        f1=f_end + freq_offset,
        t1=duration_sec,
        method="linear",
        phi=np.degrees(phase_offset),  # scipy takes degrees
    )

    return waveform.astype(np.float32)


if __name__ == "__main__":
    # --- Chirp parameters ---
    F_START = 4000        # Hz  (audible; every laptop speaker+mic handles this)
    F_END = 8000          # Hz
    DURATION_SEC = 0.075  # 75 ms  — long enough for a high time-bandwidth product
    SAMPLE_RATE = 48000   # Hz

    # NOTE ON SAMPLE RATE / NYQUIST:
    # The Nyquist-Shannon theorem requires sample_rate >= 2 x f_max.
    # Our chirp tops out at 8 kHz, so anything above 16 kHz works;
    # 48 kHz is the standard consumer audio rate and gives plenty of headroom.

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
