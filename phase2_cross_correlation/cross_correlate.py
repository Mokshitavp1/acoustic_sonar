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
    max_range_m: float = 5.0,
    debug_plot: bool = True,
) -> float:
    """
    Finds the time delay of the strongest echo by first locating the
    direct speaker-to-mic leak (t0) and then searching for the strongest
    secondary peak in the physically plausible window.
    """
    emitted = np.asarray(emitted, dtype=np.float64)
    recorded = np.asarray(recorded, dtype=np.float64)

    correlation = correlate(recorded, emitted, mode="full")
    lags = correlation_lags(recorded.shape[0], emitted.shape[0], mode="full")
    mag = np.abs(correlation)

    floor = np.median(mag[mag > 0]) + 1e-12
    strong = np.where(mag > 8 * floor)[0]
    if strong.size == 0:
        raise ValueError("No correlation signal found (mic might be muted or blocked).")

    # t0 = earliest strong peak (direct leak); require lag >= 0 to avoid wrap
    ref_candidates = strong[lags[strong] >= 0]
    if ref_candidates.size == 0:
        raise ValueError("No positive-lag correlation signal found.")

    ref = ref_candidates[0]

    min_gap = int(0.002 * sample_rate)  # ignore < ~35 cm from laptop
    max_idx = ref + int((2 * max_range_m / 343.0) * sample_rate)
    
    seg = mag[ref + min_gap : max_idx]
    if seg.size == 0 or seg.max() < 6 * floor:
        raise ValueError("No valid echo found within max range.")

    echo_idx = ref + min_gap + int(np.argmax(seg))
    delay_s = (lags[echo_idx] - lags[ref]) / sample_rate

    if debug_plot:
        plt.figure(figsize=(10, 4))
        lag_times_ms = lags / sample_rate * 1000
        plt.plot(lag_times_ms, mag, label="Cross-correlation magnitude")
        
        t0_ms = lags[ref] / sample_rate * 1000
        echo_ms = lags[echo_idx] / sample_rate * 1000
        
        plt.axvline(t0_ms, color="gray", linestyle=":", label=f"t0 (leak): {t0_ms:.2f}ms")
        plt.axvline(echo_ms, color="red", linestyle="--", label=f"Echo peak: {echo_ms:.2f}ms")
        
        plt.xlim(t0_ms - 5, echo_ms + 10)
        plt.title("Cross-correlation (Self-aligned)")
        plt.xlabel("Lag (ms)")
        plt.ylabel("Correlation magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return delay_s


if __name__ == "__main__":
    # Quick self-test with synthetic data: build a fake "recording" that's
    # silence, direct bleed, then a delayed+attenuated copy of the chirp
    # (simulating a real echo), and confirm find_echo_delay recovers the
    # correct delay while ignoring the bleed.
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "phase1_chirp_hardware"))
    from chirp_generator import generate_chirp  # noqa: E402

    SAMPLE_RATE = 48000
    # Use the same audible band as the rest of the system (4-8 kHz, 75 ms)
    chirp_signal = generate_chirp(4000, 8000, 0.075, SAMPLE_RATE)

    bleed_delay_samples = int(0.001 * SAMPLE_RATE)   # 1ms direct bleed
    echo_delay_samples = int(0.006 * SAMPLE_RATE)     # 6ms simulated echo
    total_len = echo_delay_samples + chirp_signal.shape[0] + 500

    fake_recording = np.zeros(total_len)
    fake_recording[bleed_delay_samples: bleed_delay_samples + chirp_signal.shape[0]] += 0.3 * chirp_signal
    fake_recording[echo_delay_samples: echo_delay_samples + chirp_signal.shape[0]] += 0.6 * chirp_signal

    try:
        detected_delay = find_echo_delay(
            chirp_signal, fake_recording, SAMPLE_RATE, max_range_m=5.0, debug_plot=False
        )
        print(f"Expected echo delay: 6.00ms, Detected: {detected_delay*1000:.2f}ms")
    except ValueError as e:
        print(f"Self-test failed: {e}")
