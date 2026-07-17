"""
audible_tone_test.py

Quick sanity check: plays a clearly audible 2kHz tone and records
simultaneously, to confirm the speaker->mic playback/record loop
actually transmits sound at all — isolating that from the separate
question of whether 18-22kHz specifically is too high for your
laptop's speaker or mic to handle.
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
import audio_io  # noqa: E402


if __name__ == "__main__":
    SAMPLE_RATE = 48000
    FREQ_HZ = 2000        # clearly audible, well within any speaker/mic's range
    DURATION_SEC = 0.3    # 300ms tone, long enough to see clearly on a plot
    RECORD_DURATION_SEC = 0.5

    t = np.linspace(0, DURATION_SEC, int(DURATION_SEC * SAMPLE_RATE), endpoint=False)
    tone = (0.8 * np.sin(2 * np.pi * FREQ_HZ * t)).astype(np.float32)

    print(f"Playing a {FREQ_HZ}Hz tone and recording {RECORD_DURATION_SEC*1000:.0f}ms...")
    recorded = audio_io.play_and_record(tone, SAMPLE_RATE, RECORD_DURATION_SEC)

    max_amp = np.max(np.abs(recorded))
    print(f"Max recorded amplitude: {max_amp:.6f}")

    if max_amp < 0.001:
        print("\n[WARNING] Even an audible 2kHz tone isn't being picked up above the noise floor.")
        print("    This suggests a system audio routing issue (wrong output device, muted")
        print("    output, or system volume near zero) rather than an ultrasonic-specific")
        print("    hardware limit. Check Windows Sound Settings > Output, and turn system")
        print("    volume up.")
    else:
        print("\n[SUCCESS] The playback/record loop works — the tone was captured clearly.")
        print("    This confirms the earlier near-zero result with the 18-22kHz chirp is")
        print("    specifically an ultrasonic rolloff issue, not a general playback problem.")
        print("    Run freq_response_test.py next to find your mic's actual usable usable ceiling.")

    t_rec = np.linspace(0, RECORD_DURATION_SEC, recorded.shape[0], endpoint=False) * 1000
    plt.figure(figsize=(10, 4))
    plt.plot(t_rec, recorded)
    plt.title(f"Recorded audio ({FREQ_HZ}Hz test tone)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
