"""
device_check.py

Quick diagnostic: lists available audio devices, shows which ones are
currently set as default, and records 2 seconds of audio while you talk
or make noise, so you can confirm the mic is actually capturing normal-
amplitude signal before troubleshooting the ultrasonic chirp specifically.
"""

import sounddevice as sd
import numpy as np


if __name__ == "__main__":
    print("=== Available audio devices ===")
    print(sd.query_devices())

    print("\n=== Current default devices ===")
    print(f"Default input device index: {sd.default.device[0]}")
    print(f"Default output device index: {sd.default.device[1]}")

    default_input_info = sd.query_devices(sd.default.device[0])
    print(f"Default input device name: {default_input_info['name']}")

    print("\n=== Recording 2 seconds — talk, clap, or make noise now ===")
    sample_rate = 48000
    duration_sec = 2.0
    recording = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    recording = recording.flatten()

    print(f"\nRecorded {recording.shape[0]} samples.")
    print(f"Max absolute amplitude: {np.max(np.abs(recording)):.6f}")
    print(f"RMS amplitude: {np.sqrt(np.mean(recording**2)):.6f}")

    if np.max(np.abs(recording)) < 0.001:
        print("\n⚠️  Amplitude is still near-zero. The mic likely isn't being captured at all —")
        print("    check Windows Sound Settings > Input, confirm the right mic is selected")
        print("    and its volume/level isn't muted or set to 0.")
    else:
        print("\n✅ Mic is capturing normal-range audio. The earlier near-zero amplitude was")
        print("    likely specific to the ultrasonic chirp frequency range, not a device issue.")
