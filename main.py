"""
main.py

Ties the whole acoustic sonar system together: runs the Phase 3
continuous sensing loop in a background thread, updating shared live
state (distance, motion flag, latest recording), while the Phase 4
matplotlib visualization (radar gauge + scrolling waveform, side by
side) reads that live state on the main thread instead of using stub
functions.

Run with: python main.py
Press Ctrl+C (or close the plot window) to stop.
"""

import sys
import os
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "phase1_chirp_hardware"))
sys.path.append(os.path.join(os.path.dirname(__file__), "phase2_cross_correlation"))
sys.path.append(os.path.join(os.path.dirname(__file__), "phase3_continuous_sensing"))

import audio_io  # noqa: E402
from chirp_generator import generate_chirp, generate_varied_chirp  # noqa: E402
from config_loader import load_sonar_config, config_summary  # noqa: E402
from distance_calc import delay_to_distance  # noqa: E402
from rolling_buffer import RollingBuffer  # noqa: E402
from motion_detector import compare_profiles  # noqa: E402
from continuous_loop import compute_correlation_profile  # noqa: E402


# --- Sensing parameters — chirp band loaded from auto-calibration ---
_cfg = load_sonar_config()
F_START = _cfg["f_start"]
F_END = _cfg["f_end"]
SAMPLE_RATE = _cfg["sample_rate"]
print(f"[main] Loaded sonar config: {config_summary(_cfg)}")

CHIRP_DURATION_SEC = 0.075    # 75 ms chirp: high TBP, barely-audible whoosh
RECORD_DURATION_SEC = 0.4     # 400 ms: catches echoes up to ~68 m
MIN_DELAY_SEC = 0.003
CYCLE_INTERVAL_SEC = 0.6      # must be > RECORD_DURATION_SEC
BUFFER_SIZE = 5
MOTION_THRESHOLD = 0.15

MAX_DISTANCE_M = 2.0
MOTION_DISPLAY_HOLD_SEC = 1.0  # how long the "MOTION DETECTED" text stays lit


class SharedSensorState:
    """
    Thread-safe container for the sensing loop's live output, read by
    the visualization on the main thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.distance_m = MAX_DISTANCE_M / 2
        self.motion_detected = False
        self.last_motion_time = 0.0
        self.latest_recording = np.zeros(int(RECORD_DURATION_SEC * SAMPLE_RATE))

    def update(self, distance_m, motion_detected, latest_recording):
        with self._lock:
            self.distance_m = distance_m
            self.motion_detected = motion_detected
            if motion_detected:
                self.last_motion_time = time.time()
            self.latest_recording = latest_recording

    def snapshot(self):
        with self._lock:
            motion_recent = (time.time() - self.last_motion_time) < MOTION_DISPLAY_HOLD_SEC
            return self.distance_m, motion_recent, self.latest_recording


def sensing_loop(state: SharedSensorState, stop_event: threading.Event):
    """
    Real version of Phase 3's continuous_loop.py, writing results into
    `state` instead of printing them, so the visualization can read them.

    Each cycle regenerates the chirp with a fresh ping_index so the OS
    acoustic echo canceller (AEC) cannot lock onto a repeating pattern.
    The same waveform is used as both the playback signal and the
    correlation template — the matched filter is always exact.
    """
    buffer = RollingBuffer(max_size=BUFFER_SIZE)
    ping_index = 0
    motion_history = []

    while not stop_event.is_set():
        cycle_start = time.time()

        # Fresh chirp variant for this ping — defeats statistical AEC models.
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
                emitted, recorded, SAMPLE_RATE, MIN_DELAY_SEC
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
        if distance_m is not None:
            if len(buffer) > 0:
                reference_profile = buffer.average()
                raw_motion, _ = compare_profiles(reference_profile, profile, MOTION_THRESHOLD)
            buffer.add(profile)
            
        motion_history.append(raw_motion)
        if len(motion_history) > 5:
            motion_history.pop(0)
            
        # Require 3 of 5 valid motion triggers
        motion_detected = sum(motion_history) >= 3

        report_dist = distance_m if distance_m is not None else 5.0
        state.update(report_dist, motion_detected, recorded)

        elapsed = time.time() - cycle_start
        stop_event.wait(max(0.0, CYCLE_INTERVAL_SEC - elapsed))


def _distance_to_angle_deg(distance_m: float) -> float:
    clamped = max(0.0, min(distance_m, MAX_DISTANCE_M))
    fraction = clamped / MAX_DISTANCE_M
    return 180.0 * (1.0 - fraction)


def build_visualization(state: SharedSensorState):
    fig, (radar_ax, wave_ax) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Radar subplot setup ---
    radar_ax.set_xlim(-1.2, 1.2)
    radar_ax.set_ylim(-0.2, 1.2)
    radar_ax.set_aspect("equal")
    radar_ax.axis("off")
    theta = np.linspace(0, np.pi, 100)
    radar_ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=2)
    for frac, label in [(0.0, f"{MAX_DISTANCE_M:.0f}m"), (0.5, f"{MAX_DISTANCE_M/2:.1f}m"), (1.0, "0m")]:
        angle_rad = np.pi * frac
        radar_ax.text(np.cos(angle_rad) * 1.1, np.sin(angle_rad) * 1.1, label, ha="center", va="center", fontsize=9)
    needle_line, = radar_ax.plot([], [], color="tab:red", linewidth=3)
    distance_text = radar_ax.text(0, -0.1, "", ha="center", va="center", fontsize=12)
    motion_text = radar_ax.text(0, 0.5, "", ha="center", va="center", fontsize=16, color="red", weight="bold")

    # --- Waveform subplot setup ---
    wave_ax.set_title("Live recorded echo waveform")
    wave_ax.set_xlabel("Time (ms)")
    wave_ax.set_ylabel("Amplitude")
    wave_ax.set_ylim(-1.0, 1.0)
    wave_ax.grid(True)
    wave_line, = wave_ax.plot([], [], color="tab:orange")

    def update(frame):
        distance_m, motion_recent, recording = state.snapshot()

        angle_rad = np.radians(_distance_to_angle_deg(distance_m))
        needle_line.set_data([0, np.cos(angle_rad)], [0, np.sin(angle_rad)])
        distance_text.set_text(f"{distance_m*100:.1f} cm")
        motion_text.set_text("MOTION DETECTED" if motion_recent else "")

        t_ms = np.linspace(0, recording.shape[0] / SAMPLE_RATE * 1000, recording.shape[0])
        wave_line.set_data(t_ms, recording)
        wave_ax.set_xlim(t_ms[0], t_ms[-1] if t_ms[-1] > 0 else 1)

        return needle_line, distance_text, motion_text, wave_line

    anim = animation.FuncAnimation(
        fig, update, interval=int(CYCLE_INTERVAL_SEC * 1000), blit=False, cache_frame_data=False
    )
    return fig, anim


if __name__ == "__main__":
    state = SharedSensorState()
    stop_event = threading.Event()

    sensing_thread = threading.Thread(target=sensing_loop, args=(state, stop_event), daemon=True)
    sensing_thread.start()

    fig, anim = build_visualization(state)

    try:
        plt.tight_layout()
        plt.show()  # blocks on the main thread until the window is closed
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping sensing loop...")
        stop_event.set()
        sensing_thread.join(timeout=2.0)
