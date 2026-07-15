"""
radar_view.py

Phase 4: a live-updating matplotlib animation showing a radar-sweep-style
circular gauge, where a needle position maps to the current estimated
distance, plus a "MOTION DETECTED" text overlay that flashes when
triggered.

Uses stub get_latest_distance() / get_motion_flag() functions returning
dummy values for now — swap these out for real calls into the Phase 3
continuous_loop's live state once this view is confirmed working.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Gauge range: needle sweeps from 0 to MAX_DISTANCE_M meters.
MAX_DISTANCE_M = 2.0

# Module-level state to fake a "motion just happened" flash effect.
_last_motion_time = 0.0
_MOTION_FLASH_DURATION_SEC = 1.0


def get_latest_distance() -> float:
    """
    STUB: returns a fake "current estimated distance" for testing this
    view in isolation. Replace with a real function reading the latest
    distance from the Phase 3 sensing loop.

    Returns:
        Distance estimate in meters, between 0 and MAX_DISTANCE_M.
    """
    # Wander slowly around the middle of the range so the needle has
    # something plausible-looking to do.
    t = time.time()
    return MAX_DISTANCE_M / 2 + 0.3 * np.sin(t)


def get_motion_flag() -> bool:
    """
    STUB: returns a fake "motion detected" flag for testing this view in
    isolation. Replace with a real function reading the latest motion
    state from the Phase 3 sensing loop.

    Returns:
        True if motion should be flagged as just-detected, else False.
    """
    global _last_motion_time
    # Randomly "detect motion" every so often so the flash effect can be
    # visually verified.
    if np.random.rand() < 0.05:
        _last_motion_time = time.time()
    return (time.time() - _last_motion_time) < _MOTION_FLASH_DURATION_SEC


def _distance_to_angle_deg(distance_m: float) -> float:
    """
    Maps a distance (0 to MAX_DISTANCE_M) onto a radar sweep angle,
    where 0 distance = 180 degrees (left) and MAX_DISTANCE_M = 0 degrees
    (right), sweeping across the top half of the gauge like a
    semicircular radar display.
    """
    clamped = max(0.0, min(distance_m, MAX_DISTANCE_M))
    fraction = clamped / MAX_DISTANCE_M
    return 180.0 * (1.0 - fraction)


if __name__ == "__main__":
    UPDATE_INTERVAL_MS = 200

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw the static semicircular gauge background.
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), color="gray", linewidth=2)

    # Distance tick labels around the arc.
    for frac, label in [(0.0, f"{MAX_DISTANCE_M:.0f}m"), (0.5, f"{MAX_DISTANCE_M/2:.1f}m"), (1.0, "0m")]:
        angle_rad = np.pi * frac
        ax.text(np.cos(angle_rad) * 1.1, np.sin(angle_rad) * 1.1, label, ha="center", va="center", fontsize=9)

    needle_line, = ax.plot([], [], color="tab:red", linewidth=3)
    distance_text = ax.text(0, -0.1, "", ha="center", va="center", fontsize=12)
    motion_text = ax.text(0, 0.5, "", ha="center", va="center", fontsize=16, color="red", weight="bold")

    def update(frame):
        distance_m = get_latest_distance()
        motion = get_motion_flag()

        angle_deg = _distance_to_angle_deg(distance_m)
        angle_rad = np.radians(angle_deg)
        needle_line.set_data([0, np.cos(angle_rad)], [0, np.sin(angle_rad)])

        distance_text.set_text(f"{distance_m*100:.1f} cm")
        motion_text.set_text("MOTION DETECTED" if motion else "")

        return needle_line, distance_text, motion_text

    anim = animation.FuncAnimation(
        fig, update, interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()
