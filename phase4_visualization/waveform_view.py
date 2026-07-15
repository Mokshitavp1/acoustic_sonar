"""
waveform_view.py

Phase 4: a live-updating matplotlib animation that continuously plots
the most recently recorded echo waveform, replacing the previous frame
each update.

Uses a stub get_latest_recording() that returns random data for now —
swap this out for a real call into the Phase 3 continuous_loop's live
recording once this view is confirmed working.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_latest_recording() -> np.ndarray:
    """
    STUB: returns a fake "most recent recording" for testing this view
    in isolation. Replace this with a real function that pulls the
    latest recorded audio array from the Phase 3 sensing loop (e.g. via
    a shared variable, queue, or callback set up in main.py).

    Returns:
        1D numpy array of fake audio samples.
    """
    sample_rate = 48000
    duration_sec = 0.05
    n = int(duration_sec * sample_rate)
    # Simulate a recording: mostly noise, with an occasional "echo blip"
    # at a random position, so the view has something visually
    # interesting to scroll through.
    fake_recording = np.random.normal(scale=0.02, size=n)
    if np.random.rand() < 0.5:
        blip_start = np.random.randint(0, n - 200)
        t = np.linspace(0, 1, 200)
        fake_recording[blip_start: blip_start + 200] += 0.5 * np.sin(2 * np.pi * 10 * t) * np.hanning(200)
    return fake_recording


if __name__ == "__main__":
    SAMPLE_RATE = 48000
    UPDATE_INTERVAL_MS = 300  # matches Phase 3's default 300ms cycle

    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], color="tab:orange")

    ax.set_title("Live recorded echo waveform")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True)

    def update(frame):
        recording = get_latest_recording()
        t_ms = np.linspace(0, recording.shape[0] / SAMPLE_RATE * 1000, recording.shape[0])
        line.set_data(t_ms, recording)
        ax.set_xlim(t_ms[0], t_ms[-1])
        return (line,)

    anim = animation.FuncAnimation(
        fig, update, interval=UPDATE_INTERVAL_MS, blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()
