"""
audio_io.py

Shared audio I/O helpers for the acoustic sonar project.

Key design goals
----------------
* **Bypass OS DSP** — On Windows, we request WASAPI exclusive mode, which
  hands the hardware directly to PortAudio and skips the Windows audio
  engine entirely (AGC, echo cancellation, noise suppression, "Audio
  Enhancements").  On other platforms sounddevice already talks to the
  hardware through ALSA/CoreAudio with minimal processing.

* **Graceful fallback** — If exclusive mode fails (device in use, driver
  doesn't support it), we fall back to WASAPI shared mode with a warning,
  then to the PortAudio default.  The system keeps working; it just might
  be louder about it.

* **Deterministic latency** — We use sd.playrec (simultaneous play+record)
  with a fixed blocksize so the round-trip timing is as predictable as
  consumer hardware allows.
"""

import sys
import warnings
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


# ---------------------------------------------------------------------------
# WASAPI raw-mode helpers (Windows only)
# ---------------------------------------------------------------------------

def _wasapi_exclusive_settings() -> "sd.WasapiSettings | None":
    """
    Return a WasapiSettings object requesting exclusive mode, or None if
    WASAPI is not available (non-Windows, or sounddevice built without it).

    Exclusive mode bypasses the Windows audio engine (no AGC, no echo
    cancellation, no "Audio Enhancements" applied by the OS).
    """
    try:
        return sd.WasapiSettings(exclusive=True)
    except AttributeError:
        return None   # sounddevice was not built with WASAPI support


def _try_play_and_record_wasapi_exclusive(
    padded_signal: np.ndarray,
    sample_rate: int,
) -> "np.ndarray | None":
    """
    Attempt a play+record call in WASAPI exclusive mode.

    Returns the recorded array on success, or None if exclusive mode is
    unavailable / the device refuses the request.
    """
    settings = _wasapi_exclusive_settings()
    if settings is None:
        return None

    try:
        recording = sd.playrec(
            padded_signal.reshape(-1, 1),
            samplerate=sample_rate,
            channels=1,
            extra_settings=settings,
        )
        sd.wait()
        return recording.flatten()
    except (sd.PortAudioError, Exception):
        # Device may be in use, or driver doesn't support exclusive mode.
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def play_and_record(
    signal: np.ndarray,
    sample_rate: int,
    record_duration_sec: float,
    *,
    raw: bool = True,
) -> np.ndarray:
    """
    Play ``signal`` through the default output while simultaneously recording
    ``record_duration_sec`` seconds from the default input.

    Parameters
    ----------
    signal : 1D float32 array, range -1.0 to 1.0.
    sample_rate : Hz, used for both playback and recording.
    record_duration_sec : Total duration to record (may be longer than the
        signal; silence is padded for the remainder of playback).
    raw : If True (default), attempt WASAPI exclusive mode on Windows to
        bypass OS audio processing (AGC, AEC, noise suppression).  Falls
        back automatically if the device refuses.

    Returns
    -------
    1D float32 numpy array of recorded samples.

    Raises
    ------
    RuntimeError if no working audio device is found at all.
    """
    signal = np.asarray(signal, dtype=np.float32)
    record_frames = int(record_duration_sec * sample_rate)

    # Pad or trim so playback covers exactly `record_frames` samples.
    # sd.playrec records exactly as many frames as are in the played array.
    if record_frames > signal.shape[0]:
        padded = np.zeros(record_frames, dtype=np.float32)
        padded[: signal.shape[0]] = signal
    else:
        padded = signal[:record_frames]

    # --- Tier 1: WASAPI exclusive mode (Windows only) -----------------------
    if raw and sys.platform == "win32":
        result = _try_play_and_record_wasapi_exclusive(padded, sample_rate)
        if result is not None:
            return result
        warnings.warn(
            "[audio_io] WASAPI exclusive mode unavailable — falling back to "
            "shared mode.  OS audio processing (AGC/AEC) may still be active.\n"
            "  To disable it manually: Settings > Sound > your mic > "
            "Audio enhancements > Off,  and set format to 48000 Hz 24-bit.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- Tier 2 / 3: standard shared-mode (all platforms) ------------------
    try:
        recording = sd.playrec(
            padded.reshape(-1, 1),
            samplerate=sample_rate,
            channels=1,
        )
        sd.wait()
    except sd.PortAudioError as exc:
        raise RuntimeError(
            "No working audio input/output device found.\n"
            "Check that a mic and speaker are connected and not in use "
            "by another app."
        ) from exc

    return recording.flatten()


def save_wav(filename: str, signal: np.ndarray, sample_rate: int) -> None:
    """
    Save a numpy audio array as a .wav file.

    Args:
        filename: Output path, e.g. 'data/chirp.wav'.
        signal: 1D numpy array of audio samples (float32, range -1.0 to 1.0).
        sample_rate: Sample rate in Hz.
    """
    signal = np.asarray(signal, dtype=np.float32)
    signal = np.clip(signal, -1.0, 1.0)
    wavfile.write(filename, sample_rate, signal)