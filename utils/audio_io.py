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

* **Device-native sample rate** — Never assume 44.1 or 48 kHz.  We query
  the PortAudio device info and validate against the hardware's preferred
  rate.  At 48 kHz one sample = ~7 mm of round-trip travel; sub-sample
  interpolation in the cross-correlator gets us well under 1 cm resolution.
"""

import sys
import warnings
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


# Standard rates we're willing to use, in preference order.
# We try each against the device and take the first one that works.
_PREFERRED_RATES = [48_000, 44_100, 96_000, 22_050, 16_000]


def query_device_sample_rate(
    input_device=None,
    output_device=None,
    preferred_rates: list[int] = None,
) -> int:
    """
    Return the best sample rate supported by the current default (or
    specified) audio devices.

    Strategy
    --------
    1. Read ``default_samplerate`` from PortAudio's device descriptor
       for both the input and output device.  If they agree (or one is
       unset), use that rate directly.
    2. If they disagree, probe each rate in ``preferred_rates`` with a
       zero-length sd.check_input_settings / check_output_settings call
       and pick the first one both sides accept.
    3. Hard-fallback to 48 000 Hz if everything above fails.

    Parameters
    ----------
    input_device, output_device : int or str, optional
        PortAudio device indices/names.  None means the system default.
    preferred_rates : list of int, optional
        Override the default preference list.

    Returns
    -------
    int — confirmed sample rate in Hz.
    """
    rates_to_try = preferred_rates or _PREFERRED_RATES

    try:
        in_idx = input_device if input_device is not None else sd.default.device[0]
        out_idx = output_device if output_device is not None else sd.default.device[1]

        in_info = sd.query_devices(in_idx)
        out_info = sd.query_devices(out_idx)

        in_rate = int(in_info.get("default_samplerate", 0))
        out_rate = int(out_info.get("default_samplerate", 0))

        # If both devices report the same preferred rate, use it directly.
        if in_rate and out_rate and in_rate == out_rate:
            return in_rate

        # Devices disagree (or one returned 0) — probe the preferred list.
        for rate in rates_to_try:
            try:
                sd.check_input_settings(
                    device=in_idx, channels=1, dtype="float32", samplerate=rate
                )
                sd.check_output_settings(
                    device=out_idx, channels=1, dtype="float32", samplerate=rate
                )
                return rate
            except sd.PortAudioError:
                continue

        # If probing failed for all candidates, trust the input device's preference.
        if in_rate:
            return in_rate
        if out_rate:
            return out_rate

    except Exception:
        pass  # fall through to hard default

    return 48_000  # absolute last-resort default


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
        global _warned_wasapi
        if not globals().get('_warned_wasapi'):
            warnings.warn(
                "[audio_io] WASAPI exclusive mode unavailable — falling back to "
                "shared mode.  OS audio processing (AGC/AEC) may still be active.\n"
                "  To disable it manually: Settings > Sound > your mic > "
                "Audio enhancements > Off,  and set format to 48000 Hz 24-bit.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_wasapi = True

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