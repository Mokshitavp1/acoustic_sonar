"""
audio_io.py

Shared audio I/O helpers for the acoustic sonar project.
Provides simultaneous play+record via the default speaker/mic, and
a simple .wav save utility.
"""

import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def play_and_record(signal: np.ndarray, sample_rate: int, record_duration_sec: float) -> np.ndarray:
    """
    Plays `signal` through the default output device while simultaneously
    recording `record_duration_sec` seconds of audio from the default
    input device.

    Args:
        signal: 1D numpy array of audio samples (float32, range -1.0 to 1.0) to play.
        sample_rate: Sample rate in Hz, used for both playback and recording.
        record_duration_sec: How long to record, in seconds.

    Returns:
        1D numpy array of recorded audio samples (float32).

    Raises:
        RuntimeError: if no working input/output audio device is found.
    """
    signal = np.asarray(signal, dtype=np.float32)
    record_frames = int(record_duration_sec * sample_rate)

    try:
        # sd.playrec plays `signal` and records `record_frames` frames at the
        # same time, using the default input/output devices.
        recording = sd.playrec(
            signal.reshape(-1, 1),
            samplerate=sample_rate,
            channels=1,
            frames=record_frames if record_frames > signal.shape[0] else None,
        )
        sd.wait()  # block until playback/recording finishes
    except sd.PortAudioError as e:
        raise RuntimeError(
            "No working audio input/output device found. "
            "Check that a mic and speaker are connected and not in use by another app."
        ) from e

    return recording.flatten()


def save_wav(filename: str, signal: np.ndarray, sample_rate: int) -> None:
    """
    Saves a numpy audio array as a .wav file.

    Args:
        filename: Output path, e.g. 'data/chirp.wav'.
        signal: 1D numpy array of audio samples (float32, range -1.0 to 1.0).
        sample_rate: Sample rate in Hz.
    """
    signal = np.asarray(signal, dtype=np.float32)
    # Clip to valid range before writing, in case of any overshoot.
    signal = np.clip(signal, -1.0, 1.0)
    wavfile.write(filename, sample_rate, signal)
