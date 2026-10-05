"""
config_loader.py

Shared helper that loads config.json written by the Phase 1 auto-calibrator
(freq_response_test.py).  All later phases call ``load_sonar_config()``
instead of hard-coding chirp frequencies, so every machine automatically
uses the optimal band chosen during calibration.

Default band: 4-8 kHz (audible).  Sounds like a soft bird chirp at low
volume; works on every laptop speaker and mic without any hardware quirks.

Usage
-----
    from config_loader import load_sonar_config
    cfg = load_sonar_config()
    # cfg["f_start"], cfg["f_end"], cfg["sample_rate"] are always present.
"""

import os
import json

# Import the device-query helper from the same utils/ directory.
try:
    import sys as _sys
    _sys.path.append(os.path.dirname(__file__))
    from audio_io import query_device_sample_rate as _query_rate
    _DEVICE_RATE: int = _query_rate()
except Exception:
    _DEVICE_RATE = 48_000  # safe last-resort if sounddevice isn't installed yet

# The config file lives at the project root (one level above utils/).
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Hard-coded fallback: 4-8 kHz is reproduced cleanly by every modern laptop
# speaker and mic, so the system works out-of-the-box even without running
# the calibrator first.  We use the actual device rate queried above so the
# fallback is already correct for 44.1 kHz-native hardware.
_FALLBACK = {
    "f_start": 4_000,
    "f_end": 8_000,
    "sample_rate": _DEVICE_RATE,
    "calibrated": False,
    "note": f"built-in fallback (4-8 kHz @ {_DEVICE_RATE} Hz) — run freq_response_test.py to auto-calibrate",
}


def load_sonar_config(config_path: str = None) -> dict:
    """
    Load and return the sonar configuration dictionary.

    The dict is guaranteed to contain at least:
        f_start      (float)  — chirp start frequency in Hz
        f_end        (float)  — chirp end   frequency in Hz
        sample_rate  (int)    — audio sample rate in Hz
        calibrated   (bool)   — True if the file was written by auto-calibration
        note         (str)    — human-readable description of how it was created

    If config.json is missing or malformed, the built-in fallback is returned
    and a warning is printed so the user knows to run calibration.

    Parameters
    ----------
    config_path : str, optional
        Override the default config.json path.  Useful for testing.
    """
    path = config_path or _DEFAULT_CONFIG_PATH

    if not os.path.exists(path):
        print(
            f"[config_loader] WARNING: {os.path.abspath(path)} not found.\n"
            "  Run  python phase1_chirp_hardware/freq_response_test.py  to calibrate.\n"
            f"  Using built-in fallback: {_FALLBACK['f_start']/1000:.0f}–"
            f"{_FALLBACK['f_end']/1000:.0f} kHz"
        )
        return dict(_FALLBACK)

    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[config_loader] WARNING: could not parse {path}: {exc}\n"
            f"  Using built-in fallback: {_FALLBACK['f_start']/1000:.0f}–"
            f"{_FALLBACK['f_end']/1000:.0f} kHz"
        )
        return dict(_FALLBACK)

    # Merge: ensure every required key is present (guards against partial files)
    merged = dict(_FALLBACK)
    merged.update(cfg)

    if not cfg.get("calibrated", False):
        print(
            "[config_loader] NOTE: config.json contains fallback values "
            f"({cfg.get('note', '')}).\n"
            "  Run  python phase1_chirp_hardware/freq_response_test.py  "
            "for machine-specific calibration."
        )

    return merged


def config_summary(cfg: dict) -> str:
    """Return a one-line summary string suitable for log output."""
    status = "calibrated" if cfg.get("calibrated") else "fallback"
    return (
        f"chirp {cfg['f_start']/1000:.1f}–{cfg['f_end']/1000:.1f} kHz  "
        f"@ {cfg['sample_rate']} Hz  [{status}]"
    )
