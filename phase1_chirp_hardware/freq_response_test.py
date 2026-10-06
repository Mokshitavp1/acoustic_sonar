"""
freq_response_test.py  —  Phase 1 Auto-Calibrator
===================================================

On first run this script acts as a self-calibrating tool:

  1. Plays a 1-second frequency sweep covering 2 kHz -> 12 kHz
     (the audible range both laptop speakers and mics handle well).
  2. Records the round-trip (speaker -> room -> mic).
  3. Slices the FFT into 2 kHz-wide bands and measures the RMS energy
     the mic picks up in each band.
  4. Finds the *best-energy* contiguous window (2 bands wide = 4 kHz)
     where all bands exceed a noise-floor threshold.
  5. Writes the chosen [f_start, f_end] + sample_rate to
     ``../config.json`` so every other phase reads it automatically.

Why audible (4-8 kHz) instead of ultrasonic?
  - Every laptop speaker and mic reproduces 4-8 kHz cleanly.
  - A 75 ms swept chirp in this band sounds like a faint bird chirp
    at low volume - tolerable as a demo.
  - Cross-correlation with a chirp gives the same excellent range
    resolution and noise immunity regardless of the frequency band.
  - No calibration surprises: the physics works on every machine.

Run:
    python freq_response_test.py              # calibrate + save config
    python freq_response_test.py --plot-only  # re-plot last calibration
    python freq_response_test.py --force      # re-calibrate even if config exists
"""

import sys
import os
import json
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow importing from utils/ and this phase's own chirp_generator.py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.dirname(__file__))
import audio_io  # noqa: E402
from audio_io import query_device_sample_rate  # noqa: E402
from chirp_generator import generate_chirp  # noqa: E402


# ---------------------------------------------------------------------------
# Constants resolved at startup
# ---------------------------------------------------------------------------

# Query the device's actual preferred sample rate instead of assuming 48 kHz.
# At 48 kHz: 1 sample = 343 m/s / 48000 Hz / 2 (round-trip) ≈ 3.6 mm.
# At 44.1 kHz: 1 sample ≈ 3.9 mm.  Sub-sample interpolation in the
# cross-correlator brings effective resolution well under 1 cm on either.
SAMPLE_RATE: int = query_device_sample_rate()
print(f"[calibrator] Device sample rate: {SAMPLE_RATE} Hz")

SWEEP_F_START = 2_000        # Hz  — bottom of the audible probe sweep
SWEEP_F_END = 12_000         # Hz  — top  (well within any laptop speaker/mic)
SWEEP_DURATION_SEC = 1.0     # long enough for a clean FFT of narrow bands
RECORD_DURATION_SEC = 1.3    # record a little longer than the sweep

BAND_WIDTH_HZ = 2_000        # width of each analysis band (Hz)
WINDOW_BANDS = 2             # number of consecutive bands in the chosen window
                              # -> total chirp bandwidth = BAND_WIDTH_HZ x WINDOW_BANDS = 4 kHz

# A band is "good" if its energy >= THRESHOLD_FACTOR x median band energy.
THRESHOLD_FACTOR = 0.35

# When comparing candidate windows, prefer the one with the highest total
# energy (no high-freq bias — in the audible range we just want the best band).
PREFER_HIGH_FACTOR = 0.90

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

# Sensible fallback: 4-8 kHz works on every modern laptop.
# Rate is filled in at runtime after the device query above.
FALLBACK_CONFIG = {
    "f_start": 4_000,
    "f_end": 8_000,
    "sample_rate": SAMPLE_RATE,
    "calibrated": False,
    "note": "fallback — calibration did not find a good band",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_band_energies(
    recording: np.ndarray,
    sample_rate: int,
    f_low: float,
    f_high: float,
    band_width: float,
):
    """
    Slice the FFT of ``recording`` into adjacent ``band_width``-wide bins
    between ``f_low`` and ``f_high`` and return the RMS energy per band.

    Returns
    -------
    band_centres : 1D array of band centre frequencies (Hz)
    energies     : 1D array of RMS energy for each band
    """
    n = recording.shape[0]
    fft_vals = np.fft.rfft(recording * np.hanning(n))   # Hann window reduces leakage
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    magnitude = np.abs(fft_vals)

    bands = np.arange(f_low, f_high, band_width)
    centres = bands + band_width / 2
    energies = np.zeros(len(bands))

    for i, f in enumerate(bands):
        mask = (freqs >= f) & (freqs < f + band_width)
        if mask.any():
            energies[i] = float(np.sqrt(np.mean(magnitude[mask] ** 2)))

    return centres, energies


def find_best_window(
    band_centres: np.ndarray,
    energies: np.ndarray,
    window_bands: int,
    threshold_factor: float,
    prefer_high_factor: float,
):
    """
    Among all consecutive ``window_bands``-wide windows, return the one that:
      - has ALL bands above ``threshold_factor x median energy``
      - is the *highest-frequency* such window whose energy is at least
        ``prefer_high_factor`` of the single best (by total energy) window.

    Returns (f_start, f_end, window_index) or (None, None, None) on failure.
    """
    n = len(energies)
    if n < window_bands:
        return None, None, None

    median_e = float(np.median(energies))
    good = energies >= threshold_factor * median_e   # bool mask per band

    # Slide window: score = sum of energies in the window
    window_scores = np.array(
        [energies[i : i + window_bands].sum() for i in range(n - window_bands + 1)]
    )
    window_all_good = np.array(
        [good[i : i + window_bands].all() for i in range(n - window_bands + 1)]
    )

    valid = np.where(window_all_good)[0]
    if len(valid) == 0:
        return None, None, None

    best_score = window_scores[valid].max()

    # Pick the window with the highest total energy (among valid windows
    # whose score is at least prefer_high_factor x best_score).
    # In the audible band we simply want the cleanest window, not necessarily
    # the highest-frequency one.
    candidates = valid[window_scores[valid] >= prefer_high_factor * best_score]
    best_idx = int(candidates[np.argmax(window_scores[candidates])])

    band_width = band_centres[1] - band_centres[0]
    f_start = band_centres[best_idx] - band_width / 2
    f_end = band_centres[best_idx + window_bands - 1] + band_width / 2
    return float(f_start), float(f_end), int(best_idx)


def run_calibration(verbose: bool = True) -> dict:
    """
    Play the probe sweep, record, analyse, pick the best band, return a
    config dict.  Does NOT write to disk.
    """
    if verbose:
        print(f">>  Playing probe sweep  {SWEEP_F_START/1000:.0f} kHz -> "
              f"{SWEEP_F_END/1000:.0f} kHz  ({SWEEP_DURATION_SEC:.1f} s)...")
        print("   Keep quiet and avoid moving objects near the laptop.")
        time.sleep(0.4)   # brief pause so the print flushes before audio

    sweep = generate_chirp(SWEEP_F_START, SWEEP_F_END, SWEEP_DURATION_SEC, SAMPLE_RATE)
    recording = audio_io.play_and_record(sweep, SAMPLE_RATE, RECORD_DURATION_SEC)

    if verbose:
        print("   Recording done.  Analysing...")

    centres, energies = compute_band_energies(
        recording, SAMPLE_RATE,
        SWEEP_F_START, SWEEP_F_END, BAND_WIDTH_HZ
    )

    f_start, f_end, win_idx = find_best_window(
        centres, energies, WINDOW_BANDS, THRESHOLD_FACTOR, PREFER_HIGH_FACTOR
    )

    if f_start is None:
        if verbose:
            print("WARNING: Could not find a good band - using fallback config.")
        cfg = dict(FALLBACK_CONFIG)
        cfg["band_centres_hz"] = centres.tolist()
        cfg["band_energies"] = energies.tolist()
        return cfg

    if verbose:
        print(f"OK  Best band: {f_start/1000:.1f} kHz -> {f_end/1000:.1f} kHz")
        print(">>  Measuring system latency...")

    # Emit a short click to find loopback system latency (OS DSP + buffer)
    click = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    click[100:132] = 0.9
    rec = audio_io.play_and_record(click, SAMPLE_RATE, 0.4)
    latency_s = float(np.argmax(np.abs(rec)) / SAMPLE_RATE)
    
    if verbose:
        print(f"OK  System latency: {latency_s*1000:.1f} ms")

    cfg = {
        "f_start": f_start,
        "f_end": f_end,
        "sample_rate": SAMPLE_RATE,
        "calibrated": True,
        "latency_s": latency_s,
        "note": (f"auto-calibrated on {time.strftime('%Y-%m-%d %T')} - "
                 f"best band {f_start/1000:.1f}-{f_end/1000:.1f} kHz"),
        "band_centres_hz": centres.tolist(),
        "band_energies": energies.tolist(),
        "best_window_index": win_idx,
    }
    return cfg


def write_config(cfg: dict) -> None:
    """Serialise ``cfg`` to ``CONFIG_PATH`` with pretty-printing."""
    os.makedirs(os.path.dirname(os.path.abspath(CONFIG_PATH)), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"   Config written -> {os.path.abspath(CONFIG_PATH)}")


def load_config() -> dict:
    """Return the existing config dict, or None if the file doesn't exist."""
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(cfg: dict) -> None:
    """
    Draw a two-panel figure:
      Left  - bar chart of per-band energy with the chosen window highlighted
      Right - 15 ms preview of the calibrated chirp waveform
    """
    centres = np.asarray(cfg.get("band_centres_hz", []))
    energies = np.asarray(cfg.get("band_energies", []))
    win_idx = cfg.get("best_window_index", None)
    f_start = cfg.get("f_start")
    f_end = cfg.get("f_end")
    calibrated = cfg.get("calibrated", False)

    fig = plt.figure(figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

    # ---- Left panel: energy bars -------------------------------------------
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.set_facecolor("#1a1a2e")

    if len(centres) > 0 and len(energies) > 0:
        colors = []
        for i in range(len(centres)):
            if (win_idx is not None and
                    win_idx <= i < win_idx + WINDOW_BANDS):
                colors.append("#00e5ff")   # cyan highlight = chosen window
            else:
                colors.append("#4a4a8a")   # muted purple = other bands

        bar_width = centres[1] - centres[0] if len(centres) > 1 else BAND_WIDTH_HZ
        ax_bar.bar(
            centres / 1000, energies,
            width=bar_width / 1000 * 0.85,
            color=colors, edgecolor="none", zorder=2,
        )

        # median threshold line
        median_e = float(np.median(energies))
        threshold = THRESHOLD_FACTOR * median_e
        ax_bar.axhline(
            threshold, color="#ff6b6b", linestyle="--", linewidth=1.2,
            label=f"Threshold ({THRESHOLD_FACTOR:.0%} of median)", zorder=3
        )

        if f_start is not None and f_end is not None:
            ax_bar.axvspan(f_start / 1000, f_end / 1000,
                           alpha=0.12, color="#00e5ff", zorder=1)

    ax_bar.set_title(
        f"[{'OK' if calibrated else 'WARNING'}]  "
        f"{'Calibrated chirp band' if calibrated else 'Fallback band'}"
        + (f":  {f_start/1000:.1f}-{f_end/1000:.1f} kHz" if f_start else ""),
        color="white", fontsize=12, pad=10
    )
    ax_bar.set_xlabel("Frequency band centre (kHz)", color="#aaa")
    ax_bar.set_ylabel("RMS energy (mic response)", color="#aaa")
    ax_bar.tick_params(colors="#aaa")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333")
    ax_bar.legend(facecolor="#1a1a2e", edgecolor="#555", labelcolor="white", fontsize=9)
    ax_bar.grid(axis="y", color="#2a2a4a", linewidth=0.7, zorder=0)

    # ---- Right panel: chirp preview ----------------------------------------
    ax_chirp = fig.add_subplot(gs[1])
    ax_chirp.set_facecolor("#1a1a2e")

    if f_start is not None and f_end is not None:
        preview_dur = 0.075
        preview = generate_chirp(f_start, f_end, preview_dur, SAMPLE_RATE)
        t_ms = np.linspace(0, preview_dur * 1000, preview.shape[0], endpoint=False)
        ax_chirp.plot(t_ms, preview, color="#00e5ff", linewidth=0.5)
        title_str = (f"Calibrated chirp\n"
                     f"{f_start/1000:.1f} -> {f_end/1000:.1f} kHz (75 ms)")
    else:
        title_str = "Chirp preview unavailable"

    ax_chirp.set_title(title_str, color="white", fontsize=10, pad=10)
    ax_chirp.set_xlabel("Time (ms)", color="#aaa")
    ax_chirp.set_ylabel("Amplitude", color="#aaa")
    ax_chirp.tick_params(colors="#aaa")
    for spine in ax_chirp.spines.values():
        spine.set_edgecolor("#333")
    ax_chirp.grid(color="#2a2a4a", linewidth=0.7)
    ax_chirp.set_ylim(-1.1, 1.1)

    fig.suptitle(
        "Acoustic Sonar  —  Frequency Response Auto-Calibration",
        color="white", fontsize=14, y=1.01, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 auto-calibrator for the acoustic sonar system"
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Re-plot the last calibration without playing audio"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-calibrate even if config.json already exists"
    )
    args = parser.parse_args()

    if args.plot_only:
        cfg = load_config()
        if cfg is None:
            print("No config.json found — run without --plot-only first.")
            sys.exit(1)
        print(f"Loaded config: {cfg.get('note', '')}")
        plot_results(cfg)
        return

    existing_cfg = load_config()
    if existing_cfg is not None and existing_cfg.get("calibrated") and not args.force:
        print(f"Config already exists ({existing_cfg.get('note', '')})")
        print("Use --force to re-calibrate, or --plot-only to view last results.")
        plot_results(existing_cfg)
        return

    print("=" * 60)
    print("  Acoustic Sonar  —  Phase 1 Auto-Calibration")
    print("=" * 60)

    cfg = run_calibration(verbose=True)
    write_config(cfg)

    print()
    print(f"  f_start  : {cfg['f_start']/1000:.1f} kHz")
    print(f"  f_end    : {cfg['f_end']/1000:.1f} kHz")
    print(f"  Bandwidth: {(cfg['f_end']-cfg['f_start'])/1000:.1f} kHz")
    print(f"  Status   : {'calibrated OK' if cfg['calibrated'] else 'fallback WARNING'}")
    print()
    print("Launching result plot... (close the window to exit)")
    plot_results(cfg)


if __name__ == "__main__":
    main()
