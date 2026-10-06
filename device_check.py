"""
device_check.py

Quick diagnostic: lists available audio devices, shows which ones are
currently set as default, and records 2 seconds of audio while you talk
or make noise, so you can confirm the mic is actually capturing normal-
amplitude signal before troubleshooting the chirp specifically.

Can also generate a hardware compatibility report.
Run with `python device_check.py --report` to append your laptop's
results to the compatibility table.
"""

import os
import sys
import platform
import argparse
import csv
import json
import sounddevice as sd
import numpy as np

# Allow importing config_loader from utils/
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from config_loader import load_sonar_config
    from audio_io import query_device_sample_rate
except ImportError:
    # If utils isn't set up yet, provide a dummy fallback
    def load_sonar_config(): return None
    def query_device_sample_rate(): return 48000


def run_diagnostic():
    print("=== Available audio devices ===")
    print(sd.query_devices())

    print("\n=== Current default devices ===")
    print(f"Default input device index: {sd.default.device[0]}")
    print(f"Default output device index: {sd.default.device[1]}")

    try:
        default_input_info = sd.query_devices(sd.default.device[0])
        print(f"Default input device name: {default_input_info['name']}")
    except Exception as e:
        print(f"Error querying default input device: {e}")

    sample_rate = query_device_sample_rate()
    print(f"\nUsing device sample rate: {sample_rate} Hz")

    print("\n=== Recording 2 seconds — talk, clap, or make noise now ===")
    duration_sec = 2.0
    try:
        recording = sd.rec(int(duration_sec * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        recording = recording.flatten()

        print(f"\nRecorded {recording.shape[0]} samples.")
        max_amp = np.max(np.abs(recording))
        print(f"Max absolute amplitude: {max_amp:.6f}")
        print(f"RMS amplitude: {np.sqrt(np.mean(recording**2)):.6f}")

        if max_amp < 0.001:
            print("\n[WARNING] Amplitude is still near-zero. The mic likely isn't being captured at all —")
            print("    check Windows Sound Settings > Input, confirm the right mic is selected")
            print("    and its volume/level isn't muted or set to 0.")
        else:
            print("\n[SUCCESS] Mic is capturing normal-range audio. The earlier near-zero amplitude was")
            print("    likely specific to the chirp frequency range, not a device issue.")
    except Exception as e:
        print(f"\n[ERROR] Failed to record audio: {e}")


def generate_report():
    print("Generating hardware compatibility report...")
    
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    
    # Try to get a more specific machine model
    machine_model = platform.node() # Fallback to hostname
    if platform.system() == "Windows":
        try:
            import subprocess
            output = subprocess.check_output("wmic csproduct get name", shell=True, text=True)
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            if len(lines) > 1 and lines[1] != "To be filled by O.E.M.":
                machine_model = lines[1]
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            import subprocess
            output = subprocess.check_output("sysctl -n hw.model", shell=True, text=True)
            machine_model = output.strip()
        except Exception:
            pass
            
    # Try to get audio device name
    audio_device = "Unknown"
    try:
        input_info = sd.query_devices(sd.default.device[0])
        audio_device = input_info.get("name", "Unknown")
    except Exception:
        pass

    # Load calibration info
    cfg = load_sonar_config()
    if cfg and cfg.get("calibrated"):
        band = f"{cfg['f_start']/1000:.1f}-{cfg['f_end']/1000:.1f} kHz"
        status = "Calibrated"
    else:
        band = "4.0-8.0 kHz (Fallback)"
        status = "Uncalibrated"
        
    sample_rate = cfg.get("sample_rate", "Unknown") if cfg else "Unknown"

    report_row = {
        "OS": os_info,
        "Machine Model": machine_model,
        "Audio Device": audio_device,
        "Sample Rate": f"{sample_rate} Hz",
        "Calibrated Band": band,
        "Status": status
    }
    
    csv_path = os.path.join(os.path.dirname(__file__), "compatibility_reports.csv")
    file_exists = os.path.exists(csv_path)
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=report_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(report_row)
        print(f"\nReport appended to {csv_path}")
    except Exception as e:
        print(f"\nFailed to write to CSV: {e}")
        
    print("\nShare this one-liner in the issue tracker:")
    print(f"| {report_row['Machine Model']} | {report_row['OS']} | {report_row['Audio Device']} | {report_row['Sample Rate']} | {report_row['Calibrated Band']} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Device check and compatibility reporting.")
    parser.add_argument("--report", action="store_true", help="Generate and append a compatibility report to CSV")
    args = parser.parse_args()

    if args.report:
        generate_report()
    else:
        run_diagnostic()
