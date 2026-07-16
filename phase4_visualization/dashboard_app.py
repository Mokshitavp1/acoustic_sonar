"""
dashboard_app.py

Phase 4 (optional, web-based alternative to the matplotlib views):
a minimal Flask + Flask-SocketIO app serving a single HTML page with a
canvas-based radar gauge and a live waveform chart (Chart.js via CDN),
updated in real time over WebSocket.

The backend currently emits dummy distance/motion/waveform data every
300ms so you can verify the frontend renders correctly before wiring in
the real Phase 3 sensing loop. Once confirmed working, replace
`generate_dummy_reading()` with a call into the real sensing loop's
live state.

Run with: python dashboard_app.py
Then open: http://localhost:5000
"""

import time
import threading

import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "acoustic-sonar-dev"  # fine for local dev only
socketio = SocketIO(app, cors_allowed_origins="*")

UPDATE_INTERVAL_SEC = 0.3
MAX_DISTANCE_M = 2.0

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Acoustic Sonar Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        body { font-family: sans-serif; background: #111; color: #eee; text-align: center; }
        #radarCanvas { background: #000; border-radius: 8px; margin-top: 20px; }
        #motionBanner { font-size: 24px; font-weight: bold; color: red; height: 32px; }
        #waveformChart { max-width: 700px; margin: 20px auto; }
    </style>
</head>
<body>
    <h1>Acoustic Sonar — Live Dashboard</h1>
    <div id="motionBanner"></div>
    <canvas id="radarCanvas" width="400" height="250"></canvas>
    <div id="distanceLabel" style="font-size: 20px;"></div>
    <div id="waveformChart">
        <canvas id="waveformCanvas"></canvas>
    </div>

    <script>
        const socket = io();
        const radarCanvas = document.getElementById('radarCanvas');
        const ctx = radarCanvas.getContext('2d');
        const cx = radarCanvas.width / 2;
        const cy = radarCanvas.height - 10;
        const radius = 180;
        const maxDistance = {{ max_distance }};

        function drawRadar(distanceM) {
            ctx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);
            // Arc background
            ctx.beginPath();
            ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
            ctx.strokeStyle = '#555';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Needle
            const fraction = Math.min(Math.max(distanceM / maxDistance, 0), 1);
            const angle = Math.PI * (1 - fraction); // PI (left) -> 0 (right)
            const nx = cx + radius * Math.cos(angle);
            const ny = cy - radius * Math.sin(angle);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(nx, ny);
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
            ctx.stroke();
        }

        const waveformChart = new Chart(document.getElementById('waveformCanvas'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Recorded echo',
                    data: [],
                    borderColor: 'orange',
                    borderWidth: 1,
                    pointRadius: 0,
                }]
            },
            options: {
                animation: false,
                scales: {
                    y: { min: -1, max: 1 }
                }
            }
        });

        socket.on('sonar_update', (data) => {
            drawRadar(data.distance_m);
            document.getElementById('distanceLabel').innerText =
                (data.distance_m * 100).toFixed(1) + ' cm';
            document.getElementById('motionBanner').innerText =
                data.motion ? 'MOTION DETECTED' : '';

            waveformChart.data.labels = data.waveform.map((_, i) => i);
            waveformChart.data.datasets[0].data = data.waveform;
            waveformChart.update();
        });
    </script>
</body>
</html>
"""


def generate_dummy_reading():
    """
    STUB: generates a fake sonar reading (distance, motion flag, and a
    short waveform snippet) for testing the dashboard in isolation.
    Replace this with a real read from the Phase 3 sensing loop's live
    state once the frontend is confirmed working.
    """
    t = time.time()
    distance_m = MAX_DISTANCE_M / 2 + 0.3 * np.sin(t)
    motion = np.random.rand() < 0.1
    waveform = (np.random.normal(scale=0.05, size=60)).tolist()
    return {
        "distance_m": distance_m,
        "motion": bool(motion),
        "waveform": waveform,
    }


def background_emitter():
    """Runs forever in a background thread, pushing a new reading to all
    connected clients every UPDATE_INTERVAL_SEC seconds."""
    while True:
        reading = generate_dummy_reading()
        socketio.emit("sonar_update", reading)
        time.sleep(UPDATE_INTERVAL_SEC)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE, max_distance=MAX_DISTANCE_M)


if __name__ == "__main__":
    emitter_thread = threading.Thread(target=background_emitter, daemon=True)
    emitter_thread.start()

    print("Starting dashboard at http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
