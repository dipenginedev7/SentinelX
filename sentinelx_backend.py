import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.append("yolov5")  # Add yolov5 folder to path

import torch
import cv2
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentinel_x_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLOv5 model (YOLOv5s.pt)
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
model.eval()

# Classes to track
TARGET_CLASSES = ['person', 'car', 'truck']

@app.route('/')
def index():
    return render_template('index.html')

def detect_threat_yolo(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        cls_name = row['name']
        conf = row['confidence']
        if cls_name in TARGET_CLASSES and conf > 0.5:
            return {
                'type': 'high' if cls_name == 'truck' else 'medium',
                'message': f'{cls_name.capitalize()} detected ({conf:.2f})'
            }
    return None

def video_stream(source=0):
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800, 450))

        # YOLOv5 detection
        results = model(frame)
        detections = results.pandas().xyxy[0]

        # Real-time stats
        person_count = len(detections[detections['name'] == 'person'])
        vehicle_count = len(detections[detections['name'].isin(['car', 'truck', 'bus'])])
        threat_count = len(detections[detections['confidence'] > 0.5])

        # Emit stats to frontend
        socketio.emit('stats_update', {
            'people': person_count,
            'vehicles': vehicle_count,
            'threats': threat_count
        })

        # Emit alert if any target detected
        alert = detect_threat_yolo(frame)
        if alert:
            socketio.emit('alert', alert)

        # Encode frame and emit as base64
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'data': jpg_as_text})

        socketio.sleep(0.1)

    cap.release()

if __name__ == '__main__':
    socketio.start_background_task(target=video_stream, source=0)
    socketio.run(app, host='0.0.0.0', port=5000)
