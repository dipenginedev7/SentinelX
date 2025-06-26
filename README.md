# SentinelX - AI Surveillance System

SentinelX is an advanced AI-powered surveillance and threat detection system. It leverages real-time video analytics, object detection (YOLOv5/YOLOv8), and a modern web dashboard to provide live monitoring, alerts, and statistics for enhanced security.

## Features

- **Live Surveillance Feed:** Real-time video streaming with AI-based detection.
- **Threat Detection:** Detects people, vehicles, and potential threats using YOLO models.
- **Live Alerts:** Instant alerts for suspicious activities, with threat levels and siren sound.
- **Statistics Dashboard:** Tracks people, vehicles, threats, and system uptime.
- **Modern UI:** Responsive, visually appealing dashboard with live updates via Socket.IO.
- **Alarm System:** Manual and automatic alarm triggers with audio-visual feedback.

## Project Structure

```
SentinelX/
├── sentinelx.py                # Main backend server (Flask/Socket.IO)
├── sentinelx_backend.py        # Additional backend logic
├── imageenhancer.py            # Image enhancement utilities
├── yolov5/                     # YOLOv5 model and scripts
│   ├── detect.py               # Object detection script
│   └── ...
├── static/
│   └── siren.mp3               # Siren sound for alerts
├── assets/
│   └── Arghyalogo.png          # Logo/icon assets
├── templates/
│   └── index.html              # Main web dashboard UI
├── config.json                 # Configuration file
├── *.db, *.log                 # Database and log files
├── requirements.txt            # Python dependencies (in yolov5/)
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA-enabled GPU for faster inference

### Installation
1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd SentinelX
   ```
2. **Install dependencies:**
   ```sh
   pip install -r yolov5/requirements.txt
   pip install flask flask-socketio opencv-python
   ```
3. **Download YOLO Weights:**
   - Place `yolov5s.pt` or `yolov8n.pt` in the project root or `yolov5/` directory.

### Running the Application
1. **Start the backend server:**
   ```sh
   python sentinelx.py
   ```
2. **Open the dashboard:**
   - Visit [http://localhost:5000](http://localhost:5000) in your browser.

### Demo Mode
- The UI includes demo controls for simulation if backend is not running.

## Customization
- **Cameras:** Configure camera sources in `config.json`.
- **Alerts:** Customize alert logic in `sentinelx.py` and frontend JS.
- **UI:** Edit `templates/index.html` and assets for branding.

## Technologies Used
- Python, Flask, Flask-SocketIO
- OpenCV, YOLOv5/YOLOv8
- HTML, CSS, JavaScript (Socket.IO)

## Credits
- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- UI/UX by Arghyadip

## License
This project is for educational and personal use. For commercial use, please contact the author.
