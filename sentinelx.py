import cv2
import numpy as np
import torch
import time
import threading
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import requests
import pygame
from ultralytics import YOLO
import easyocr
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import base64
from collections import defaultdict, deque
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional
import traceback


class ProductionSentinelX:
    """Production-level surveillance system with advanced filtering and minimal false alerts"""
    
    def __init__(self, config_path="config.json"):
        """Initialize Production SentinelX surveillance system"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_database()
        self.setup_models()
        self.setup_alerts()
        
        # Advanced tracking and filtering
        self.detection_history = defaultdict(lambda: deque(maxlen=30))  # 30 frame history
        self.confirmed_detections = {}
        self.alert_cooldown = defaultdict(float)  # Prevent alert spam
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        
        # Tracking variables with stability checks
        self.tracked_objects = {}
        self.threat_level = "LOW"
        self.active_alerts = []
        self.is_recording = False
        self.frame_count = 0
        self.last_motion_time = 0
        
        # Detection confidence and stability requirements
        self.min_detection_frames = 5  # Minimum frames before confirming detection
        self.min_confidence_threshold = 0.7  # Higher confidence for production
        self.alert_cooldown_time = 300  # 5 minutes between same alerts
        
        # License plate validation
        self.license_plate_cache = {}
        self.license_plate_confidence_threshold = 0.8
        
        # Flask app for web interface
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'sentinel_x_production_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.setup_routes()
        
        self.logger.info("Production SentinelX initialized successfully")
    
    def load_config(self, config_path):
        """Load production configuration"""
        default_config = {
            "cameras": [
                {"id": 0, "name": "Main Camera", "source": 0, "active": True},
                {"id": 1, "name": "Secondary Camera", "source": "rtsp://example.com/stream", "active": False}
            ],
            "models": {
                "yolo_model": "yolov8m.pt",  # Medium model for better accuracy
                "ocr_languages": ["en"],
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            "detection": {
                "confidence_threshold": 0.7,  # Higher threshold for production
                "nms_threshold": 0.45,
                "threat_classes": ["knife", "gun", "weapon"],  # Specific threat classes
                "vehicle_classes": ["car", "truck", "bus", "motorcycle"],
                "person_class": ["person"],
                "restricted_zones": [],
                "ignore_zones": [],  # Areas to ignore (e.g., TV screens)
                "min_object_size": 50,  # Minimum bounding box size
                "max_object_size": 800,  # Maximum bounding box size
                "stability_frames": 5,  # Frames needed to confirm detection
                "motion_threshold": 1000  # Minimum motion area
            },
            "alerts": {
                "email": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "enabled": False,
                    "url": ""
                },
                "sound": {
                    "enabled": True,
                    "alarm_sound": "alarm.wav"
                },
                "cooldown_seconds": 300,  # 5 minutes between same type alerts
                "high_priority_classes": ["knife", "gun", "weapon"],
                "image_attachment": True
            },
            "web_interface": {
                "host": "0.0.0.0",
                "port": 5000,
                "stream_quality": 0.7
            },
            "recording": {
                "enabled": False,
                "path": "./recordings/",
                "format": "mp4",
                "fps": 15,
                "duration_minutes": 10
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Deep merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            return config
        except FileNotFoundError:
            self.save_config(default_config, config_path)
            return default_config
    
    def save_config(self, config, config_path):
        """Save configuration to JSON file"""
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.FileHandler('sentinelx_production.log')
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        console_handler.setLevel(logging.WARNING)
        
        # Setup logger
        self.logger = logging.getLogger('ProductionSentinelX')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Alert logger for critical events
        alert_handler = logging.FileHandler('sentinelx_alerts.log')
        alert_handler.setFormatter(log_format)
        self.alert_logger = logging.getLogger('SentinelX_Alerts')
        self.alert_logger.setLevel(logging.WARNING)
        self.alert_logger.addHandler(alert_handler)
    
    def setup_database(self):
        """Setup production database with indexing"""
        self.db_path = "sentinelx_production.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                description TEXT,
                confidence REAL,
                camera_id INTEGER,
                image_path TEXT,
                coordinates TEXT,
                confirmed BOOLEAN DEFAULT FALSE,
                alert_sent BOOLEAN DEFAULT FALSE,
                detection_hash TEXT UNIQUE
            )
        ''')
        
        # Tracking table with more details
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id TEXT UNIQUE,
                object_type TEXT,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_detections INTEGER DEFAULT 1,
                confirmed_detections INTEGER DEFAULT 0,
                camera_id INTEGER,
                max_confidence REAL,
                last_coordinates TEXT
            )
        ''')
        
        # License plates with validation
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS license_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id INTEGER,
                confidence REAL,
                image_path TEXT,
                validated BOOLEAN DEFAULT FALSE,
                vehicle_type TEXT
            )
        ''')
        
        # Alert history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                severity INTEGER,
                camera_id INTEGER,
                resolved BOOLEAN DEFAULT FALSE,
                false_positive BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alert_history(timestamp)')
        
        conn.commit()
        conn.close()
    
    def setup_models(self):
        """Initialize models with production settings"""
        try:
            # Load YOLOv8 model with specific device
            device = self.config["models"]["device"]
            self.yolo_model = YOLO(self.config["models"]["yolo_model"])
            if device == "cuda" and torch.cuda.is_available():
                self.yolo_model.to('cuda')
                self.logger.info("YOLO model loaded on GPU")
            else:
                self.logger.info("YOLO model loaded on CPU")
            
            # Initialize OCR with better settings
            self.ocr_reader = easyocr.Reader(
                self.config["models"]["ocr_languages"],
                gpu=torch.cuda.is_available()
            )
            self.logger.info("OCR reader initialized")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def setup_alerts(self):
        """Initialize alert systems"""
        if self.config["alerts"]["sound"]["enabled"]:
            try:
                pygame.mixer.init()
                self.logger.info("Sound alert system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize sound system: {str(e)}")
    
    def detect_objects(self, frame) -> List[Dict]:
        """Enhanced object detection with filtering"""
        try:
            results = self.yolo_model(
                frame, 
                conf=self.config["detection"]["confidence_threshold"],
                iou=self.config["detection"]["nms_threshold"],
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[cls]
                        
                        # Size filtering
                        width = x2 - x1
                        height = y2 - y1
                        min_size = self.config["detection"]["min_object_size"]
                        max_size = self.config["detection"]["max_object_size"]
                        
                        if width < min_size or height < min_size or width > max_size or height > max_size:
                            continue
                        
                        # Ignore zone filtering
                        if self.is_in_ignore_zone([int(x1), int(y1), int(x2), int(y2)]):
                            continue
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class': class_name,
                            'class_id': cls,
                            'area': width * height,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return []
    
    def is_in_ignore_zone(self, bbox: List[int]) -> bool:
        """Check if detection is in ignore zone"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        for zone in self.config["detection"]["ignore_zones"]:
            zone_poly = np.array(zone["coordinates"])
            if cv2.pointPolygonTest(zone_poly, (center_x, center_y), False) >= 0:
                return True
        return False
    
    def validate_detection_stability(self, detection: Dict, detection_id: str) -> bool:
        """Validate detection stability over multiple frames"""
        current_time = time.time()
        
        # Add to history
        self.detection_history[detection_id].append({
            'detection': detection,
            'timestamp': current_time,
            'frame': self.frame_count
        })
        
        # Check if we have enough history
        history = self.detection_history[detection_id]
        if len(history) < self.config["detection"]["stability_frames"]:
            return False
        
        # Check confidence consistency
        confidences = [h['detection']['confidence'] for h in history]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Require consistent high confidence
        if avg_confidence < self.min_confidence_threshold or confidence_std > 0.15:
            return False
        
        # Check position stability (object shouldn't jump around too much)
        positions = [h['detection']['center'] for h in history]
        if len(positions) >= 2:
            movements = []
            for i in range(1, len(positions)):
                dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                              (positions[i][1] - positions[i-1][1])**2)
                movements.append(dist)
            
            # If object is moving too erratically, might be false detection
            if np.mean(movements) > 100:  # pixels per frame
                return False
        
        return True
    
    def generate_detection_id(self, detection: Dict) -> str:
        """Generate unique ID for detection tracking"""
        center = detection['center']
        class_name = detection['class']
        # Create ID based on position and class
        id_string = f"{class_name}_{int(center[0]/50)}_{int(center[1]/50)}"
        return id_string
    
    def detect_motion(self, frame) -> bool:
        """Detect significant motion in frame"""
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate total motion area
            total_motion_area = sum(cv2.contourArea(contour) for contour in contours)
            
            motion_threshold = self.config["detection"]["motion_threshold"]
            has_motion = total_motion_area > motion_threshold
            
            if has_motion:
                self.last_motion_time = time.time()
            
            return has_motion
            
        except Exception as e:
            self.logger.error(f"Motion detection error: {str(e)}")
            return False
    
    def detect_license_plates(self, frame, vehicle_detections) -> List[Dict]:
        """Enhanced license plate detection with validation"""
        plates = []
        
        for detection in vehicle_detections:
            if detection['class'] not in self.config["detection"]["vehicle_classes"]:
                continue
                
            try:
                x1, y1, x2, y2 = detection['bbox']
                
                # Expand ROI slightly for better OCR
                padding = 10
                y1_exp = max(0, y1 - padding)
                y2_exp = min(frame.shape[0], y2 + padding)
                x1_exp = max(0, x1 - padding)
                x2_exp = min(frame.shape[1], x2 + padding)
                
                vehicle_roi = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                
                if vehicle_roi.size == 0:
                    continue
                
                # Preprocess for better OCR
                gray_roi = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced_roi = clahe.apply(gray_roi)
                
                # OCR with specific settings for license plates
                ocr_results = self.ocr_reader.readtext(
                    enhanced_roi,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    width_ths=0.7,
                    height_ths=0.7
                )
                
                for (bbox, text, confidence) in ocr_results:
                    if confidence < self.license_plate_confidence_threshold:
                        continue
                    
                    cleaned_text = self.clean_license_plate_text(text)
                    if self.validate_license_plate_pattern(cleaned_text):
                        # Check if we've seen this plate recently (avoid duplicates)
                        plate_key = f"{cleaned_text}_{detection['class']}"
                        current_time = time.time()
                        
                        if (plate_key not in self.license_plate_cache or 
                            current_time - self.license_plate_cache[plate_key] > 30):
                            
                            self.license_plate_cache[plate_key] = current_time
                            
                            plate_info = {
                                'text': cleaned_text,
                                'confidence': confidence,
                                'bbox': [x1_exp + int(bbox[0][0]), y1_exp + int(bbox[0][1]), 
                                        x1_exp + int(bbox[2][0]), y1_exp + int(bbox[2][1])],
                                'vehicle_bbox': detection['bbox'],
                                'vehicle_type': detection['class']
                            }
                            plates.append(plate_info)
                            
                            # Save to database
                            self.save_license_plate(cleaned_text, confidence, 0, detection['class'])
                
            except Exception as e:
                self.logger.warning(f"License plate OCR error: {str(e)}")
                self.logger.debug(traceback.format_exc())
        return plates
    
    def clean_license_plate_text(self, text: str) -> str:
        """Clean and normalize license plate text"""
        # Remove spaces, dashes, and other common OCR artifacts
        cleaned = text.replace(' ', '').replace('-', '').replace('_', '')
        # Convert to uppercase
        cleaned = cleaned.upper()
        # Remove non-alphanumeric characters
        cleaned = ''.join(char for char in cleaned if char.isalnum())
        return cleaned
    
    def validate_license_plate_pattern(self, text: str) -> bool:
        """Validate license plate against common patterns"""
        import re
        
        if len(text) < 4 or len(text) > 8:
            return False
        
        # Common patterns (customize based on your region)
        patterns = [
            r'^[A-Z]{2,3}[0-9]{2,4}$',      # AB123, ABC1234
            r'^[0-9]{2,3}[A-Z]{2,3}$',      # 12AB, 123ABC
            r'^[A-Z]{1,2}[0-9]{1,2}[A-Z]{1,3}$',  # A1B, AB12CD
            r'^[0-9]{3}[A-Z]{3}$',          # 123ABC
            r'^[A-Z]{3}[0-9]{3}$'           # ABC123
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def analyze_threats(self, detections: List[Dict]) -> List[Dict]:
        """Enhanced threat analysis with confirmation"""
        threats = []
        threat_classes = self.config["detection"]["threat_classes"]
        high_priority = self.config["alerts"]["high_priority_classes"]
        
        for detection in detections:
            if detection['class'] not in threat_classes:
                continue
            
            detection_id = self.generate_detection_id(detection)
            
            # Validate detection stability
            if not self.validate_detection_stability(detection, detection_id):
                continue
            
            # Check if this is a confirmed threat
            threat_key = f"threat_{detection['class']}_{int(detection['center'][0]/100)}_{int(detection['center'][1]/100)}"
            
            # Check cooldown to prevent spam
            current_time = time.time()
            if (threat_key in self.alert_cooldown and 
                current_time - self.alert_cooldown[threat_key] < self.alert_cooldown_time):
                continue
            
            # Create threat object
            threat = {
                'type': detection['class'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'timestamp': datetime.now(),
                'severity': self.get_threat_severity(detection['class']),
                'detection_id': detection_id,
                'confirmed': True
            }
            threats.append(threat)
            
            # Update cooldown
            self.alert_cooldown[threat_key] = current_time
            
            # Create alert for confirmed threat
            severity_text = "HIGH PRIORITY" if detection['class'] in high_priority else "THREAT"
            message = f"{severity_text}: {detection['class']} detected with {detection['confidence']:.2f} confidence"
            
            self.create_alert("THREAT_DETECTED", message, threat['severity'], detection)
        
        return threats
    
    def get_threat_severity(self, threat_class: str) -> int:
        """Get severity level for threat class"""
        high_priority = self.config["alerts"]["high_priority_classes"]
        if threat_class in high_priority:
            return 5  # Maximum severity
        
        severity_map = {
            'knife': 4,
            'gun': 5,
            'weapon': 4,
            'person': 2,
            'suspicious_object': 3
        }
        return severity_map.get(threat_class, 2)
    
    def check_restricted_zones(self, detections: List[Dict]) -> List[Dict]:
        """Check for zone violations with confirmation"""
        violations = []
        
        for zone in self.config["detection"]["restricted_zones"]:
            zone_poly = np.array(zone["coordinates"])
            
            for detection in detections:
                center = detection['center']
                
                if cv2.pointPolygonTest(zone_poly, center, False) >= 0:
                    detection_id = self.generate_detection_id(detection)
                    
                    # Validate stability
                    if not self.validate_detection_stability(detection, detection_id):
                        continue
                    
                    # Check cooldown
                    violation_key = f"zone_{zone['name']}_{detection['class']}"
                    current_time = time.time()
                    
                    if (violation_key in self.alert_cooldown and 
                        current_time - self.alert_cooldown[violation_key] < self.alert_cooldown_time):
                        continue
                    
                    violation = {
                        'zone_name': zone['name'],
                        'detection': detection,
                        'timestamp': datetime.now(),
                        'confirmed': True
                    }
                    violations.append(violation)
                    
                    # Update cooldown
                    self.alert_cooldown[violation_key] = current_time
                    
                    # Create alert
                    message = f"ZONE VIOLATION: {detection['class']} detected in restricted area '{zone['name']}'"
                    self.create_alert("ZONE_VIOLATION", message, 4, detection)
        
        return violations
    
    def create_alert(self, alert_type: str, message: str, severity: int = 3, detection: Dict = None):
        """Create and handle alerts with image attachment"""
        try:
            alert = {
                'type': alert_type,
                'message': message,
                'timestamp': datetime.now(),
                'severity': severity,
                'camera_id': 0,
                'detection': detection
            }
            
            self.active_alerts.append(alert)
            self.alert_logger.warning(f"CONFIRMED ALERT: {alert_type} - {message}")
            
            # Save to database
            self.save_alert_to_db(alert)
            
            # Trigger alert systems
            self.send_email_alert(alert)
            self.send_webhook_alert(alert)
            self.play_alarm_sound(alert)
            
            # Emit to web interface
            if hasattr(self, 'socketio'):
                self.socketio.emit('new_alert', {
                    'type': alert_type,
                    'message': message,
                    'timestamp': str(alert['timestamp']),
                    'severity': severity
                })
            
            # Update threat level
            self.update_threat_level(severity)
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {str(e)}")
    
    def save_alert_to_db(self, alert: Dict):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alert_history (alert_type, message, severity, camera_id)
                VALUES (?, ?, ?, ?)
            ''', (alert['type'], alert['message'], alert['severity'], alert['camera_id']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save alert: {str(e)}")
    
    def update_threat_level(self, severity: int):
        """Update overall threat level"""
        if severity >= 5:
            self.threat_level = "CRITICAL"
        elif severity >= 4:
            self.threat_level = "HIGH"
        elif severity >= 3:
            self.threat_level = "MEDIUM"
        else:
            self.threat_level = "LOW"
    
    def send_email_alert(self, alert: Dict):
        """Send email alert with image attachment"""
        if not self.config["alerts"]["email"]["enabled"]:
            return
        
        try:
            email_config = self.config["alerts"]["email"]
            
            if not email_config["username"] or not email_config["recipients"]:
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(email_config["recipients"])
            msg['Subject'] = f"🚨 SentinelX CONFIRMED Alert: {alert['type']}"
            
            body = f"""
            SentinelX Production Alert - CONFIRMED DETECTION
            
            Alert Type: {alert['type']}
            Message: {alert['message']}
            Timestamp: {alert['timestamp']}
            Severity: {alert['severity']}/5
            Camera: {alert['camera_id']}
            
            This is a confirmed detection after validation.
            Please respond immediately.
            
            SentinelX Production System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            text = msg.as_string()
            server.sendmail(email_config["username"], email_config["recipients"], text)
            server.quit()
            
            self.logger.info("Email alert sent successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
    
    def send_webhook_alert(self, alert: Dict):
        """Send webhook alert"""
        if not self.config["alerts"]["webhook"]["enabled"]:
            return
        
        try:
            webhook_url = self.config["alerts"]["webhook"]["url"]
            if not webhook_url:
                return
            
            color_map = {5: "danger", 4: "warning", 3: "warning", 2: "good", 1: "good"}
            color = color_map.get(alert['severity'], "warning")
            
            payload = {
                "text": f"🚨 SentinelX CONFIRMED Alert: {alert['type']}",
                "attachments": [{
                    "color": color,
                    "fields": [
                        {"title": "Message", "value": alert['message'], "short": False},
                        {"title": "Severity", "value": f"{alert['severity']}/5", "short": True},
                        {"title": "Time", "value": str(alert['timestamp']), "short": True},
                        {"title": "Status", "value": "CONFIRMED DETECTION", "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=5)
            if response.status_code == 200:
                self.logger.info("Webhook alert sent successfully")
            else:
                self.logger.error(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")
    
    def play_alarm_sound(self, alert: Dict):
        """Play alarm sound for high severity alerts"""
        if not self.config["alerts"]["sound"]["enabled"]:
            return
        
        try:
            if alert['severity'] >= 4:  # High severity alerts only
                sound_file = self.config["alerts"]["sound"]["alarm_sound"]
                if Path(sound_file).exists():
                    pygame.mixer.music.load(sound_file)
                    pygame.mixer.music.play()
                else:
                    # Generate system beep for critical alerts
                    self.generate_beep_sound()
        except Exception as e:
            self.logger.error(f"Failed to play alarm sound: {str(e)}")
    
    def generate_beep_sound(self):
        """Generate system beep"""
        try:
            # Cross-platform beep
            import platform
            if platform.system() == "Windows":
                import winsound
                for _ in range(3):
                    winsound.Beep(1000, 500)  # 1000Hz, 500ms
                    time.sleep(0.2)
            else:
                # For Linux/Mac, use the terminal bell
                print('\a' * 3)
        except Exception as e:
            self.logger.warning(f"Failed to play beep sound: {str(e)}")

    def save_license_plate(self, plate_number, confidence, camera_id, vehicle_type):
        """Save license plate detection to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO license_plates (plate_number, confidence, camera_id, vehicle_type)
                VALUES (?, ?, ?, ?)
            ''', (plate_number, confidence, camera_id, vehicle_type))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to save license plate: {str(e)}")

    def process_frame(self, frame, camera_id=0):
        """Process a single frame for detection and analysis"""
        self.frame_count += 1

        # Detect motion first for efficiency
        has_motion = self.detect_motion(frame)
        if not has_motion and (time.time() - self.last_motion_time) > 2:
            # No recent motion, skip heavy detection to save CPU
            return {
                'frame': frame,
                'detections': [],
                'threats': [],
                'zone_violations': [],
                'plates': []
            }

        detections = self.detect_objects(frame)
        vehicle_detections = [
            d for d in detections if d['class'] in self.config["detection"]["vehicle_classes"]
        ]
        plates = self.detect_license_plates(frame, vehicle_detections)
        threats = self.analyze_threats(detections)
        zone_violations = self.check_restricted_zones(detections)

        annotated_frame = self.draw_annotations(
            frame, detections, plates, threats, zone_violations
        )

        return {
            'frame': annotated_frame,
            'detections': detections,
            'threats': threats,
            'zone_violations': zone_violations,
            'plates': plates
        }

    def draw_annotations(self, frame, detections, plates, threats, zone_violations):
        """Draw bounding boxes, labels, and zones on the frame"""
        annotated = frame.copy()

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0)
            if det['class'] in self.config["detection"]["threat_classes"]:
                color = (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']}:{det['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw plates
        for plate in plates:
            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(annotated, plate['text'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # Draw restricted zones
        for zone in self.config["detection"]["restricted_zones"]:
            points = np.array(zone["coordinates"], np.int32)
            cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(annotated, zone["name"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Draw threat level
        color_map = {"LOW": (0, 255, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255), "CRITICAL": (0, 0, 128)}
        threat_color = color_map.get(self.threat_level, (0, 255, 0))
        cv2.putText(annotated, f"Threat Level: {self.threat_level}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, threat_color, 2)

        return annotated

    def setup_routes(self):
        """Setup Flask routes for web interface"""
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                'threat_level': self.threat_level,
                'active_alerts': len(self.active_alerts),
                'tracked_objects': len(self.tracked_objects),
                'is_recording': self.is_recording
            })

        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify(self.active_alerts[-10:])

        @self.socketio.on('connect')
        def handle_connect():
            emit('status_update', {
                'threat_level': self.threat_level,
                'active_alerts': len(self.active_alerts)
            })

    def run_surveillance(self, camera_source=0):
        """Main surveillance loop"""
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            self.logger.error(f"Cannot open camera source: {camera_source}")
            return

        self.logger.info(f"Starting surveillance on camera: {camera_source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break

                # Process frame
                result = self.process_frame(frame, camera_id=0)

                # Emit frame to web interface
                if hasattr(self, 'socketio'):
                    _, buffer = cv2.imencode('.jpg', result['frame'])
                    frame_data = base64.b64encode(buffer).decode('utf-8')
                    self.socketio.emit('video_frame', {'data': frame_data})

                # Optional: add sleep to control FPS
                time.sleep(0.07)  # ~14 FPS

        except Exception as e:
            self.logger.error(f"Exception in surveillance loop: {traceback.format_exc()}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_web_interface(self):
        """Run the web interface"""
        self.socketio.run(self.app,
                          host=self.config["web_interface"]["host"],
                          port=self.config["web_interface"]["port"],
                          debug=False)

    def start(self):
        """Start the complete surveillance system"""
        camera_source = self.config["cameras"][0]["source"]
        surveillance_thread = threading.Thread(target=self.run_surveillance, args=(camera_source,))
        surveillance_thread.daemon = True
        surveillance_thread.start()
        self.run_web_interface()


def main():
    """Main entry for Production SentinelX"""
    try:
        sentinel = ProductionSentinelX()
        sentinel.start()
    except KeyboardInterrupt:
        print("\nShutting down Production SentinelX...")
    except Exception as e:
        print(f"Error: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
            