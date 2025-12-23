# utils.py - Utility functions for the traffic monitoring system

import cv2
import numpy as np
import torch
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import os
import json
from pathlib import Path

class ViolationZoneManager:
    """Manage violation detection zones"""
    
    def __init__(self):
        self.zones = {}
        self.zone_colors = {
            'no_parking': (0, 0, 255),      # Red
            'signal': (0, 255, 255),        # Yellow
            'wrong_side': (255, 0, 0),      # Blue
            'speed_limit': (255, 0, 255)    # Magenta
        }
    
    def add_zone(self, zone_name, coordinates, zone_type):
        """Add a new violation zone"""
        self.zones[zone_name] = {
            'coordinates': coordinates,
            'type': zone_type,
            'color': self.zone_colors.get(zone_type, (128, 128, 128))
        }
    
    def is_in_zone(self, point, zone_name):
        """Check if a point is inside a zone"""
        if zone_name not in self.zones:
            return False
        
        coords = self.zones[zone_name]['coordinates']
        x, y = point
        x1, y1, x2, y2 = coords[0][0], coords[0][1], coords[1][0], coords[1][1]
        
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def draw_zones(self, frame):
        """Draw all zones on the frame"""
        for zone_name, zone_data in self.zones.items():
            coords = zone_data['coordinates']
            color = zone_data['color']
            
            # Draw rectangle
            cv2.rectangle(frame, tuple(coords[0]), tuple(coords[1]), color, 2)
            
            # Add label
            cv2.putText(frame, zone_name.upper(), 
                       (coords[0][0], coords[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame

class SpeedCalculator:
    """Calculate vehicle speed using optical flow"""
    
    def __init__(self, pixels_per_meter=10, fps=30):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.prev_positions = {}
        self.speed_history = {}
    
    def calculate_speed(self, vehicle_id, current_position, current_frame):
        """Calculate speed of a vehicle"""
        if vehicle_id not in self.prev_positions:
            self.prev_positions[vehicle_id] = {
                'position': current_position,
                'frame': current_frame
            }
            return 0.0
        
        prev_data = self.prev_positions[vehicle_id]
        prev_position = prev_data['position']
        prev_frame = prev_data['frame']
        
        # Calculate distance in pixels
        distance_pixels = np.sqrt(
            (current_position[0] - prev_position[0])**2 + 
            (current_position[1] - prev_position[1])**2
        )
        
        # Convert to meters
        distance_meters = distance_pixels / self.pixels_per_meter
        
        # Calculate time difference
        time_diff = (current_frame - prev_frame) / self.fps
        
        if time_diff > 0:
            # Speed in m/s
            speed_ms = distance_meters / time_diff
            # Convert to km/h
            speed_kmh = speed_ms * 3.6
            
            # Update history
            if vehicle_id not in self.speed_history:
                self.speed_history[vehicle_id] = []
            
            self.speed_history[vehicle_id].append(speed_kmh)
            
            # Keep only last 10 readings for smoothing
            if len(self.speed_history[vehicle_id]) > 10:
                self.speed_history[vehicle_id] = self.speed_history[vehicle_id][-10:]
            
            # Return average speed
            avg_speed = np.mean(self.speed_history[vehicle_id])
            
            # Update previous position
            self.prev_positions[vehicle_id] = {
                'position': current_position,
                'frame': current_frame
            }
            
            return avg_speed
        
        return 0.0

class NotificationManager:
    """Handle email and SMS notifications"""
    
    def __init__(self, email_config):
        self.email_config = email_config
    
    def send_email_notification(self, violation_data, image_path=None):
        """Send email notification for violation"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = f"Traffic Violation Alert - {violation_data['type']}"
            
            # Email body
            body = f"""
Traffic Violation Detected

Violation Type: {violation_data['type']}
Vehicle Number: {violation_data.get('license_plate', 'Unknown')}
Fine Amount: Rs. {violation_data.get('fine', 0)}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: Camera {violation_data.get('camera_id', '001')}

This is an automated alert from the Traffic Monitoring System.
Please take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image if provided
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    img = MIMEImage(img_data)
                    img.add_header('Content-Disposition', 'attachment', 
                                 filename=f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    msg.attach(img)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            text = msg.as_string()
            server.sendmail(self.email_config['sender_email'], self.email_config['recipient_emails'], text)
            server.quit()
            
            print(f"Email notification sent for {violation_data['type']}")
            return True
            
        except Exception as e:
            print(f"Failed to send email notification: {e}")
            return False

class VehicleTracker:
    """Track vehicles across frames using centroid tracking"""
    
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """Update tracked objects with new detections"""
        if len(rects) == 0:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return {}
        
        # Initialize centroids array
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Get existing object centroids
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                # Update object position
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and tracklets
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                # More tracklets than detections
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # More detections than tracklets
                for col in unused_cols:
                    self.register(input_centroids[col])
        
        return self.objects

# real_time_processor.py - Real-time processing optimizations

import threading
import queue
import time
from collections import deque

class FrameBuffer:
    """Thread-safe frame buffer for real-time processing"""
    
    def __init__(self, maxsize=30):
        self.buffer = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def put_frame(self, frame):
        """Add frame to buffer"""
        try:
            self.buffer.put_nowait(frame)
            with self.lock:
                self.latest_frame = frame
        except queue.Full:
            # Remove oldest frame and add new one
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(frame)
                with self.lock:
                    self.latest_frame = frame
            except queue.Empty:
                pass
    
    def get_frame(self):
        """Get frame from buffer"""
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None
    
    def get_latest_frame(self):
        """Get the most recent frame"""
        with self.lock:
            return self.latest_frame

class RealTimeProcessor:
    """Real-time video processing with threading"""
    
    def __init__(self, detector, camera_id=0):
        self.detector = detector
        self.camera_id = camera_id
        self.cap = None
        self.frame_buffer = FrameBuffer()
        self.result_queue = queue.Queue()
        
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
    
    def start(self):
        """Start real-time processing"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("Real-time processor started")
    
    def stop(self):
        """Stop real-time processing"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        
        if self.process_thread:
            self.process_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
        
        print("Real-time processor stopped")
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.put_frame(frame)
                
                # Calculate FPS
                current_time = time.time()
                self.fps_counter.append(1.0 / (current_time - self.last_time))
                self.last_time = current_time
            else:
                time.sleep(0.01)  # Small delay if no frame
    
    def _process_frames(self):
        """Process frames in separate thread"""
        while self.is_running:
            frame = self.frame_buffer.get_frame()
            if frame is not None:
                try:
                    # Process frame for violations
                    violations = self.detector.process_frame(frame)
                    
                    # Put results in queue
                    if violations:
                        self.result_queue.put({
                            'frame': frame,
                            'violations': violations,
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    print(f"Processing error: {e}")
            else:
                time.sleep(0.01)  # Small delay if no frame to process
    
    def get_results(self):
        """Get processing results"""
        results = []
        try:
            while True:
                result = self.result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.fps_counter) > 0:
            return sum(self.fps_counter) / len(self.fps_counter)
        return 0.0
    
    def get_display_frame(self):
        """Get latest frame for display"""
        return self.frame_buffer.get_latest_frame()

# performance_monitor.py - Monitor system performance

import psutil
import threading
import time
import json
from datetime import datetime

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, log_file="performance.log"):
        self.log_file = log_file
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'fps': [],
            'processing_time': []
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self._save_metrics()
        print("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': memory.percent
                })
                
                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_load = gpus[0].load * 100
                        self.metrics['gpu_usage'].append({
                            'timestamp': datetime.now().isoformat(),
                            'value': gpu_load
                        })
                except ImportError:
                    pass  # GPU monitoring not available
                
                # Keep only last 100 measurements
                for key in self.metrics:
                    if len(self.metrics[key]) > 100:
                        self.metrics[key] = self.metrics[key][-100:]
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
    
    def add_fps_measurement(self, fps):
        """Add FPS measurement"""
        self.metrics['fps'].append({
            'timestamp': datetime.now().isoformat(),
            'value': fps
        })
    
    def add_processing_time(self, processing_time):
        """Add processing time measurement"""
        self.metrics['processing_time'].append({
            'timestamp': datetime.now().isoformat(),
            'value': processing_time
        })
    
    def _save_metrics(self):
        """Save metrics to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Failed to save metrics: {e}")
    
    def get_current_metrics(self):
        """Get current performance metrics"""
        current_metrics = {}
        
        for metric_name, measurements in self.metrics.items():
            if measurements:
                recent_values = [m['value'] for m in measurements[-10:]]  # Last 10 measurements
                current_metrics[metric_name] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'average': sum(recent_values) / len(recent_values) if recent_values else 0,
                    'max': max(recent_values) if recent_values else 0,
                    'min': min(recent_values) if recent_values else 0
                }
            else:
                current_metrics[metric_name] = {
                    'current': 0, 'average': 0, 'max': 0, 'min': 0
                }
        
        return current_metrics

# advanced_main.py - Enhanced main application with real-time capabilities

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from datetime import datetime
import json

class AdvancedTrafficMonitoringApp:
    """Advanced traffic monitoring application with real-time processing"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Traffic Monitoring System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Initialize components
        self.detector = None
        self.real_time_processor = None
        self.performance_monitor = PerformanceMonitor()
        self.zone_manager = ViolationZoneManager()
        self.notification_manager = None
        
        # UI variables
        self.is_monitoring = False
        self.current_frame = None
        self.violation_count = {'total': 0, 'today': 0}
        
        self.setup_advanced_gui()
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize system components"""
        try:
            from main import TrafficViolationDetector
            from config import Config
            
            self.detector = TrafficViolationDetector()
            self.notification_manager = NotificationManager(Config.EMAIL_CONFIG)
            
            # Setup default violation zones
            self.zone_manager.add_zone('no_parking', [(100, 100), (300, 200)], 'no_parking')
            self.zone_manager.add_zone('signal', [(400, 150), (600, 250)], 'signal')
            self.zone_manager.add_zone('wrong_side', [(50, 300), (250, 400)], 'wrong_side')
            
            self.log_message("System components initialized successfully")
            
        except Exception as e:
            self.log_message(f"Failed to initialize components: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize: {e}")
    
    def setup_advanced_gui(self):
        """Setup advanced GUI with modern design"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Header.TLabel', background='#34495e', foreground='white', font=('Arial', 12, 'bold'))
        style.configure('Status.TLabel', background='#2c3e50', foreground='#ecf0f1')
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#34495e', height=60)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚦 Advanced Traffic Monitoring System", 
                              bg='#34495e', fg='white', font=('Arial', 16, 'bold'))
        title_label.pack(pady=15)
        
        # Control panel
        self.setup_control_panel(main_container)
        
        # Main content area
        content_frame = tk.Frame(main_container, bg='#2c3e50')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video display
        self.setup_video_panel(content_frame)
        
        # Right panel - Controls and stats
        self.setup_info_panel(content_frame)
        
        # Bottom panel - Status and logs
        self.setup_status_panel(main_container)
    
    def setup_control_panel(self, parent):
        """Setup main control panel"""
        control_frame = tk.LabelFrame(parent, text="Control Panel", bg='#34495e', fg='white', 
                                     font=('Arial', 10, 'bold'))
        control_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Control buttons
        self.start_btn = tk.Button(button_frame, text="🎥 Start Camera", command=self.start_monitoring,
                                  bg='#27ae60', fg='white', font=('Arial', 10, 'bold'), width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop Camera", command=self.stop_monitoring,
                                 bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'), width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📁 Load Video", command=self.load_video,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="📊 View Reports", command=self.view_violations,
                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="⚙️ Settings", command=self.open_settings,
                 bg='#95a5a6', fg='white', font=('Arial', 10, 'bold'), width=12).pack(side=tk.LEFT, padx=5)
    
    def setup_video_panel(self, parent):
        """Setup video display panel"""
        video_frame = tk.LabelFrame(parent, text="Live Video Feed", bg='#34495e', fg='white',
                                   font=('Arial', 10, 'bold'))
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        self.video_label = tk.Label(video_frame, bg='black', text="No video feed",
                                   fg='white', font=('Arial', 12))
        self.video_label.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        
        # Video controls
        video_controls = tk.Frame(video_frame, bg='#34495e')
        video_controls.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.fps_label = tk.Label(video_controls, text="FPS: 0", bg='#34495e', fg='white')
        self.fps_label.pack(side=tk.LEFT)
        
        self.resolution_label = tk.Label(video_controls, text="Resolution: N/A", bg='#34495e', fg='white')
        self.resolution_label.pack(side=tk.RIGHT)
    
    def setup_info_panel(self, parent):
        """Setup information and statistics panel"""
        info_frame = tk.Frame(parent, bg='#2c3e50', width=350)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)
        
        # Statistics panel
        stats_frame = tk.LabelFrame(info_frame, text="Statistics", bg='#34495e', fg='white',
                                   font=('Arial', 10, 'bold'))
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_labels = {}
        stats_items = [
            ('Total Violations', 'total_violations'),
            ('Today\'s Violations', 'today_violations'),
            ('No Helmet', 'no_helmet'),
            ('Signal Violations', 'signal_violations'),
            ('Parking Violations', 'parking_violations'),
            ('Triple Riding', 'triple_riding')
        ]
        
        for i, (label_text, key) in enumerate(stats_items):
            frame = tk.Frame(stats_frame, bg='#34495e')
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(frame, text=f"{label_text}:", bg='#34495e', fg='white', 
                    font=('Arial', 9)).pack(side=tk.LEFT)
            
            value_label = tk.Label(frame, text="0", bg='#34495e', fg='#e74c3c',
                                  font=('Arial', 9, 'bold'))
            value_label.pack(side=tk.RIGHT)
            self.stats_labels[key] = value_label
        
        # Performance panel
        perf_frame = tk.LabelFrame(info_frame, text="Performance", bg='#34495e', fg='white',
                                  font=('Arial', 10, 'bold'))
        perf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.perf_labels = {}
        perf_items = [
            ('CPU Usage', 'cpu'),
            ('Memory Usage', 'memory'),
            ('GPU Usage', 'gpu'),
            ('Processing FPS', 'fps')
        ]
        
        for label_text, key in perf_items:
            frame = tk.Frame(perf_frame, bg='#34495e')
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(frame, text=f"{label_text}:", bg='#34495e', fg='white',
                    font=('Arial', 9)).pack(side=tk.LEFT)
            
            value_label = tk.Label(frame, text="0%", bg='#34495e', fg='#3498db',
                                  font=('Arial', 9, 'bold'))
            value_label.pack(side=tk.RIGHT)
            self.perf_labels[key] = value_label
        
        # Recent violations panel
        violations_frame = tk.LabelFrame(info_frame, text="Recent Violations", bg='#34495e', fg='white',
                                        font=('Arial', 10, 'bold'))
        violations_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.violations_listbox = tk.Listbox(violations_frame, bg='#2c3e50', fg='white',
                                           font=('Arial', 8), selectbackground='#3498db')
        scrollbar = tk.Scrollbar(violations_frame, orient=tk.VERTICAL)
        
        self.violations_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.violations_listbox.yview)
        
        self.violations_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
    
    def setup_status_panel(self, parent):
        """Setup status and logging panel"""
        status_frame = tk.LabelFrame(parent, text="System Status", bg='#34495e', fg='white',
                                    font=('Arial', 10, 'bold'), height=150)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        # Status text with scrollbar
        text_frame = tk.Frame(status_frame, bg='#34495e')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.status_text = tk.Text(text_frame, height=6, bg='#2c3e50', fg='#ecf0f1',
                                  font=('Courier', 9), wrap=tk.WORD)
        status_scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initial status
        self.log_message("Advanced Traffic Monitoring System initialized")
        self.log_message("Ready for operation...")
    
    def log_message(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, formatted_message)
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.is_monitoring:
            try:
                # Initialize real-time processor
                from utils import RealTimeProcessor
                self.real_time_processor = RealTimeProcessor(self.detector, camera_id=0)
                self.real_time_processor.start()
                
                # Start performance monitoring
                self.performance_monitor.start_monitoring()
                
                self.is_monitoring = True
                self.start_btn.config(state='disabled', bg='#95a5a6')
                self.stop_btn.config(state='normal', bg='#e74c3c')
                
                self.log_message("Real-time monitoring started")
                
                # Start update loop
                self.update_display()
                
            except Exception as e:
                self.log_message(f"Failed to start monitoring: {e}")
                messagebox.showerror("Error", f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            
            if self.real_time_processor:
                self.real_time_processor.stop()
            
            self.performance_monitor.stop_monitoring()
            
            self.start_btn.config(state='normal', bg='#27ae60')
            self.stop_btn.config(state='disabled', bg='#95a5a6')
            
            self.log_message("Real-time monitoring stopped")
    
    def update_display(self):
        """Update display with latest frame and information"""
        if self.is_monitoring and self.real_time_processor:
            try:
                # Get latest frame
                frame = self.real_time_processor.get_display_frame()
                if frame is not None:
                    # Draw violation zones
                    display_frame = self.zone_manager.draw_zones(frame.copy())
                    
                    # Resize for display
                    display_frame = cv2.resize(display_frame, (640, 480))
                    
                    # Convert to RGB for tkinter
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb_frame)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo
                
                # Update FPS
                fps = self.real_time_processor.get_fps()
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                
                # Process violations
                results = self.real_time_processor.get_results()
                for result in results:
                    self.process_violations(result['violations'], result['frame'])
                
                # Update performance metrics
                self.update_performance_display()
                
                # Schedule next update
                if self.is_monitoring:
                    self.root.after(50, self.update_display)  # 20 FPS display update
                    
            except Exception as e:
                self.log_message(f"Display update error: {e}")
    
    def process_violations(self, violations, frame):
        """Process detected violations"""
        for violation in violations:
            # Save violation image
            if self.detector:
                image_path = self.detector.save_violation_image(frame, violation)
                
                # Add to database
                self.detector.db_manager.add_violation(
                    violation['type'],
                    violation.get('license_plate', 'UNKNOWN'),
                    violation.get('fine', 0),
                    image_path
                )
            
            # Update statistics
            self.update_statistics(violation['type'])
            
            # Add to recent violations list
            violation_text = f"{violation['type'].upper()} - {violation.get('license_plate', 'UNKNOWN')} - Rs.{violation.get('fine', 0)}"
            self.violations_listbox.insert(0, violation_text)
            
            # Keep only last 20 violations in display
            if self.violations_listbox.size() > 20:
                self.violations_listbox.delete(self.violations_listbox.size()-1)
            
            # Log violation
            self.log_message(f"VIOLATION: {violation['message']}")
            
            # Send notification
            if self.notification_manager:
                threading.Thread(target=self.notification_manager.send_email_notification,
                               args=(violation, image_path), daemon=True).start()
    
    def update_statistics(self, violation_type):
        """Update violation statistics"""
        self.violation_count['total'] += 1
        self.violation_count['today'] += 1
        
        # Update display
        self.stats_labels['total_violations'].config(text=str(self.violation_count['total']))
        self.stats_labels['today_violations'].config(text=str(self.violation_count['today']))
        
        # Update specific violation type
        if violation_type in self.stats_labels:
            current = int(self.stats_labels[violation_type].cget('text'))
            self.stats_labels[violation_type].config(text=str(current + 1))
    
    def update_performance_display(self):
        """Update performance metrics display"""
        try:
            metrics = self.performance_monitor.get_current_metrics()
            
            self.perf_labels['cpu'].config(text=f"{metrics['cpu_usage']['current']:.1f}%")
            self.perf_labels['memory'].config(text=f"{metrics['memory_usage']['current']:.1f}%")
            
            if 'gpu_usage' in metrics and metrics['gpu_usage']['current'] > 0:
                self.perf_labels['gpu'].config(text=f"{metrics['gpu_usage']['current']:.1f}%")
            
            if 'fps' in metrics and metrics['fps']['current'] > 0:
                self.perf_labels['fps'].config(text=f"{metrics['fps']['current']:.1f}")
                
        except Exception as e:
            pass  # Silently handle performance update errors
    
    def load_video(self):
        """Load and process video file"""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        )
        
        if video_path:
            self.log_message(f"Loading video: {video_path}")
            # Process video in separate thread
            threading.Thread(target=self.process_video_file, args=(video_path,), daemon=True).start()
    
    def process_video_file(self, video_path):
        """Process video file for violations"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_message("Failed to open video file")
            return
        
        frame_count = 0
        violations_found = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                try:
                    violations = self.detector.process_frame(frame)
                    if violations:
                        violations_found += len(violations)
                        self.process_violations(violations, frame)
                        
                except Exception as e:
                    self.log_message(f"Video processing error: {e}")
        
        cap.release()
        self.log_message(f"Video processing completed. Frames: {frame_count}, Violations: {violations_found}")
    
    def view_violations(self):
        """Open violations report window"""
        # This would open a detailed violations report window
        # Implementation similar to the previous violation viewer
        pass
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("600x400")
        settings_window.configure(bg='#2c3e50')
        
        # Settings content would go here
        tk.Label(settings_window, text="Settings panel - Configure zones, thresholds, etc.",
                bg='#2c3e50', fg='white', font=('Arial', 12)).pack(pady=50)
    
    def run(self):
        """Start the advanced application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.cleanup()
            self.root.destroy()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.is_monitoring:
            self.stop_monitoring()
        
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()

if __name__ == "__main__":
    app = AdvancedTrafficMonitoringApp()
    app.run()