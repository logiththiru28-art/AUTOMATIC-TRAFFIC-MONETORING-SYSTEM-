# Complete Automated Traffic Monitoring System
# main.py

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
import easyocr
from datetime import datetime
import os
import json
import tkinter as tk
from tkinter import messagebox, ttk
import threading
from PIL import Image, ImageTk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3

# Helmet Detection CNN Model
class HelmetDetector(nn.Module):
    def __init__(self):
        super(HelmetDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Input: 224x224 -> after 4 pools: 14x14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # helmet, no_helmet
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Database Manager
class DatabaseManager:
    def __init__(self, db_path="traffic_violations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                violation_type TEXT,
                vehicle_number TEXT,
                fine_amount INTEGER,
                image_path TEXT,
                status TEXT DEFAULT 'PENDING'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_violation(self, violation_type, vehicle_number, fine_amount, image_path):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO violations (timestamp, violation_type, vehicle_number, fine_amount, image_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), violation_type, vehicle_number, fine_amount, image_path))
        
        conn.commit()
        conn.close()
    
    def get_violations(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM violations ORDER BY timestamp DESC')
        violations = cursor.fetchall()
        
        conn.close()
        return violations

# Traffic Violation Detector
class TrafficViolationDetector:
    def __init__(self):
        # Initialize models
        self.yolo_model = YOLO('yolov8n.pt')  # You can use yolov8s.pt for better accuracy
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Initialize helmet detector
        self.helmet_model = HelmetDetector()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.helmet_model.to(self.device)
        
        # Load trained weights if available
        try:
            self.helmet_model.load_state_dict(torch.load('helmet_model.pth', map_location=self.device))
            self.helmet_model.eval()
            print("Loaded pre-trained helmet model")
        except:
            print("No pre-trained helmet model found. Training required.")
        
        # Initialize database
        self.db_manager = DatabaseManager()
        
        # Violation thresholds
        self.violation_zones = {
            'no_parking': [(100, 100), (300, 200)],  # Define no parking zones
            'signal': [(400, 150), (600, 250)]       # Define signal zones
        }
        
        # Fine amounts
        self.fines = {
            'no_helmet': 200,
            'triple_riding': 500,
            'no_parking': 300,
            'signal_violation': 1000,
            'wrong_side': 750
        }
        
        # Track vehicles across frames
        self.tracked_vehicles = {}
        self.violation_cooldown = {}
    
    def preprocess_for_helmet(self, image):
        """Preprocess image for helmet detection"""
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return torch.FloatTensor(image).unsqueeze(0).to(self.device)
    
    def detect_helmet(self, person_crop):
        """Detect if person is wearing helmet"""
        if person_crop is None or person_crop.size == 0:
            return False, 0.0
        
        try:
            input_tensor = self.preprocess_for_helmet(person_crop)
            with torch.no_grad():
                outputs = self.helmet_model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, 1)
                confidence = torch.max(probabilities, 1)[0].item()
                
                # 0: helmet, 1: no_helmet
                has_helmet = predicted.item() == 0
                return has_helmet, confidence
        except Exception as e:
            print(f"Helmet detection error: {e}")
            return False, 0.0
    
    def extract_license_plate(self, vehicle_crop):
        """Extract license plate number from vehicle crop"""
        try:
            # Preprocess for better OCR
            gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Find edges
            edged = cv2.Canny(gray, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            # Look for rectangular contour (license plate)
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    # Extract the license plate region
                    mask = np.zeros(gray.shape, np.uint8)
                    cv2.drawContours(mask, [approx], -1, 255, -1)
                    plate_crop = cv2.bitwise_and(gray, gray, mask=mask)
                    
                    # OCR on the extracted plate
                    results = self.ocr_reader.readtext(plate_crop)
                    if results:
                        # Combine all detected text
                        plate_text = ''.join([result[1] for result in results if result[2] > 0.5])
                        plate_text = ''.join(c for c in plate_text if c.isalnum())
                        if len(plate_text) >= 4:  # Minimum length for valid plate
                            return plate_text
            
            # If no contour-based detection, try OCR on whole crop
            results = self.ocr_reader.readtext(vehicle_crop)
            if results:
                plate_text = ''.join([result[1] for result in results if result[2] > 0.5])
                plate_text = ''.join(c for c in plate_text if c.isalnum())
                if len(plate_text) >= 4:
                    return plate_text
            
            return "UNKNOWN"
        except Exception as e:
            print(f"License plate extraction error: {e}")
            return "UNKNOWN"
    
    def is_in_zone(self, bbox, zone_coords):
        """Check if bounding box center is in specified zone"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        zone_x1, zone_y1 = zone_coords[0]
        zone_x2, zone_y2 = zone_coords[1]
        
        return zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2
    
    def count_people_on_vehicle(self, vehicle_bbox, person_detections):
        """Count people on a specific vehicle"""
        vx1, vy1, vx2, vy2 = vehicle_bbox
        count = 0
        
        for person_bbox in person_detections:
            px1, py1, px2, py2 = person_bbox
            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2
            
            # Check if person is on the vehicle (with some tolerance)
            if (vx1 - 20 <= person_center_x <= vx2 + 20 and 
                vy1 - 20 <= person_center_y <= vy2 + 20):
                count += 1
        
        return count
    
    def process_frame(self, frame):
        """Process a single frame for traffic violations"""
        violations = []
        
        # Run YOLO detection
        results = self.yolo_model(frame)
        
        # Extract detections
        vehicle_detections = []
        person_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Vehicle classes (motorcycle=3, car=2, bus=5, truck=7)
                        if cls in [2, 3, 5, 7]:
                            vehicle_detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'class': cls,
                                'confidence': conf
                            })
                        
                        # Person class
                        elif cls == 0:
                            person_detections.append([int(x1), int(y1), int(x2), int(y2)])
        
        # Process each vehicle
        for vehicle in vehicle_detections:
            bbox = vehicle['bbox']
            vehicle_class = vehicle['class']
            
            # Extract vehicle crop
            vehicle_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            # Extract license plate
            license_plate = self.extract_license_plate(vehicle_crop)
            
            # Check for violations
            current_time = datetime.now().timestamp()
            vehicle_key = f"{bbox[0]}_{bbox[1]}"
            
            # Avoid duplicate violations for same vehicle
            if vehicle_key in self.violation_cooldown:
                if current_time - self.violation_cooldown[vehicle_key] < 5:  # 5 second cooldown
                    continue
            
            # 1. No Parking Violation
            if self.is_in_zone(bbox, self.violation_zones['no_parking']):
                violations.append({
                    'type': 'no_parking',
                    'license_plate': license_plate,
                    'fine': self.fines['no_parking'],
                    'bbox': bbox,
                    'message': f"No Parking Violation! Vehicle: {license_plate}, Fine: Rs.{self.fines['no_parking']}"
                })
                self.violation_cooldown[vehicle_key] = current_time
            
            # 2. Signal Violation
            elif self.is_in_zone(bbox, self.violation_zones['signal']):
                violations.append({
                    'type': 'signal_violation',
                    'license_plate': license_plate,
                    'fine': self.fines['signal_violation'],
                    'bbox': bbox,
                    'message': f"Signal Violation! Vehicle: {license_plate}, Fine: Rs.{self.fines['signal_violation']}"
                })
                self.violation_cooldown[vehicle_key] = current_time
            
            # 3. For motorcycles, check helmet and triple riding
            if vehicle_class == 3:  # Motorcycle
                people_count = self.count_people_on_vehicle(bbox, person_detections)
                
                # Triple riding check
                if people_count >= 3:
                    violations.append({
                        'type': 'triple_riding',
                        'license_plate': license_plate,
                        'fine': self.fines['triple_riding'],
                        'bbox': bbox,
                        'message': f"Triple Riding Violation! Vehicle: {license_plate}, Fine: Rs.{self.fines['triple_riding']}"
                    })
                    self.violation_cooldown[vehicle_key] = current_time
                
                # Helmet check for people on motorcycle
                elif people_count > 0:
                    helmet_violations = 0
                    for person_bbox in person_detections:
                        px1, py1, px2, py2 = person_bbox
                        person_center_x = (px1 + px2) / 2
                        person_center_y = (py1 + py2) / 2
                        
                        # Check if person is on motorcycle
                        if (bbox[0] - 20 <= person_center_x <= bbox[2] + 20 and 
                            bbox[1] - 20 <= person_center_y <= bbox[3] + 20):
                            
                            # Extract person crop
                            person_crop = frame[py1:py2, px1:px2]
                            
                            # Check helmet
                            has_helmet, confidence = self.detect_helmet(person_crop)
                            
                            if not has_helmet and confidence > 0.6:
                                helmet_violations += 1
                    
                    if helmet_violations > 0:
                        violations.append({
                            'type': 'no_helmet',
                            'license_plate': license_plate,
                            'fine': self.fines['no_helmet'],
                            'bbox': bbox,
                            'message': f"No Helmet Violation! Vehicle: {license_plate}, Fine: Rs.{self.fines['no_helmet']}"
                        })
                        self.violation_cooldown[vehicle_key] = current_time
        
        return violations
    
    def save_violation_image(self, frame, violation, output_dir="violations"):
        """Save violation image to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{violation['type']}_{violation['license_plate']}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Draw bounding box on frame
        bbox = violation['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        cv2.putText(frame, violation['type'], (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imwrite(filepath, frame)
        return filepath

# GUI Application
class TrafficMonitoringApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Automated Traffic Monitoring System")
        self.root.geometry("1200x800")
        
        self.detector = TrafficViolationDetector()
        self.cap = None
        self.is_monitoring = False
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Start Camera", command=self.start_monitoring).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Stop Camera", command=self.stop_monitoring).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Load Video", command=self.load_video).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="View Violations", command=self.view_violations).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Video display
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed")
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(padx=10, pady=10)
        
        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Status")
        status_frame.pack(fill=tk.X)
        
        self.status_text = tk.Text(status_frame, height=8)
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        self.log_message("Traffic Monitoring System Initialized")
    
    def log_message(self, message):
        """Log message to status panel"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
    
    def start_monitoring(self):
        """Start camera monitoring"""
        if not self.is_monitoring:
            self.cap = cv2.VideoCapture(0)  # Use default camera
            if self.cap.isOpened():
                self.is_monitoring = True
                self.log_message("Started camera monitoring")
                self.monitor_loop()
            else:
                messagebox.showerror("Error", "Could not open camera")
    
    def stop_monitoring(self):
        """Stop camera monitoring"""
        self.is_monitoring = False
        if self.cap:
            self.cap.release()
        self.log_message("Stopped camera monitoring")
    
    def load_video(self):
        """Load and process video file"""
        from tkinter import filedialog
        
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.is_monitoring = True
                self.log_message(f"Loaded video: {video_path}")
                self.monitor_loop()
            else:
                messagebox.showerror("Error", "Could not load video file")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        if self.is_monitoring and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret:
                # Resize frame for display
                display_frame = cv2.resize(frame, (640, 480))
                
                # Process frame for violations
                violations = self.detector.process_frame(frame)
                
                # Draw violation zones
                cv2.rectangle(display_frame, 
                            tuple(self.detector.violation_zones['no_parking'][0]),
                            tuple(self.detector.violation_zones['no_parking'][1]),
                            (255, 0, 0), 2)
                cv2.putText(display_frame, "NO PARKING", 
                          (self.detector.violation_zones['no_parking'][0][0], 
                           self.detector.violation_zones['no_parking'][0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.rectangle(display_frame,
                            tuple(self.detector.violation_zones['signal'][0]),
                            tuple(self.detector.violation_zones['signal'][1]),
                            (0, 255, 255), 2)
                cv2.putText(display_frame, "SIGNAL ZONE",
                          (self.detector.violation_zones['signal'][0][0],
                           self.detector.violation_zones['signal'][0][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Process violations
                for violation in violations:
                    # Save violation image
                    violation_image_path = self.detector.save_violation_image(frame.copy(), violation)
                    
                    # Add to database
                    self.detector.db_manager.add_violation(
                        violation['type'],
                        violation['license_plate'],
                        violation['fine'],
                        violation_image_path
                    )
                    
                    # Log violation
                    self.log_message(violation['message'])
                    
                    # Draw bounding box on display frame
                    bbox = violation['bbox']
                    scale_x = 640 / frame.shape[1]
                    scale_y = 480 / frame.shape[0]
                    
                    scaled_bbox = [
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y)
                    ]
                    
                    cv2.rectangle(display_frame, 
                                (scaled_bbox[0], scaled_bbox[1]),
                                (scaled_bbox[2], scaled_bbox[3]),
                                (0, 0, 255), 2)
                    cv2.putText(display_frame, violation['type'],
                              (scaled_bbox[0], scaled_bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Convert frame for tkinter display
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Schedule next frame
                self.root.after(30, self.monitor_loop)
            else:
                self.stop_monitoring()
    
    def view_violations(self):
        """Open violations viewer window"""
        violations_window = tk.Toplevel(self.root)
        violations_window.title("Violation Records")
        violations_window.geometry("800x600")
        
        # Create treeview for violations
        columns = ("ID", "Timestamp", "Type", "Vehicle", "Fine", "Status")
        tree = ttk.Treeview(violations_window, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(violations_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Load violation data
        violations = self.detector.db_manager.get_violations()
        for violation in violations:
            tree.insert("", "end", values=violation)
        
        # Add buttons
        button_frame = ttk.Frame(violations_window)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Refresh", 
                  command=lambda: self.refresh_violations(tree)).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Export", 
                  command=lambda: self.export_violations()).pack(side="left", padx=5)
    
    def refresh_violations(self, tree):
        """Refresh violations in treeview"""
        for item in tree.get_children():
            tree.delete(item)
        
        violations = self.detector.db_manager.get_violations()
        for violation in violations:
            tree.insert("", "end", values=violation)
    
    def export_violations(self):
        """Export violations to CSV"""
        from tkinter import filedialog
        import csv
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            violations = self.detector.db_manager.get_violations()
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["ID", "Timestamp", "Type", "Vehicle", "Fine", "Image", "Status"])
                writer.writerows(violations)
            
            messagebox.showinfo("Success", f"Violations exported to {file_path}")
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_monitoring()
            cv2.destroyAllWindows()

# Training Script for Helmet Detection
def train_helmet_model():
    """Train the helmet detection model"""
    # This is a simplified training script
    # In practice, you'd need a proper dataset with helmet/no-helmet images
    
    print("Training helmet detection model...")
    print("Note: This requires a dataset of helmet/no-helmet images")
    print("Place your training data in 'dataset/helmet' and 'dataset/no_helmet' folders")
    
    # Placeholder for training code
    # You would implement proper data loading, training loop, etc.
    model = HelmetDetector()
    # ... training code ...
    torch.save(model.state_dict(), 'helmet_model.pth')
    print("Model saved as helmet_model.pth")

# Main execution
if __name__ == "__main__":
    # Check if training is needed
    if not os.path.exists('helmet_model.pth'):
        print("No trained helmet model found.")
        response = input("Do you want to train a new model? (y/n): ")
        if response.lower() == 'y':
            train_helmet_model()
    
    # Start the application
    app = TrafficMonitoringApp()
    app.run()
