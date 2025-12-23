# Traffic Monitoring System - Complete Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ installed
- Webcam or IP camera
- CUDA-compatible GPU (optional but recommended)
- 8GB+ RAM recommended

### Installation Steps

1. **Clone/Download the Project**
```bash
mkdir traffic_monitoring
cd traffic_monitoring
```

2. **Install Dependencies**
```bash
pip install torch torchvision ultralytics opencv-python easyocr numpy Pillow matplotlib psutil GPUtil
```

3. **Run Setup Script**
```bash
python setup.py
```

4. **Start the Application**
```bash
python advanced_main.py
```

## 📁 Project Structure

```
traffic_monitoring/
├── main.py                    # Core detection system
├── advanced_main.py           # Advanced GUI application
├── config.py                  # Configuration settings
├── utils.py                   # Utility functions
├── setup.py                   # Setup and installation script
├── train_helmet_model.py      # Helmet model training
├── requirements.txt           # Python dependencies
├── models/                    # Trained models directory
│   └── helmet_model.pth      # Helmet detection model
├── dataset/                   # Training data (if training)
│   ├── train/
│   │   ├── helmet/
│   │   └── no_helmet/
│   └── val/
│       ├── helmet/
│       └── no_helmet/
├── violations/                # Saved violation images
├── data/                      # Application data
└── traffic_violations.db      # SQLite database
```

## ⚙️ Configuration

### 1. Camera Setup
- Adjust camera settings in `config.py`
- Set `CAMERA_SETTINGS['default_camera']` to your camera index (0 for built-in, 1+ for external)

### 2. Violation Zones
- Modify `VIOLATION_ZONES` in `config.py` based on your camera view
- Use the settings panel in the application to adjust zones visually

### 3. Fine Amounts
- Update `FINES` dictionary in `config.py` with your local currency values

### 4. Email Notifications
- Configure `EMAIL_CONFIG` in `config.py` for automated alerts
- Use app passwords for Gmail

## 🔧 Features

### Real-time Detection
- Helmet violation detection
- Triple riding detection
- No parking violations
- Signal jumping detection
- Wrong-side driving detection
- Speed monitoring (with calibration)

### Database Management
- SQLite database for violation records
- Automatic data backup
- Export to CSV functionality
- Violation history tracking

### Performance Monitoring
- Real-time CPU/Memory/GPU usage
- FPS monitoring
- Processing time metrics
- System health alerts

### Notifications
- Email alerts for violations
- Automatic report generation
- SMS integration (with additional setup)

## 🎯 Usage Instructions

### Starting the System

1. **Launch Advanced Application**
```bash
python advanced_main.py
```

2. **Select Input Source**
   - Click "Start Camera" for live monitoring
   - Click "Load Video" to process recorded footage

3. **Monitor Violations**
   - View live feed with violation zones highlighted
   - Check statistics panel for real-time counts
   - Review recent violations in the side panel

### Training Custom Helmet Model

1. **Prepare Dataset**
```bash
mkdir -p dataset/train/{helmet,no_helmet}
mkdir -p dataset/val/{helmet,no_helmet}
```

2. **Add Training Images**
   - Place helmet images in `dataset/train/helmet/`
   - Place no-helmet images in `dataset/train/no_helmet/`
   - Add validation images to respective `val/` folders

3. **Train Model**
```bash
python train_helmet_model.py
```

### Configuring Violation Zones

Edit `config.py` to adjust zones based on your camera setup:

```python
VIOLATION_ZONES = {
    'no_parking': [(100, 100), (300, 200)],  # Top-left area
    'signal': [(400, 150), (600, 250)],      # Center intersection
    'wrong_side': [(50, 300), (250, 400)]   # Left side of road
}
```

## 📊 System Requirements

### Minimum Requirements
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 10GB free space
- Camera: 720p webcam or IP camera

### Recommended Requirements
- CPU: Intel i7 or AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA GTX 1060 or better (CUDA support)
- Storage: 50GB+ SSD
- Camera: 1080p camera with good lighting

## 🔧 Troubleshooting

### Common Issues

**1. Camera Not Detected**
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```
Solution: Update camera index in config.py

**2. YOLO Model Download Issues**
- Ensure internet connection
- Check firewall settings
- Manual download from Ultralytics

**3. Poor Detection Accuracy**
- Ensure good lighting conditions
- Adjust confidence thresholds in config
- Train custom helmet model with your data

**4. High CPU/Memory Usage**
- Reduce frame processing rate
- Lower input resolution
- Use GPU acceleration if available

### Performance Optimization

**For Raspberry Pi/Edge Devices:**
```python
# In config.py, use lighter models
YOLO_MODEL_PATH = "yolov8n.pt"  # Nano version
CONFIDENCE_THRESHOLD = 0.6      # Higher threshold
```

**For High-Performance Systems:**
```python
# Use larger, more accurate models
YOLO_MODEL_PATH = "yolov8x.pt"  # Extra Large version
CONFIDENCE_THRESHOLD = 0.3      # Lower threshold for better detection
```

## 🔒 Security Considerations

### Data Privacy
- Violation images are stored locally
- Database encryption available
- GDPR compliance features

### Network Security
- Use encrypted connections for IP cameras
- Secure email credentials
- Regular security updates

## 🚀 Deployment Options

### 1. Desktop Application
- Direct execution on Windows/Linux/Mac
- Local processing and storage
- Manual monitoring

### 2. Server Deployment
```bash
# Install as service (Linux)
sudo systemctl enable traffic-monitor
sudo systemctl start traffic-monitor
```

### 3. Cloud Deployment
- Docker containerization available
- AWS/Azure/GCP compatible
- Scalable processing

### 4. Edge Device Deployment
- Raspberry Pi 4+ support
- NVIDIA Jetson compatibility
- Optimized for resource constraints

## 📈 Analytics and Reporting

### Daily Reports
- Violation summaries
- Peak violation times
- Vehicle type analysis
- Fine collection totals

### Weekly/Monthly Analytics
- Trend analysis
- Hotspot identification
- Officer performance metrics
- System uptime statistics

## 🔌 Integration Options

### Third-party Integrations
- Traffic management systems
- Police databases
- Court systems
- Payment gateways

### API Endpoints
- REST API for external systems
- Webhook notifications
- Real-time data streaming

## 📚 API Documentation

### Basic API Usage
```python
from main import TrafficViolationDetector

# Initialize detector
detector = TrafficViolationDetector()

# Process single frame
violations = detector.process_frame(frame)

# Save violation
detector.db_manager.add_violation(
    violation_type="no_helmet",
    vehicle_number="ABC123",
    fine_amount=200,
    image_path="violation.jpg"
)
```

### Database Schema
```sql
-- Violations table
CREATE TABLE violations (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    violation_type TEXT,
    vehicle_number TEXT,
    fine_amount INTEGER,
    image_path TEXT,
    status TEXT DEFAULT 'PENDING'
);
```

## 🛠 Customization

### Adding New Violation Types
1. Update `FINES` dictionary in config.py
2. Modify detection logic in main.py
3. Add UI elements if needed

### Custom Models
1. Train specialized models for your region
2. Integrate license plate recognition for local formats
3. Add vehicle classification for specific types

## 📞 Support

### Getting Help
- Check documentation first
- Review troubleshooting section
- Create GitHub issues for bugs
- Join community forums

### Contributing
- Fork the repository
- Create feature branches
- Submit pull requests
- Follow coding standards

## 📋 License and Legal

### Usage Rights
- Educational and research use
- Commercial licensing available
- Compliance with local laws required

### Legal Considerations
- Ensure proper authorization for surveillance
- Comply with privacy regulations
- Follow traffic enforcement procedures
- Maintain chain of custody for evidence

## 🔄 Updates and Maintenance

### Regular Maintenance
- Update YOLO models monthly
- Backup database weekly
- Clean violation images periodically
- Monitor system performance

### Version Updates
- Check for updates regularly
- Test in staging environment
- Backup before upgrades
- Review change logs

---

## Quick Reference Commands

```bash
# Installation
pip install -r requirements.txt
python setup.py

# Training
python train_helmet_model.py

# Execution
python advanced_main.py          # GUI application
python main.py                   # CLI version

# Testing
python -m pytest tests/          # Run tests
python test_detection.py         # Test detection

# Maintenance
python backup_database.py        # Backup data
python clean_violations.py       # Clean old files
python performance_check.py      # System health
```

This complete traffic monitoring system provides enterprise-level functionality with real-time processing, comprehensive violation detection, and professional reporting capabilities. The modular design allows for easy customization and integration with existing traffic management infrastructure.