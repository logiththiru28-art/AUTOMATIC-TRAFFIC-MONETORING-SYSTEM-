# requirements.txt
torch>=1.9.0
torchvision>=0.10.0
ultralytics>=8.0.0
opencv-python>=4.5.0
easyocr>=1.6.0
numpy>=1.21.0
Pillow>=8.3.0
tkinter
sqlite3
smtplib

# config.py - Configuration file for the traffic monitoring system

import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    VIOLATIONS_DIR = BASE_DIR / "violations"
    DATABASE_PATH = BASE_DIR / "traffic_violations.db"
    
    # Model configurations
    YOLO_MODEL_PATH = "yolov8n.pt"  # Will download automatically
    HELMET_MODEL_PATH = MODELS_DIR / "helmet_model.pth"
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.5
    HELMET_CONFIDENCE_THRESHOLD = 0.6
    
    # Violation zones (x1, y1, x2, y2) - adjust based on your camera setup
    VIOLATION_ZONES = {
        'no_parking': [(100, 100), (300, 200)],
        'signal': [(400, 150), (600, 250)],
        'wrong_side': [(50, 300), (250, 400)]
    }
    
    # Fine amounts (in currency units)
    FINES = {
        'no_helmet': 200,
        'triple_riding': 500,
        'no_parking': 300,
        'signal_violation': 1000,
        'wrong_side': 750,
        'overspeeding': 1500
    }
    
    # Vehicle classes from YOLO
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    # Tracking parameters
    VIOLATION_COOLDOWN = 5  # seconds
    MAX_TRACKING_DISTANCE = 100  # pixels
    
    # Email configuration (for notifications)
    EMAIL_CONFIG = {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': 'your_email@gmail.com',
        'sender_password': 'your_app_password',
        'recipient_emails': ['traffic_police@city.gov', 'admin@traffic.com']
    }
    
    # Camera settings
    CAMERA_SETTINGS = {
        'default_camera': 0,
        'frame_width': 1280,
        'frame_height': 720,
        'fps': 30
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VIOLATIONS_DIR.mkdir(exist_ok=True)

# setup.py - Setup script for the project

import os
import sys
from pathlib import Path
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "easyocr>=1.6.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "matplotlib>=3.3.0"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")
            return False
    
    return True

def download_yolo_model():
    """Download YOLO model if not present"""
    print("Checking YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✓ YOLO model ready")
        return True
    except Exception as e:
        print(f"✗ Failed to load YOLO model: {e}")
        return False

def setup_database():
    """Initialize the database"""
    print("Setting up database...")
    
    try:
        import sqlite3
        from config import Config
        
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                vehicle_number TEXT,
                fine_amount INTEGER,
                image_path TEXT,
                location TEXT,
                status TEXT DEFAULT 'PENDING',
                officer_id TEXT,
                notes TEXT
            )
        ''')
        
        # Create officers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS officers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                officer_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                badge_number TEXT,
                station TEXT,
                contact TEXT
            )
        ''')
        
        # Create settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("✓ Database initialized")
        return True
        
    except Exception as e:
        print(f"✗ Database setup failed: {e}")
        return False

def create_sample_data():
    """Create sample violation zones and test data"""
    print("Creating sample configuration...")
    
    sample_config = """
# Sample camera configuration for traffic monitoring

# Adjust these coordinates based on your camera view
VIOLATION_ZONES = {
    'no_parking': [(100, 100), (300, 200)],  # Top-left area
    'signal': [(400, 150), (600, 250)],      # Center area
    'wrong_side': [(50, 300), (250, 400)]   # Bottom-left area
}

# Fine amounts (in your local currency)
FINES = {
    'no_helmet': 200,
    'triple_riding': 500, 
    'no_parking': 300,
    'signal_violation': 1000,
    'wrong_side': 750
}

Instructions:
1. Run the application with: python main.py
2. Use "Start Camera" to begin live monitoring
3. Use "Load Video" to process recorded footage
4. Adjust violation zones in config.py based on your camera setup
5. View violations in the "View Violations" window
"""
    
    try:
        with open("camera_setup.txt", "w") as f:
            f.write(sample_config)
        print("✓ Sample configuration created")
        return True
    except Exception as e:
        print(f"✗ Failed to create sample config: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("Traffic Monitoring System Setup")
    print("=" * 50)
    
    # Create directories
    from config import Config
    Config.create_directories()
    print("✓ Directories created")
    
    # Install requirements
    if not install_requirements():
        print("Setup failed at package installation")
        return
    
    # Download YOLO model
    if not download_yolo_model():
        print("Setup failed at YOLO model download")
        return
        
    # Setup database
    if not setup_database():
        print("Setup failed at database setup")
        return
        
    # Create sample data
    if not create_sample_data():
        print("Setup failed at sample data creation")
        return
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Configure violation zones in config.py")
    print("3. Test with your camera or video files")
    print("\nFor helmet detection training:")
    print("1. Create dataset folders: dataset/helmet/ and dataset/no_helmet/")
    print("2. Add training images to respective folders")
    print("3. Run: python train_helmet_model.py")

if __name__ == "__main__":
    main()

# train_helmet_model.py - Training script for helmet detection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image

class HelmetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['helmet', 'no_helmet']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class HelmetDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(HelmetDetector, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(data_dir, num_epochs=20, batch_size=32, learning_rate=0.001):
    """Train the helmet detection model"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = HelmetDataset(data_dir / 'train', transform=train_transform)
    val_dataset = HelmetDataset(data_dir / 'val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HelmetDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    print(f"Training on device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'helmet_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step()
    
    print(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    return model

def create_sample_dataset():
    """Create sample dataset structure"""
    dataset_dir = Path('dataset')
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in ['helmet', 'no_helmet']:
            (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("Dataset structure created:")
    print("dataset/")
    print("├── train/")
    print("│   ├── helmet/")
    print("│   └── no_helmet/")
    print("└── val/")
    print("    ├── helmet/")
    print("    └── no_helmet/")
    print("\nPlease add your training images to the respective folders.")

if __name__ == "__main__":
    data_dir = Path('dataset')
    
    if not data_dir.exists():
        print("No dataset found. Creating sample structure...")
        create_sample_dataset()
    else:
        print("Starting helmet detection model training...")
        
        # Check if dataset has images
        train_helmet = list((data_dir / 'train' / 'helmet').glob('*.jpg'))
        train_no_helmet = list((data_dir / 'train' / 'no_helmet').glob('*.jpg'))
        
        if len(train_helmet) == 0 or len(train_no_helmet) == 0:
            print("No training images found!")
            print("Please add images to:")
            print("- dataset/train/helmet/ (images of people with helmets)")
            print("- dataset/train/no_helmet/ (images of people without helmets)")
            print("- dataset/val/helmet/ (validation images with helmets)")
            print("- dataset/val/no_helmet/ (validation images without helmets)")
        else:
            model = train_model(data_dir)
            print("Model training completed!")