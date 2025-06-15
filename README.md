# Enhanced YOLOv8 SAR Ship Detection

This repository contains an **enhanced YOLOv8 object detection model** based on the research paper:
**"SAR Small Ship Detection Based on Enhanced YOLO Network"**

## ✨ Key Features

The model integrates the following improvements over standard YOLOv8n:

- **SRBlock** (Shuffle Re-parameterization Block) - Improves feature extraction efficiency
- **SPDModule** (Semantic Pyramid Module) - Enhances multi-scale feature representation
- **HybridAttention** module - Focuses on relevant features while suppressing noise
- **Shape-NWD Loss** - Advanced loss function for better bounding box regression

> 🚫 **Note**: Dataset not included in this repository. It can be downloaded using the script provided below.

## 📦 Folder Structure

```
Enhanced-YOLOv8-SARShip/
├── ultralytics/                         # Modified YOLOv8 codebase
│   ├── ultralytics/
│   │   ├── nn/
│   │   ├── models/
|   |   |   ├── custom
|   |   |       ├── yolov8n_custom.yaml  # Model architecture YAML
│   │   ├── custom_modules/              # SRBlock, SPDModule, HA.
│   │   └── ...
├── SAR-Ship-Detection/                  # Folder for dataset (downloaded separately)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Dee-Mark9480/Enhanced-YOLOv8-SARShip-Detection.git
```

### 2. Create & Activate Virtual Environment

**Python 3.10 is recommended**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download SAR-Ship Dataset

```bash
# Clone the dataset repository
git clone --depth 1 https://github.com/Akashkalasagond/SAR-Ship-Detection.git
```

## 🧠 Training the Model

### Method 1: Command Line Training

Run this command in Command Prompt/Terminal:

```bash
yolo task=detect mode=train ^
model=D:/Coding/New_folder/MRC/ultralytics/ultralytics/models/custom/yolov8n_custom.yaml ^
data=D:/Coding/New_folder/MRC/SAR-Ship-Detection/ship_v8.yaml ^
imgsz=800 ^
epochs=120 ^
batch=16 ^
optimizer=SGD ^
lr0=0.01 ^
momentum=0.937 ^
weight_decay=0.0005 ^
device=0 ^
project=eyolov8_model_results ^
name=E_YOLOv8n_HRSID
```

> **Note**: Replace the paths with your actual project paths.

### Method 2: Python Script Training

You can also train the model using Python by importing the necessary libraries:

```python
import torch
from ultralytics import YOLO

# Load the custom model
model = YOLO('path/to/your/yolov8n_custom.yaml')

# Train the model
results = model.train(
    data='path/to/your/SAR-Ship-Detection/ship_v8.yaml',
    imgsz=800,
    epochs=120,
    batch=16,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    device=0,
    project='eyolov8_model_results',
    name='E_YOLOv8n_HRSID'
)
```

## ⚠️ Memory Management

### Out of Memory Error Solutions

If you encounter CUDA OOM (Out of Memory) error, try these solutions:

#### 1. Reduce Batch Size
```bash
# Reduce batch size from 16 to 8 or 4
batch=8
```

#### 2. Adjust Learning Rate Proportionally
When reducing batch size, adjust the learning rate accordingly:

**Formula**: `New LR = Original LR × (New batch / Original batch)`

**Example**:
- Original: `batch=16`, `lr0=0.01`
- New: `batch=8`, `lr0=0.01 × (8/16) = 0.005`

```bash
# Updated command with reduced batch size and adjusted learning rate
yolo task=detect mode=train ^
model=your_model_path ^
data=your_data_path ^
batch=8 ^
lr0=0.005 ^
[other parameters...]
```
