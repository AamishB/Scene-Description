# Scene Description System for Visually Impaired

A real-time AI-powered scene understanding system that combines YOLOv8 object detection and BLIP vision-language models to assist visually impaired individuals. The system provides comprehensive audio descriptions of surroundings, including object detection, spatial awareness, and natural language scene descriptions.

## 🌟 Key Features

### Object Detection & Recognition

- **Real-time Object Detection**: Powered by YOLOv8 for fast and accurate detection of 80+ object classes
- **Multiple Detection Models**: Support for YOLOv8n (nano), YOLOv8s (small), YOLOv8m (medium), YOLOv8l (large), and YOLOv8x (extra-large)
- **Confidence Filtering**: Adjustable confidence thresholds to reduce false positives
- **Detection Stability**: History tracking to ensure stable and reliable detections

### Scene Understanding

- **Scene Recognition**: BLIP (Bootstrapping Language-Image Pre-training) generates natural language scene descriptions
- **Context Identification**: Automatically identifies indoor/outdoor environments and room types (kitchen, office, bedroom, etc.)
- **Combined Descriptions**: Intelligently merges object detection with scene captioning for comprehensive understanding

### Spatial Awareness

- **Position Detection**: Identifies object positions (left/center/right, top/middle/bottom)
- **Distance Estimation**: Provides relative distance information (near/medium/far) based on object size
- **Enhanced Visualization**: Color-coded bounding boxes with spatial information overlays

### Audio Feedback System

- **Text-to-Speech**: Natural voice descriptions using pyttsx3
- **Smart Announcements**: Announces new objects entering or leaving the scene
- **Periodic Summaries**: Configurable interval-based scene descriptions
- **Change Detection**: Real-time alerts when scene composition changes
- **Accessibility-Enhanced Captions**: Captions optimized for visually impaired users

### Performance & Optimization

- **Frame Skipping**: Process every Nth frame for improved performance
- **GPU Acceleration**: CUDA support for 5-10x faster processing
- **Dynamic Resizing**: Automatic frame resizing for optimal performance
- **Multi-source Support**: Works with IP cameras, USB webcams, and video files
- **Performance Monitoring**: Real-time FPS and processing time display

### User Interface

- **Interactive Controls**: Keyboard shortcuts for all major functions
- **Visual Feedback**: Bounding boxes, confidence scores, and FPS display
- **Scene Overlay**: Real-time scene caption display on video
- **Status Indicators**: Audio and detection status indicators

## 📁 Project Structure

```
Scene-Description/
├── cv.py                          # Main application entry point
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── yolov8n.pt                     # YOLOv8 model weights (auto-downloaded)
│
├── config/                        # Configuration module
│   ├── __init__.py
│   └── settings.py               # Centralized configuration settings
│
├── models/                        # AI model implementations
│   ├── __init__.py
│   ├── object_detector.py        # YOLOv8 object detection
│   └── scene_recognizer.py       # BLIP scene captioning
│
└── utils/                         # Utility modules
    ├── __init__.py
    ├── video_handler.py          # Video capture & streaming
    └── audio_feedback.py         # Text-to-speech system
```

### Module Descriptions

- **cv.py**: Main application loop, coordinates all components
- **object_detector.py**: YOLOv8-based object detection with spatial analysis
- **scene_recognizer.py**: BLIP-based scene captioning and context identification
- **video_handler.py**: Video stream management with auto-reconnect
- **audio_feedback.py**: Async text-to-speech with announcement queuing
- **settings.py**: Centralized configuration for all system parameters

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Camera**: Webcam, IP camera (e.g., IP Webcam app for Android), or video file
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster processing
- **Operating System**: Windows, macOS, or Linux

### Step 1: Clone the Repository

```bash
git clone https://github.com/AamishB/Scene-Description
cd Scene-Description
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Models

The required models will be automatically downloaded on first run:

- **YOLOv8n** (~6 MB) - Object detection model
- **BLIP base** (~500 MB) - Scene captioning model

Alternatively, manually download YOLOv8 models from [Ultralytics](https://github.com/ultralytics/ultralytics)

### Step 5: Configure Video Source

Edit `config/settings.py` and update the `VIDEO_SOURCE`:

```python
# For IP Camera (e.g., IP Webcam app)
VIDEO_SOURCE = "http://192.168.1.100:8080/video"

# For USB Webcam
VIDEO_SOURCE = 0  # or 1, 2 for other cameras

# For Video File
VIDEO_SOURCE = "path/to/your/video.mp4"
```

## ⚙️ Configuration

The system is highly configurable through `config/settings.py`:

### Video Settings

```python
VIDEO_SOURCE = "http://10.249.52.9:8080/video"  # IP camera URL or camera index
```

### Object Detection Settings

```python
YOLO_MODEL = "yolov8n.pt"           # Model size: n, s, m, l, x
CONFIDENCE_THRESHOLD = 0.5          # Detection confidence (0.0-1.0)
MAX_OBJECTS_TO_DESCRIBE = 5         # Max objects in audio description
STABLE_DETECTION_FRAMES = 3         # Frames for stable detection
```

### Scene Recognition Settings

```python
ENABLE_SCENE_RECOGNITION = True     # Enable/disable scene captioning
SCENE_MODEL = "Salesforce/blip-image-captioning-base"
SCENE_CAPTION_INTERVAL = 10.0       # Seconds between captions
SCENE_MAX_LENGTH = 50               # Maximum caption length
SCENE_NUM_BEAMS = 4                 # Beam search quality (1-5)
COMBINE_SCENE_AND_OBJECTS = True    # Merge scene + objects
```

### Performance Settings

```python
PROCESS_EVERY_N_FRAMES = 5          # Process every Nth frame (1-20)
ENABLE_GPU = False                  # Enable CUDA GPU acceleration
RESIZE_FRAME = True                 # Resize for better performance
FRAME_WIDTH = 640                   # Processing width
FRAME_HEIGHT = 480                  # Processing height
```

### Display Settings

```python
SHOW_BOUNDING_BOXES = True          # Show detection boxes
SHOW_CONFIDENCE = True              # Show confidence scores
SHOW_DISTANCE = True                # Show distance estimates
SHOW_FPS = True                     # Show FPS counter
```

### Audio Settings

```python
ENABLE_AUDIO = True                 # Enable text-to-speech
SPEECH_RATE = 150                   # Words per minute (100-200)
SPEECH_VOLUME = 0.9                 # Volume (0.0-1.0)
ANNOUNCEMENT_INTERVAL = 5.0         # Seconds between announcements
```

### Priority Classes

Objects that will be prioritized in descriptions:

```python
PRIORITY_CLASSES = [
    'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
    'dog', 'cat', 'traffic light', 'stop sign', 'chair', 'couch',
    'bottle', 'cup', 'book', 'laptop', 'cell phone', 'door'
]
```

## 🎮 Usage

### Starting the System

```bash
python cv.py
```

The system will:

1. Load configuration from `config/settings.py`
2. Initialize video handler and connect to camera
3. Load YOLOv8 object detection model
4. Load BLIP scene recognition model (if enabled)
5. Initialize text-to-speech system
6. Start real-time processing and display window

### Keyboard Controls

| Key            | Function                                                  |
| -------------- | --------------------------------------------------------- |
| **Q**          | Quit the application                                      |
| **D**          | Toggle bounding box display ON/OFF                        |
| **A**          | Toggle audio feedback ON/OFF                              |
| **S**          | Speak current scene description (combined)                |
| **C**          | Generate and speak scene caption immediately              |
| **P**          | Print detection summary to console                        |
| **+** or **=** | Increase frame skip (faster FPS, less frequent updates)   |
| **-**          | Decrease frame skip (slower FPS, more accurate detection) |
| **R**          | Toggle frame resize ON/OFF                                |

### Example Workflow

1. **Start the system**:

   ```bash
   python cv.py
   ```

2. **The system announces**: "Scene description system started with scene recognition. Press Q to quit, C for scene caption."

3. **Automatic operation**:

   - Continuously detects objects and updates bounding boxes
   - Announces new objects: "New objects detected: person, laptop"
   - Announces removed objects: "Objects no longer visible: cup"
   - Periodic scene summaries every 5 seconds (configurable)

4. **Manual triggers**:

   - Press **C** for immediate scene caption
   - Press **S** for full scene description with objects
   - Press **P** to see console summary

5. **Adjust performance**:
   - Press **+** to process fewer frames (faster)
   - Press **-** to process more frames (more accurate)
   - Press **R** to toggle frame resizing

## 🎯 Features Explained

### 1. Object Detection (YOLOv8)

YOLOv8 is a state-of-the-art object detection model that identifies 80 different object classes in real-time:

**Supported Objects Include:**

- **People & Body Parts**: person
- **Vehicles**: car, truck, bus, bicycle, motorcycle, airplane, boat, train
- **Animals**: dog, cat, horse, cow, sheep, bird, elephant, bear, zebra, giraffe
- **Furniture**: chair, couch, bed, dining table, toilet
- **Electronics**: TV, laptop, mouse, keyboard, cell phone, remote
- **Kitchen Items**: bottle, wine glass, cup, fork, knife, spoon, bowl, microwave, oven, toaster, sink, refrigerator
- **Indoor Objects**: book, clock, vase, scissors, teddy bear, potted plant
- **Outdoor Objects**: traffic light, fire hydrant, stop sign, parking meter, bench
- **Sports Equipment**: sports ball, baseball bat, skateboard, surfboard, tennis racket
- **And many more!**

**YOLOv8 Model Variants:**
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n.pt | 6 MB | Fastest | Good | Real-time, low-power devices |
| yolov8s.pt | 22 MB | Fast | Better | Balanced performance |
| yolov8m.pt | 52 MB | Medium | High | Better accuracy needed |
| yolov8l.pt | 88 MB | Slow | Very High | High accuracy, powerful hardware |
| yolov8x.pt | 136 MB | Slowest | Best | Maximum accuracy, GPU required |

### 2. Scene Recognition (BLIP)

BLIP (Bootstrapping Language-Image Pre-training) generates natural language descriptions of entire scenes:

**Capabilities:**

- **Full scene understanding**: "A person sitting at a desk working on a laptop in an office"
- **Context awareness**: Identifies room types (kitchen, office, bedroom, living room, bathroom)
- **Activity recognition**: Understands actions happening in the scene
- **Environment detection**: Distinguishes indoor vs outdoor scenes
- **Compositional understanding**: Describes relationships between objects

**Example Captions:**

- "A modern kitchen with stainless steel appliances and a person cooking"
- "A living room with a couch and television"
- "A person walking on a street with cars in the background"
- "An office desk with a laptop and books"

### 3. Spatial Awareness

The system provides detailed spatial information for each detected object:

**Horizontal Position:**

- **Left**: Object in left third of frame
- **Center**: Object in middle third of frame
- **Right**: Object in right third of frame

**Vertical Position:**

- **Top**: Object in upper third of frame
- **Middle**: Object in middle third of frame
- **Bottom**: Object in lower third of frame

**Distance Estimation** (based on relative object size):

- **Near**: Large object occupying >20% of frame
- **Medium**: Moderate size object (5-20% of frame)
- **Far**: Small object <5% of frame

**Example Spatial Description:**

- "I detect one person on the left, very close"
- "Two cars in the center, at medium distance"
- "A cup on the right, near"

### 4. Combined Descriptions

The system intelligently merges object detection with scene captioning:

**Example Combined Output:**

```
"This appears to be an indoor scene, specifically a kitchen.
The scene shows a person cooking at a stove with visible
objects including 1 person, 1 stove, and 2 utensils."
```

**Benefits of Combined Approach:**

- More comprehensive understanding than either model alone
- Context from scene recognition helps interpret object relationships
- Object detection provides specific counts and locations
- Natural language makes information accessible

### 5. Audio Feedback System

Intelligent text-to-speech system with multiple announcement types:

**Announcement Types:**

1. **New Object Announcements**: "New objects detected: person, laptop"
2. **Object Removal Announcements**: "Objects no longer visible: cup"
3. **Periodic Summaries**: "I detect 2 people and 1 car"
4. **Scene Captions**: "A kitchen with a table and chairs"
5. **Combined Descriptions**: Full scene + object information
6. **Spatial Descriptions**: Position and distance information

**Smart Features:**

- **Non-blocking**: Announcements don't interrupt processing
- **Change Detection**: Only announces when scene composition changes
- **Configurable Intervals**: Adjust announcement frequency
- **Priority System**: Important objects announced first
- **Stable Detection**: Requires multiple frames before announcing

## ⚡ Performance Optimization

### For Maximum Speed

1. **Use lightweight model**:

   ```python
   YOLO_MODEL = "yolov8n.pt"  # Fastest model
   ```

2. **Increase frame skipping**:

   ```python
   PROCESS_EVERY_N_FRAMES = 10  # Process fewer frames
   ```

3. **Reduce video resolution**:

   ```python
   RESIZE_FRAME = True
   FRAME_WIDTH = 320   # Lower resolution
   FRAME_HEIGHT = 240
   ```

4. **Disable unnecessary features**:

   ```python
   ENABLE_SCENE_RECOGNITION = False  # Disable scene captions
   SHOW_DISTANCE = False
   USE_SIMPLE_BOXES = True
   ```

5. **Reduce video source quality**: Lower resolution in camera settings

**Expected Performance**: 30-60 FPS on modern CPU

### For Maximum Accuracy

1. **Use larger model**:

   ```python
   YOLO_MODEL = "yolov8m.pt"  # or "yolov8l.pt"
   ```

2. **Process more frames**:

   ```python
   PROCESS_EVERY_N_FRAMES = 1  # Process every frame
   ```

3. **Higher confidence threshold**:

   ```python
   CONFIDENCE_THRESHOLD = 0.6  # Reduce false positives
   ```

4. **Enable all features**:
   ```python
   ENABLE_SCENE_RECOGNITION = True
   SCENE_NUM_BEAMS = 5  # Better quality captions
   ```

**Expected Performance**: 5-15 FPS on modern CPU

### For GPU Acceleration

1. **Install CUDA-enabled PyTorch**:

   ```bash
   # Visit https://pytorch.org/ for your specific CUDA version
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Enable GPU in config**:

   ```python
   ENABLE_GPU = True
   ```

3. **Verify GPU usage**:
   - Check console output for "Using GPU"
   - Monitor GPU usage with `nvidia-smi` (Windows/Linux)

**Expected Performance**: 60-120 FPS with modern GPU (GTX 1060 or better)

### Balanced Configuration (Recommended)

```python
# Object Detection
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
PROCESS_EVERY_N_FRAMES = 5

# Scene Recognition
ENABLE_SCENE_RECOGNITION = True
SCENE_CAPTION_INTERVAL = 10.0
SCENE_NUM_BEAMS = 4

# Performance
RESIZE_FRAME = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ENABLE_GPU = True  # If available

# Audio
ANNOUNCEMENT_INTERVAL = 5.0
```

**Expected Performance**: 20-30 FPS on CPU, 60+ FPS on GPU

### Performance Comparison

| Configuration | Hardware | FPS   | Detection Quality | Scene Captions |
| ------------- | -------- | ----- | ----------------- | -------------- |
| Ultra-Fast    | CPU      | 60+   | Good              | Disabled       |
| Balanced      | CPU      | 20-30 | Good              | Every 10s      |
| Balanced      | GPU      | 60+   | Good              | Every 10s      |
| High Accuracy | CPU      | 5-15  | Excellent         | Every 5s       |
| High Accuracy | GPU      | 30-60 | Excellent         | Every 5s       |

## 📱 Using IP Webcam (Android)

Transform your Android phone into a wireless IP camera:

### Step 1: Install IP Webcam App

Download from Google Play Store: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)

### Step 2: Start the Server

1. Open the IP Webcam app on your phone
2. Scroll down to the bottom
3. Tap **"Start server"**
4. Note the URL displayed (e.g., `http://192.168.1.100:8080`)

### Step 3: Configure the System

Edit `config/settings.py`:

```python
VIDEO_SOURCE = "http://YOUR_PHONE_IP:8080/video"
# Example: "http://192.168.1.100:8080/video"
```

### Step 4: Network Setup

**Important**: Ensure your phone and computer are on the same WiFi network

### Troubleshooting IP Webcam

- **Can't connect**: Check firewall settings on both devices
- **Laggy video**: Reduce video quality in IP Webcam settings
- **Connection drops**: Ensure phone screen stays on or use "Prevent phone sleep" option
- **Wrong URL**: Make sure to append `/video` to the IP address

### Alternative: DroidCam

Another option is [DroidCam](https://www.dev47apps.com/) which works similarly and supports USB connection mode.

## 🔮 Future Enhancements

### Planned Features

#### Advanced Vision Models

- [ ] **Depth Estimation**: Use MiDaS or DPT for accurate distance measurement
- [ ] **OCR Module**: Read text from signs, labels, documents, and menus using Tesseract/EasyOCR
- [ ] **Face Recognition**: Identify known individuals (with privacy controls)
- [ ] **Gesture Recognition**: Understand hand gestures and body language
- [ ] **Scene Segmentation**: Semantic segmentation for detailed scene understanding

#### Enhanced Accessibility

- [ ] **Multi-language Support**: Text-to-speech in multiple languages (Spanish, French, Hindi, etc.)
- [ ] **Voice Commands**: Control system with voice input
- [ ] **Haptic Feedback**: Vibration patterns for obstacle warnings (mobile integration)
- [ ] **Customizable Priorities**: User-defined object priorities and importance levels
- [ ] **Personalized Descriptions**: Adjust verbosity and detail level per user preference

#### Navigation & Safety

- [ ] **Obstacle Detection**: Specific warnings for navigation hazards
- [ ] **Crosswalk Detection**: Identify crosswalks and traffic signals
- [ ] **GPS Integration**: Location-aware descriptions and navigation assistance
- [ ] **Path Planning**: Safe route suggestions
- [ ] **Stair Detection**: Warn about stairs, slopes, and elevation changes

#### Mobile & Cloud

- [ ] **Mobile App**: Native iOS/Android app with better camera integration
- [ ] **Cloud Processing**: Offload computation to cloud for resource-constrained devices
- [ ] **Edge Computing**: Optimize for edge devices (Raspberry Pi, Jetson Nano)
- [ ] **Web Interface**: Browser-based control panel
- [ ] **Remote Monitoring**: Allow caregivers to monitor system status

#### Data & Intelligence

- [ ] **Recording Mode**: Save and replay sessions for training/review
- [ ] **Activity Logging**: Track detection history and patterns
- [ ] **Learning Mode**: Improve accuracy with user feedback
- [ ] **Context Memory**: Remember frequently visited locations and objects
- [ ] **Smart Notifications**: Predictive alerts based on learned patterns

#### Technical Improvements

- [ ] **Model Quantization**: INT8 quantization for faster inference
- [ ] **Multi-camera Support**: Simultaneous processing from multiple cameras
- [ ] **Video Stabilization**: Smooth shaky camera footage
- [ ] **Low-light Enhancement**: Better performance in dark environments
- [ ] **Battery Optimization**: Power-saving modes for mobile devices

### Community Contributions

We welcome contributions! Areas where help is especially appreciated:

- 🌍 Translation and localization
- 🧪 Testing on different hardware configurations
- 📱 Mobile app development
- 🎨 UI/UX improvements
- 📝 Documentation and tutorials
- 🐛 Bug reports and fixes

### Roadmap

**Q1 2026:**

- Depth estimation integration
- OCR module implementation
- Multi-language support (Phase 1)

**Q2 2026:**

- Mobile app beta release
- Voice commands
- Cloud processing option

**Q3 2026:**

- Enhanced navigation features
- GPS integration
- Context memory system

**Q4 2026:**

- Edge device optimization
- Advanced safety features
- Production-ready release

## 🔧 Troubleshooting

### Video Source Issues

**Problem**: "Could not open video source" or "Failed to grab frame"

**Solutions**:

- ✅ Verify IP camera is running and accessible (open URL in web browser)
- ✅ Check that the URL in `config/settings.py` is correct (including `/video` suffix)
- ✅ Try using webcam index instead: `VIDEO_SOURCE = 0` (or 1, 2)
- ✅ Ensure phone and computer are on same WiFi network
- ✅ Check firewall settings aren't blocking connection
- ✅ Test with different port or IP address
- ✅ For USB webcam, try different index numbers (0, 1, 2)

### Model Loading Issues

**Problem**: "Failed to initialize object detector" or "Error loading model"

**Solutions**:

- ✅ Ensure all dependencies installed: `pip install -r requirements.txt`
- ✅ Check internet connection (models download on first run)
- ✅ Try reinstalling ultralytics: `pip install --upgrade ultralytics`
- ✅ Try reinstalling transformers: `pip install --upgrade transformers`
- ✅ Delete model cache and re-download:
  - Windows: `%USERPROFILE%\.cache\torch\hub\ultralytics`
  - Linux/Mac: `~/.cache/torch/hub/ultralytics`

### Performance Issues

**Problem**: Slow performance, low FPS, or system lag

**Solutions**:

- ✅ Use smaller model: `YOLO_MODEL = "yolov8n.pt"`
- ✅ Increase frame skip: `PROCESS_EVERY_N_FRAMES = 10`
- ✅ Enable frame resize: `RESIZE_FRAME = True`
- ✅ Reduce resolution: `FRAME_WIDTH = 320`, `FRAME_HEIGHT = 240`
- ✅ Disable scene recognition: `ENABLE_SCENE_RECOGNITION = False`
- ✅ Close other resource-intensive applications
- ✅ Lower video quality in camera settings
- ✅ Check CPU/GPU usage in Task Manager

### Audio Issues

**Problem**: No audio output or text-to-speech not working

**Solutions**:

- ✅ Check system audio is not muted
- ✅ Verify `ENABLE_AUDIO = True` in config
- ✅ Install/reinstall pyttsx3: `pip install --upgrade pyttsx3`
- ✅ **Linux users**: Install espeak: `sudo apt-get install espeak`
- ✅ **macOS users**: Install say: `brew install espeak`
- ✅ Try disabling and re-enabling with 'A' key
- ✅ Check if other applications can use TTS
- ✅ Try different speech engine (system dependent)

### CUDA/GPU Issues

**Problem**: GPU not being used or CUDA errors

**Solutions**:

- ✅ Verify NVIDIA GPU is installed: `nvidia-smi`
- ✅ Install CUDA toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
- ✅ Install CUDA-enabled PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- ✅ Check PyTorch CUDA availability:
  ```python
  import torch
  print(torch.cuda.is_available())  # Should return True
  ```
- ✅ Update GPU drivers
- ✅ Set `ENABLE_GPU = False` to use CPU instead

### Memory Issues

**Problem**: Out of memory errors or system crashes

**Solutions**:

- ✅ Use smaller model: `YOLO_MODEL = "yolov8n.pt"`
- ✅ Reduce frame resolution: `FRAME_WIDTH = 320`
- ✅ Increase frame skip: `PROCESS_EVERY_N_FRAMES = 10`
- ✅ Disable scene recognition: `ENABLE_SCENE_RECOGNITION = False`
- ✅ Close other applications
- ✅ Restart system
- ✅ For GPU: Reduce batch size or disable GPU

### Import Errors

**Problem**: "ModuleNotFoundError" or "ImportError"

**Solutions**:

- ✅ Activate virtual environment:
  ```bash
  # Windows
  .\venv\Scripts\Activate.ps1
  # Linux/Mac
  source venv/bin/activate
  ```
- ✅ Reinstall all dependencies: `pip install -r requirements.txt`
- ✅ Check Python version: `python --version` (should be 3.8+)
- ✅ Update pip: `pip install --upgrade pip`
- ✅ Clear pip cache: `pip cache purge`

### Display Issues

**Problem**: No window appears or window crashes

**Solutions**:

- ✅ Check OpenCV installation: `pip install --upgrade opencv-python`
- ✅ **Linux users**: Install display dependencies:
  ```bash
  sudo apt-get install python3-opencv
  sudo apt-get install libgl1-mesa-glx
  ```
- ✅ **Remote systems**: OpenCV requires display; use headless mode or VNC
- ✅ Try different video backend (cv2.CAP_DSHOW, cv2.CAP_V4L2)
- ✅ Check if other OpenCV apps work

### Getting Help

If you continue experiencing issues:

1. Check the [Issues](https://github.com/AamishB/Scene-Description/issues) page on GitHub
2. Create a new issue with:
   - Error message (full traceback)
   - Your configuration (`config/settings.py`)
   - System information (OS, Python version, GPU)
   - Steps to reproduce
3. Include relevant logs from console output

## 📦 Requirements

### Core Dependencies

```
opencv-python          # Computer vision and video processing
ultralytics           # YOLOv8 object detection
torch                 # PyTorch deep learning framework
torchvision          # PyTorch vision utilities
transformers         # Hugging Face transformers for BLIP
Pillow               # Image processing
numpy                # Numerical computing
pyttsx3              # Text-to-speech engine
sentencepiece        # Tokenization for transformers
protobuf             # Protocol buffers for model serialization
```

### System Requirements

**Minimum:**

- Python 3.8+
- 4 GB RAM
- 2 GHz CPU
- 2 GB storage (for models)
- Webcam or IP camera

**Recommended:**

- Python 3.9+
- 8 GB RAM
- Quad-core CPU (Intel i5/AMD Ryzen 5 or better)
- 5 GB storage
- NVIDIA GPU with 4+ GB VRAM (for GPU acceleration)

**Optional (for GPU acceleration):**

- NVIDIA GPU (GTX 1060 or better)
- CUDA Toolkit 11.8+
- cuDNN 8.6+

### Supported Platforms

- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Linux (Ubuntu 18.04+, Debian, Fedora, etc.)

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

We welcome contributions to improve the Scene Description System! This is an open-source project designed to help visually impaired individuals, and community involvement is crucial.

### How to Contribute

1. **Fork the Repository**

   ```bash
   git clone https://github.com/AamishB/Scene-Description
   cd Scene-Description
   ```

2. **Create a Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**

   - Follow existing code style and patterns
   - Add comments for complex logic
   - Update documentation as needed

4. **Test Your Changes**

   - Test with different video sources
   - Verify audio feedback works
   - Check performance impact

5. **Commit and Push**

   ```bash
   git add .
   git commit -m "Add: description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Describe your changes clearly
   - Reference any related issues

### Adding New Models

The project has a modular architecture that makes it easy to add new AI models:

1. **Create a new module** in `models/` directory:

   ```python
   # models/your_model.py
   class YourModel:
       def __init__(self, model_name, device='cpu'):
           # Initialize your model
           pass

       def process(self, frame):
           # Process frame and return results
           pass
   ```

2. **Update `models/__init__.py`**:

   ```python
   from .your_model import YourModel
   ```

3. **Integrate in `cv.py`**:

   ```python
   from models.your_model import YourModel

   your_model = YourModel()
   results = your_model.process(frame)
   ```

4. **Add configuration** to `config/settings.py`:
   ```python
   ENABLE_YOUR_MODEL = True
   YOUR_MODEL_PARAM = "value"
   ```

### Code Style Guidelines

- Use **descriptive variable names**
- Add **docstrings** to all classes and functions
- Follow **PEP 8** style guide
- Keep functions **focused and small**
- Add **type hints** where appropriate
- Write **comments** for complex logic

### Testing Guidelines

- Test on multiple platforms (Windows, macOS, Linux)
- Test with different video sources (IP camera, webcam, video file)
- Test performance on both CPU and GPU
- Verify audio feedback works correctly
- Check for memory leaks with long-running sessions

### Documentation Guidelines

- Update README.md for new features
- Add inline code comments
- Include usage examples
- Document configuration options
- Update troubleshooting section as needed

### Reporting Bugs

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: Detailed steps to reproduce the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, hardware
- **Configuration**: Your `settings.py` configuration
- **Error Messages**: Full error traceback
- **Screenshots**: If applicable

### Suggesting Features

Feature requests are welcome! Please include:

- **Use Case**: Why this feature is needed
- **Description**: Detailed description of the feature
- **Benefits**: How it helps visually impaired users
- **Implementation Ideas**: If you have suggestions
- **Examples**: Similar features in other projects

## 📄 License

This project is licensed under the **MIT License** - see below for details.

```
MIT License

Copyright (c) 2025 Aamish Baloch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Licenses

This project uses the following open-source libraries:

- **YOLOv8** by Ultralytics (AGPL-3.0)
- **BLIP** by Salesforce (BSD-3-Clause)
- **PyTorch** by Meta AI (BSD-3-Clause)
- **OpenCV** (Apache-2.0)
- **Transformers** by Hugging Face (Apache-2.0)

Feel free to use, modify, and distribute this software for any purpose, including commercial applications.

## 🙏 Acknowledgments

This project wouldn't be possible without these amazing open-source projects and communities:

### AI Models & Frameworks

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** by Ultralytics - State-of-the-art object detection
- **[BLIP](https://github.com/salesforce/BLIP)** by Salesforce Research - Vision-language understanding
- **[PyTorch](https://pytorch.org/)** by Meta AI - Deep learning framework
- **[Transformers](https://huggingface.co/transformers/)** by Hugging Face - NLP and vision-language models

### Libraries & Tools

- **[OpenCV](https://opencv.org/)** - Computer vision and image processing
- **[pyttsx3](https://github.com/nateshmbhat/pyttsx3)** - Text-to-speech conversion
- **[NumPy](https://numpy.org/)** - Numerical computing

### Community

- OpenCV community for excellent documentation
- Hugging Face community for model hosting
- GitHub community for feedback and contributions

### Inspiration

Built with the goal of improving accessibility and independence for visually impaired individuals. Inspired by existing assistive technologies and the need for affordable, open-source solutions.

### Special Thanks

- To all contributors who help improve this project
- To the accessibility community for valuable feedback
- To researchers and developers working on assistive technologies

---

## 📞 Contact & Support

### Author

**Aamish Baloch**

- GitHub: [@AamishB](https://github.com/AamishB)
- Repository: [Scene-Description](https://github.com/AamishB/Scene-Description)

### Getting Help

- 📖 Check the [Documentation](#) in this README
- 🐛 Report bugs via [GitHub Issues](https://github.com/AamishB/Scene-Description/issues)
- 💡 Request features via [GitHub Issues](https://github.com/AamishB/Scene-Description/issues)
- 💬 Ask questions in [Discussions](https://github.com/AamishB/Scene-Description/discussions)

### Quick Links

- [Installation Guide](#-installation)
- [Configuration](#️-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ⚠️ Important Disclaimer

**This system is designed as an assistive tool and should NOT be used as the sole means of navigation or safety for visually impaired individuals.**

### Safety Guidelines

- ✋ Always use in conjunction with traditional mobility aids (white cane, guide dog, etc.)
- ✋ Do not rely solely on this system for navigation or obstacle avoidance
- ✋ Be aware that AI models can make mistakes or miss important objects
- ✋ Test thoroughly in safe environments before use in unfamiliar areas
- ✋ Keep audio volume at safe levels to maintain environmental awareness
- ✋ Use appropriate safety techniques and training for visually impaired mobility

### Limitations

- Detection accuracy depends on lighting, camera quality, and object visibility
- Scene recognition may not be 100% accurate in all situations
- System performance varies based on hardware capabilities
- Audio descriptions may lag behind real-time scene changes
- Not suitable for high-speed navigation or critical safety decisions

### Responsible Use

- This is an assistive technology, not a replacement for human judgment
- Users should receive proper training before depending on the system
- Supervise initial use in controlled environments
- Report any critical issues or safety concerns immediately
- Regular testing and calibration recommended

---

## 🚀 Quick Start Guide

Ready to get started? Follow these steps:

1. **Download IP Webcam**: [Google Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam)
2. **Clone Repository**: `git clone https://github.com/AamishB/Scene-Description`
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Configure Camera**: Edit `config/settings.py` with your IP camera URL
5. **Run System**: `python cv.py`

For detailed instructions, see the [Installation](#-installation) section.

---

<div align="center">

**Made with ❤️ for accessibility**

⭐ If you find this project useful, please consider giving it a star on GitHub! ⭐

[Report Bug](https://github.com/AamishB/Scene-Description/issues) · [Request Feature](https://github.com/AamishB/Scene-Description/issues) · [Contribute](https://github.com/AamishB/Scene-Description/pulls)

</div>
