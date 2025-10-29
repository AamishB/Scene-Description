"""
Configuration Settings for Scene Description Project
Centralized configuration management
"""


class Config:
    """Configuration class for scene description system"""
    
    # Video Source Settings
    VIDEO_SOURCE = "http://10.187.108.182:8080/video"  # IP Webcam URL or camera index (0, 1, etc.)
    
    # Object Detection Settings
    YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    CONFIDENCE_THRESHOLD = 0.5  # Detection confidence threshold (0.0 to 1.0)
    DEVICE = "cpu"  # Options: 'cpu' or 'cuda'
    
    # Performance Settings
    PROCESS_EVERY_N_FRAMES = 5  # Process every Nth frame (higher = faster, less accurate)
    ENABLE_GPU = False  # Set to True if CUDA-capable GPU available
    
    # Frame optimization
    RESIZE_FRAME = True  # Resize frames for faster processing
    FRAME_WIDTH = 640  # Resize width (lower = faster, 640 is good balance)
    FRAME_HEIGHT = 480  # Resize height (lower = faster)
    
    # Display optimization
    SKIP_DISPLAY_PROCESSING = True  # Skip unnecessary display processing
    USE_SIMPLE_BOXES = True  # Use simple boxes instead of enhanced (faster)
    
    # Display Settings
    SHOW_BOUNDING_BOXES = True
    SHOW_CONFIDENCE = True
    SHOW_DISTANCE = True
    SHOW_FPS = True
    WINDOW_NAME = "Scene Description - Object Detection"
    
    # Audio Feedback Settings
    ENABLE_AUDIO = True
    SPEECH_RATE = 150  # Words per minute
    SPEECH_VOLUME = 0.9  # 0.0 to 1.0
    ANNOUNCEMENT_INTERVAL = 5.0  # Minimum seconds between audio announcements
    
    # Detection Behavior
    MAX_OBJECTS_TO_DESCRIBE = 5  # Maximum objects to include in description
    PRIORITY_CLASSES = [
        'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
        'dog', 'cat', 'traffic light', 'stop sign', 'chair', 'couch',
        'bottle', 'cup', 'book', 'laptop', 'cell phone', 'door'
    ]
    
    # Tracking Settings
    STABLE_DETECTION_FRAMES = 3  # Frames required for stable detection
    DETECTION_HISTORY_SECONDS = 2.0  # How long to keep detection history
    
    # Scene Recognition Settings
    ENABLE_SCENE_RECOGNITION = True  # Enable scene captioning
    SCENE_MODEL = "Salesforce/blip-image-captioning-base"  # BLIP model for scene captioning
    SCENE_CAPTION_INTERVAL = 10.0  # Seconds between scene caption updates
    SCENE_MAX_LENGTH = 50  # Maximum caption length
    SCENE_NUM_BEAMS = 4  # Beam search beams (higher = better quality, slower)
    COMBINE_SCENE_AND_OBJECTS = True  # Combine scene caption with object detection
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device based on configuration"""
        if cls.ENABLE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"Warning: Unknown configuration key: {key}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*50)
        print("SCENE DESCRIPTION CONFIGURATION")
        print("="*50)
        print(f"Video Source: {cls.VIDEO_SOURCE}")
        print(f"YOLO Model: {cls.YOLO_MODEL}")
        print(f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}")
        print(f"Device: {cls.get_device()}")
        print(f"Process Every N Frames: {cls.PROCESS_EVERY_N_FRAMES}")
        print(f"Frame Resize: {'Enabled' if cls.RESIZE_FRAME else 'Disabled'} ({cls.FRAME_WIDTH}x{cls.FRAME_HEIGHT})")
        print(f"Scene Recognition: {'Enabled' if cls.ENABLE_SCENE_RECOGNITION else 'Disabled'}")
        print(f"Audio Feedback: {'Enabled' if cls.ENABLE_AUDIO else 'Disabled'}")
        print(f"Show Bounding Boxes: {'Yes' if cls.SHOW_BOUNDING_BOXES else 'No'}")
        print("="*50 + "\n")
