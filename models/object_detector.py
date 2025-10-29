"""
Object Detection Module using YOLOv8
Handles real-time object detection for scene description
"""

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time


class ObjectDetector:
    """
    YOLO-based object detector for scene understanding
    Provides object detection, tracking, and spatial awareness
    """
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, device='cpu'):
        """
        Initialize YOLO object detector
        
        Args:
            model_name (str): YOLOv8 model variant 
                             Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
                             n=nano (fastest), s=small, m=medium, l=large, x=xlarge (most accurate)
            conf_threshold (float): Confidence threshold for detections (0.0 to 1.0)
            device (str): Device to run inference on ('cpu' or 'cuda')
        """
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Set model to appropriate device
        if device == 'cuda':
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print("Using GPU for inference")
                else:
                    print("CUDA not available, using CPU")
                    self.device = 'cpu'
            except ImportError:
                print("PyTorch not found, using CPU")
                self.device = 'cpu'
        else:
            print("Using CPU for inference")
        
        # Tracking variables
        self.tracked_objects = {}
        self.object_id_counter = 0
        self.last_detection_time = time.time()
        
        # History for stable descriptions
        self.detection_history = defaultdict(list)
        self.history_length = 5  # Keep last 5 frames of history
        
        print(f"Object Detector initialized successfully!")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"Available classes: {len(self.model.names)}")
        
    def detect(self, frame, verbose=False):
        """
        Detect objects in a frame
        
        Args:
            frame (np.ndarray): Input image/frame
            verbose (bool): Whether to print detection details
            
        Returns:
            tuple: (results, annotated_frame)
                - results: YOLO detection results object
                - annotated_frame: Frame with bounding boxes and labels
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False, device=self.device)
        
        # Get annotated frame with bounding boxes
        annotated_frame = results[0].plot()
        
        if verbose and len(results[0].boxes) > 0:
            print(f"Detected {len(results[0].boxes)} objects")
        
        return results, annotated_frame
    
    def get_object_summary(self, results):
        """
        Generate a summary of detected objects with counts
        
        Args:
            results: YOLO detection results
            
        Returns:
            dict: Dictionary with object counts and confidence scores
                  Format: {class_name: {'count': int, 'confidence': [float, ...]}}
        """
        summary = {}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                confidence = float(box.conf[0])
                
                if class_name not in summary:
                    summary[class_name] = {
                        'count': 0,
                        'confidence': [],
                        'avg_confidence': 0.0
                    }
                
                summary[class_name]['count'] += 1
                summary[class_name]['confidence'].append(confidence)
        
        # Calculate average confidence for each class
        for class_name in summary:
            confidences = summary[class_name]['confidence']
            summary[class_name]['avg_confidence'] = sum(confidences) / len(confidences)
        
        return summary
    
    def _calculate_position(self, center, dimension):
        """Calculate position (left/center/right or top/middle/bottom)"""
        if center < dimension / 3:
            return ["left", "top"][dimension == 'v']
        elif center < 2 * dimension / 3:
            return ["center", "middle"][dimension == 'v']
        return ["right", "bottom"][dimension == 'v']
    
    def _get_distance_from_size(self, relative_size):
        """Calculate relative distance based on object size"""
        return "near" if relative_size > 0.15 else "medium" if relative_size > 0.05 else "far"
    
    def get_spatial_info(self, results, frame_width, frame_height):
        """
        Get spatial information about detected objects
        
        Args:
            results: YOLO detection results
            frame_width (int): Width of the frame
            frame_height (int): Height of the frame
            
        Returns:
            list: Sorted list of object information dictionaries
        """
        spatial_info = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                relative_size = ((x2 - x1) * (y2 - y1)) / (frame_width * frame_height)
                
                # Calculate positions using helper
                h_pos = "left" if center_x < frame_width / 3 else "center" if center_x < 2 * frame_width / 3 else "right"
                v_pos = "top" if center_y < frame_height / 3 else "middle" if center_y < 2 * frame_height / 3 else "bottom"
                
                spatial_info.append({
                    'object': result.names[int(box.cls[0])],
                    'h_position': h_pos,
                    'v_position': v_pos,
                    'distance': self._get_distance_from_size(relative_size),
                    'confidence': float(box.conf[0]),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (int(center_x), int(center_y)),
                    'size': relative_size
                })
        
        return sorted(spatial_info, key=lambda x: x['size'], reverse=True)
    
    def get_priority_objects(self, spatial_info, priority_classes=None):
        """
        Filter and prioritize important objects for scene description
        
        Args:
            spatial_info (list): Spatial information from get_spatial_info()
            priority_classes (list): List of high-priority object classes
            
        Returns:
            list: Filtered and prioritized objects
        """
        if priority_classes is None:
            priority_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
                              'dog', 'cat', 'traffic light', 'stop sign', 'chair', 'couch']
        
        # Partition into priority and other objects
        priority, other = [], []
        for obj in spatial_info:
            (priority if obj['object'] in priority_classes else other).append(obj)
        
        return priority + other
    
    def generate_description(self, summary, spatial_info, max_objects=5):
        """
        Generate natural language description of the scene
        
        Args:
            summary (dict): Object summary from get_object_summary()
            spatial_info (list): Spatial info from get_spatial_info()
            max_objects (int): Maximum number of objects to describe
            
        Returns:
            str: Natural language scene description
        """
        if not summary:
            return "No objects detected in the scene."
        
        priority_info = self.get_priority_objects(spatial_info)[:max_objects]
        total = sum(obj['count'] for obj in summary.values())
        
        # Build overview
        if len(summary) == 1:
            obj_name, count = list(summary.items())[0][0], list(summary.items())[0][1]['count']
            overview = f"I detect {'one ' + obj_name if count == 1 else f'{count} {obj_name}s'}"
        else:
            overview = f"I detect {total} objects"
        
        # Add spatial details
        spatial_descs = []
        for obj in priority_info[:3]:
            desc = f"a {obj['object']} on the {obj['h_position']}"
            if obj['distance'] == "near":
                desc += ", very close"
            elif obj['distance'] == "far":
                desc += ", in the distance"
            spatial_descs.append(desc)
        
        return f"{overview}. {'including ' + ', '.join(spatial_descs) + '.' if spatial_descs else ''}"
    
    def update_detection_history(self, summary):
        """
        Update detection history for stable descriptions
        
        Args:
            summary (dict): Current frame's object summary
        """
        current_time = time.time()
        
        # Add current detections to history
        for obj_name, obj_data in summary.items():
            self.detection_history[obj_name].append({
                'time': current_time,
                'count': obj_data['count'],
                'confidence': obj_data['avg_confidence']
            })
        
        # Remove old history (keep only recent frames)
        cutoff_time = current_time - 2.0  # Keep last 2 seconds
        for obj_name in list(self.detection_history.keys()):
            self.detection_history[obj_name] = [
                entry for entry in self.detection_history[obj_name]
                if entry['time'] > cutoff_time
            ]
            
            # Remove empty histories
            if not self.detection_history[obj_name]:
                del self.detection_history[obj_name]
    
    def get_stable_summary(self):
        """
        Get stable object summary based on detection history
        Reduces flickering in detections
        
        Returns:
            dict: Stable summary of consistently detected objects
        """
        stable_summary = {}
        
        for obj_name, history in self.detection_history.items():
            if len(history) >= 2:  # Object detected in at least 2 recent frames
                avg_count = int(np.mean([entry['count'] for entry in history]))
                avg_confidence = np.mean([entry['confidence'] for entry in history])
                
                stable_summary[obj_name] = {
                    'count': avg_count,
                    'avg_confidence': avg_confidence,
                    'detection_frequency': len(history)
                }
        
        return stable_summary
    
    def draw_enhanced_boxes(self, frame, spatial_info, show_distance=True):
        """
        Draw enhanced bounding boxes with additional information
        
        Args:
            frame (np.ndarray): Input frame
            spatial_info (list): Spatial information about objects
            show_distance (bool): Whether to show distance information
            
        Returns:
            np.ndarray: Frame with enhanced annotations
        """
        annotated_frame = frame.copy()
        
        for obj in spatial_info:
            x1, y1, x2, y2 = obj['bbox']
            label = obj['object']
            confidence = obj['confidence']
            distance = obj['distance']
            
            # Color based on distance (green=near, yellow=medium, red=far)
            if distance == "near":
                color = (0, 255, 0)  # Green
            elif distance == "medium":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if show_distance:
                text = f"{label} ({confidence:.2f}) - {distance}"
            else:
                text = f"{label} ({confidence:.2f})"
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated_frame
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'yolov8',
            'num_classes': len(self.model.names),
            'classes': list(self.model.names.values()),
            'device': self.device,
            'conf_threshold': self.conf_threshold
        }
