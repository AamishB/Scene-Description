"""
Scene Recognition Module using Vision-Language Models
Generates natural language descriptions of entire scenes
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import cv2
from collections import deque
import time


class SceneRecognizer:
    """
    Scene recognition and captioning using BLIP model
    Generates natural language descriptions of scenes for visually impaired users
    """
    
    def __init__(self, model_name='Salesforce/blip-image-captioning-base', device='cpu'):
        """
        Initialize Scene Recognizer with BLIP model
        
        Args:
            model_name (str): Hugging Face model name
                             Options: 
                             - 'Salesforce/blip-image-captioning-base' (balanced)
                             - 'Salesforce/blip-image-captioning-large' (better quality, slower)
            device (str): Device to run on ('cpu' or 'cuda')
        """
        print(f"Loading Scene Recognition model: {model_name}...")
        self.device = device
        self.model_name = model_name
        
        try:
            # Load BLIP model and processor
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            # Move model to device
            if device == 'cuda':
                try:
                    if torch.cuda.is_available():
                        self.model = self.model.to('cuda')
                        print("Using GPU for scene recognition")
                    else:
                        print("CUDA not available, using CPU")
                        self.device = 'cpu'
                except Exception as e:
                    print(f"GPU initialization failed: {e}, using CPU")
                    self.device = 'cpu'
            else:
                print("Using CPU for scene recognition")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Caption history for stability
            self.caption_history = deque(maxlen=5)
            self.last_caption = ""
            self.last_caption_time = 0
            self.min_caption_interval = 3.0  # Minimum seconds between new captions
            
            # Scene context tracking
            self.scene_contexts = []
            self.context_keywords = {
                'indoor': ['room', 'table', 'chair', 'wall', 'floor', 'ceiling', 'door', 'window'],
                'outdoor': ['street', 'road', 'tree', 'sky', 'building', 'car', 'park', 'grass'],
                'kitchen': ['kitchen', 'stove', 'refrigerator', 'sink', 'counter', 'appliance'],
                'office': ['desk', 'computer', 'office', 'monitor', 'keyboard', 'workplace'],
                'bedroom': ['bed', 'bedroom', 'pillow', 'blanket', 'nightstand'],
                'living_room': ['couch', 'sofa', 'tv', 'living room', 'coffee table'],
                'bathroom': ['bathroom', 'toilet', 'sink', 'shower', 'bathtub'],
                'vehicle': ['inside car', 'dashboard', 'steering wheel', 'vehicle interior']
            }
            
            print("Scene Recognizer initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing Scene Recognizer: {e}")
            raise
    
    def recognize_scene(self, frame, max_length=50, num_beams=4):
        """
        Generate caption for a scene
        
        Args:
            frame (np.ndarray): Input frame (BGR format from OpenCV)
            max_length (int): Maximum caption length
            num_beams (int): Number of beams for beam search (higher = better but slower)
            
        Returns:
            str: Generated scene caption
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Process image
            inputs = self.processor(pil_image, return_tensors="pt")
            
            # Move inputs to device
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate caption
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Add to history
            self.caption_history.append(caption)
            self.last_caption = caption
            self.last_caption_time = time.time()
            
            return caption
            
        except Exception as e:
            print(f"Error generating scene caption: {e}")
            return "Unable to describe scene"
    
    def get_stable_caption(self):
        """
        Get stable caption based on recent history
        Reduces flickering in scene descriptions
        
        Returns:
            str: Most common or recent stable caption
        """
        if not self.caption_history:
            return "Scene analysis in progress"
        
        # If we have multiple similar captions, use the most recent
        if len(self.caption_history) >= 3:
            # Check if recent captions are similar
            recent_captions = list(self.caption_history)[-3:]
            words_sets = [set(caption.lower().split()) for caption in recent_captions]
            
            # Calculate similarity (common words)
            if len(words_sets) >= 2:
                common_words = words_sets[0].intersection(*words_sets[1:])
                if len(common_words) >= 3:  # If 3+ common words, captions are stable
                    return recent_captions[-1]
        
        return self.last_caption
    
    def should_update_caption(self, force=False):
        """
        Check if caption should be updated based on time interval
        
        Args:
            force (bool): Force update regardless of interval
            
        Returns:
            bool: True if caption should be updated
        """
        if force:
            return True
        
        current_time = time.time()
        elapsed = current_time - self.last_caption_time
        
        return elapsed >= self.min_caption_interval
    
    def identify_scene_context(self, caption):
        """
        Identify scene context from caption (indoor/outdoor, room type, etc.)
        
        Args:
            caption (str): Scene caption
            
        Returns:
            dict: Scene context information
        """
        caption_lower = caption.lower()
        context = {
            'location_type': 'unknown',
            'room_type': 'unknown',
            'confidence': 0.0,
            'keywords': []
        }
        
        # Check for location type
        indoor_score = sum(1 for word in self.context_keywords['indoor'] if word in caption_lower)
        outdoor_score = sum(1 for word in self.context_keywords['outdoor'] if word in caption_lower)
        
        if indoor_score > outdoor_score:
            context['location_type'] = 'indoor'
            context['confidence'] = min(indoor_score / 3.0, 1.0)
        elif outdoor_score > indoor_score:
            context['location_type'] = 'outdoor'
            context['confidence'] = min(outdoor_score / 3.0, 1.0)
        
        # Check for specific room types
        room_scores = {}
        for room_type, keywords in self.context_keywords.items():
            if room_type not in ['indoor', 'outdoor']:
                score = sum(1 for word in keywords if word in caption_lower)
                if score > 0:
                    room_scores[room_type] = score
        
        if room_scores:
            best_room = max(room_scores, key=room_scores.get)
            if room_scores[best_room] >= 1:
                context['room_type'] = best_room.replace('_', ' ')
                context['confidence'] = min(room_scores[best_room] / 2.0, 1.0)
        
        # Extract keywords
        words = caption_lower.split()
        context['keywords'] = [w for w in words if len(w) > 3]
        
        return context
    
    def generate_detailed_description(self, caption, context, object_summary=None):
        """Generate detailed description combining scene caption with context"""
        parts = []
        
        # Add location context
        if context['confidence'] > 0.5:
            if context['location_type'] != 'unknown':
                parts.append(f"This appears to be an {context['location_type']} scene")
            if context['room_type'] != 'unknown':
                parts.append(f"specifically a {context['room_type']}")
        
        # Add main caption
        parts.append(f"The scene shows {caption}")
        
        # Add object information
        if object_summary:
            top_objects = sorted(object_summary.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
            if top_objects:
                obj_list = ', '.join(f"{info['count']} {name}{'s' if info['count'] > 1 else ''}" 
                                    for name, info in top_objects)
                parts.append(f"with visible objects including {obj_list}")
        
        return ". ".join(parts) + "."
    
    def compare_captions(self, caption1, caption2):
        """
        Compare two captions for similarity
        
        Args:
            caption1 (str): First caption
            caption2 (str): Second caption
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        words1 = set(caption1.lower().split())
        words2 = set(caption2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def enhance_caption_for_accessibility(self, caption):
        """Enhance caption for visually impaired users with more context"""
        # Add article if missing and capitalize
        if not caption[0].isupper():
            caption = caption[0].upper() + caption[1:]
        
        if not caption.startswith(('A ', 'An ', 'The ')):
            article = 'An' if caption[0].lower() in 'aeiou' else 'A'
            caption = f"{article} {caption}"
        
        # Ensure proper ending
        if not caption.endswith(('.', '!', '?')):
            caption += '.'
        
        return caption
    
    def get_scene_summary(self, caption, context):
        """Get concise scene summary for quick announcements"""
        parts = []
        
        if context['location_type'] != 'unknown':
            parts.append(context['location_type'])
        if context['room_type'] != 'unknown':
            parts.append(context['room_type'])
        
        key_phrase = ' '.join(caption.split()[:5])
        return f"{', '.join(parts)}: {key_phrase}" if parts else key_phrase
    
    def set_caption_interval(self, seconds):
        """Set minimum interval between caption updates"""
        self.min_caption_interval = seconds
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'caption_interval': self.min_caption_interval,
            'history_size': len(self.caption_history)
        }
    
    def clear_history(self):
        """Clear caption history"""
        self.caption_history.clear()
        self.last_caption = ""
        self.scene_contexts = []
