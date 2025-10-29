"""
Audio Feedback Module
Provides text-to-speech capabilities for scene descriptions
"""

import pyttsx3
import threading
import time
from queue import Queue


class AudioFeedback:
    """
    Text-to-speech engine for providing audio descriptions
    Supports asynchronous speech to avoid blocking video processing
    """
    
    def __init__(self, rate=150, volume=0.9, voice_index=0):
        """
        Initialize text-to-speech engine
        
        Args:
            rate (int): Speech rate in words per minute (default: 150)
            volume (float): Volume level from 0.0 to 1.0 (default: 0.9)
            voice_index (int): Voice index to use (default: 0)
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Set voice if available
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > voice_index:
                self.engine.setProperty('voice', voices[voice_index].id)
            
            self.is_speaking = False
            self.speech_queue = Queue()
            self.min_interval = 3.0  # Minimum seconds between announcements
            self.last_speech_time = 0
            self.enabled = True
            
            # Start speech worker thread
            self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.worker_thread.start()
            
            print("Audio feedback initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize text-to-speech: {e}")
            self.engine = None
            self.enabled = False
    
    def speak(self, text, priority=False):
        """
        Queue text to be spoken
        
        Args:
            text (str): Text to speak
            priority (bool): If True, speak immediately; otherwise respect interval
        """
        if not self.enabled or self.engine is None:
            return
        
        current_time = time.time()
        
        # Check if enough time has passed since last speech
        if priority or (current_time - self.last_speech_time >= self.min_interval):
            self.speech_queue.put(text)
    
    def _speech_worker(self):
        """Worker thread that processes speech queue"""
        while True:
            try:
                text = self.speech_queue.get()
                if text and self.engine:
                    self.is_speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.last_speech_time = time.time()
                    self.is_speaking = False
                    time.sleep(0.1)  # Small delay between speeches
            except Exception as e:
                print(f"Speech error: {e}")
                self.is_speaking = False
    
    def speak_blocking(self, text):
        """
        Speak text and wait for completion (blocking)
        
        Args:
            text (str): Text to speak
        """
        if not self.enabled or self.engine is None:
            return
        
        try:
            self.is_speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
            self.last_speech_time = time.time()
            self.is_speaking = False
        except Exception as e:
            print(f"Speech error: {e}")
            self.is_speaking = False
    
    def set_rate(self, rate):
        """Set speech rate"""
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """Set volume level (0.0 to 1.0)"""
        if self.engine:
            self.engine.setProperty('volume', volume)
    
    def set_interval(self, seconds):
        """Set minimum interval between speeches"""
        self.min_interval = seconds
    
    def enable(self):
        """Enable audio feedback"""
        self.enabled = True
    
    def disable(self):
        """Disable audio feedback"""
        self.enabled = False
    
    def is_enabled(self):
        """Check if audio feedback is enabled"""
        return self.enabled
    
    def clear_queue(self):
        """Clear pending speech queue"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except:
                break
    
    def generate_scene_description(self, summary, spatial_info=None):
        """Generate natural language scene description"""
        if not summary:
            return "No objects detected."
        
        total_count = sum(info['count'] for info in summary.values())
        
        # Single object type
        if len(summary) == 1:
            obj_name, count = list(summary.items())[0][0], list(summary.items())[0][1]['count']
            desc = f"I see {'one ' + obj_name if count == 1 else f'{count} {obj_name}' + ('s' if count > 1 else '')}"
        
        # Two object types
        elif len(summary) == 2:
            items = list(summary.items())
            obj1 = f"{items[0][1]['count']} {items[0][0]}{'s' if items[0][1]['count'] > 1 else ''}"
            obj2 = f"{items[1][1]['count']} {items[1][0]}{'s' if items[1][1]['count'] > 1 else ''}"
            desc = f"I see {obj1} and {obj2}"
        
        # Multiple objects
        else:
            top_3 = sorted(summary.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
            obj_list = [f"{count['count']} {name}{'s' if count['count'] > 1 else ''}" 
                       for name, count in top_3]
            desc = f"I see {total_count} objects including {', '.join(obj_list)}"
        
        # Add spatial info for main object
        if spatial_info and len(spatial_info) > 0:
            obj = spatial_info[0]
            desc += f". The main {obj['object']} is on the {obj.get('h_position', '')}"
            dist = obj.get('distance', '')
            if dist == "near":
                desc += ", very close to you"
            elif dist == "far":
                desc += ", in the distance"
        
        return desc + "."
    
    def announce_change(self, new_objects, removed_objects):
        """Announce changes in the scene (new or removed objects)"""
        announcements = []
        
        if new_objects:
            obj_list = ', '.join(new_objects) if len(new_objects) > 1 else new_objects[0]
            announcements.append(f"New {'objects detected: ' + obj_list if len(new_objects) > 1 else obj_list + ' detected'}")
        
        if removed_objects:
            obj_list = ', '.join(removed_objects) if len(removed_objects) > 1 else removed_objects[0]
            announcements.append(f"{'Objects no longer visible: ' + obj_list if len(removed_objects) > 1 else obj_list + ' is no longer visible'}")
        
        if announcements:
            self.speak(". ".join(announcements))
