"""
Scene Description System - Main Entry Point
Real-time object detection and scene recognition for visually impaired assistance
"""

import cv2
import time
from models.object_detector import ObjectDetector
from models.scene_recognizer import SceneRecognizer
from utils.video_handler import VideoHandler
from utils.audio_feedback import AudioFeedback
from config.settings import Config


class SceneDescriptionSystem:
    """Main system orchestrator for scene description"""
    
    def __init__(self):
        self.video_handler = None
        self.detector = None
        self.scene_recognizer = None
        self.audio = None
        self.show_boxes = Config.SHOW_BOUNDING_BOXES
        self.frame_count = 0
        self.previous_objects = set()
        
        # Cache for processed results
        self.cache = {
            'results': None,
            'summary': {},
            'spatial_info': [],
            'annotated_frame': None,
            'scene_caption': '',
            'scene_context': {},
            'processing_times': []
        }
        
        # Timing
        self.last_description_time = time.time()
        self.last_scene_caption_time = 0
    
    def initialize_components(self):
        """Initialize all system components"""
        Config.print_config()
        
        # Initialize video handler
        print("Initializing video handler...")
        self.video_handler = VideoHandler(Config.VIDEO_SOURCE)
        if not self.video_handler.is_open():
            print("Error: Could not open video source!")
            print("Please check your video source configuration in config/settings.py")
            return False
        
        # Initialize object detector
        print("\nInitializing YOLO object detector...")
        try:
            self.detector = ObjectDetector(
                model_name=Config.YOLO_MODEL,
                conf_threshold=Config.CONFIDENCE_THRESHOLD,
                device=Config.get_device()
            )
        except Exception as e:
            print(f"Error initializing object detector: {e}")
            print("Make sure you have installed all requirements: pip install -r requirements.txt")
            self.cleanup()
            return False
        
        # Initialize scene recognizer
        if Config.ENABLE_SCENE_RECOGNITION:
            print("\nInitializing Scene Recognition model...")
            try:
                self.scene_recognizer = SceneRecognizer(
                    model_name=Config.SCENE_MODEL,
                    device=Config.get_device()
                )
                self.scene_recognizer.set_caption_interval(Config.SCENE_CAPTION_INTERVAL)
            except Exception as e:
                print(f"Warning: Scene Recognition unavailable: {e}")
                print("Continuing with object detection only...")
        
        # Initialize audio feedback
        if Config.ENABLE_AUDIO:
            print("\nInitializing audio feedback...")
            try:
                self.audio = AudioFeedback(rate=Config.SPEECH_RATE, volume=Config.SPEECH_VOLUME)
                self.audio.set_interval(Config.ANNOUNCEMENT_INTERVAL)
                start_message = "Scene description system started"
                if self.scene_recognizer:
                    start_message += " with scene recognition"
                start_message += ". Press Q to quit, C for scene caption."
                self.audio.speak_blocking(start_message)
            except Exception as e:
                print(f"Warning: Audio feedback unavailable: {e}")
        
        return True
    
    
    def print_controls(self):
        """Print system controls"""
        print("\n" + "="*50)
        print("SCENE DESCRIPTION SYSTEM RUNNING")
        print("="*50)
        print("Controls:")
        print("  Q - Quit  |  D - Toggle boxes  |  A - Toggle audio")
        print("  S - Speak scene  |  C - Scene caption  |  P - Print summary")
        print("  +/- - Frame skip  |  R - Toggle resize")
        print("="*50 + "\n")
    
    
    def process_frame(self, frame):
        """Process frame for object detection and scene recognition"""
        current_time = time.time()
        should_process = self.frame_count % Config.PROCESS_EVERY_N_FRAMES == 0
        
        if should_process:
            detect_start = time.time()
            
            # Perform object detection
            results, annotated_frame = self.detector.detect(frame)
            summary = self.detector.get_object_summary(results)
            self.detector.update_detection_history(summary)
            
            # Get spatial information
            height, width = frame.shape[:2]
            spatial_info = self.detector.get_spatial_info(results, width, height)
            
            # Update cache
            self.cache.update({
                'results': results,
                'summary': summary,
                'spatial_info': spatial_info,
                'annotated_frame': annotated_frame
            })
            
            # Track processing time
            detect_time = time.time() - detect_start
            self.cache['processing_times'].append(detect_time)
            if len(self.cache['processing_times']) > 30:
                self.cache['processing_times'].pop(0)
            
            # Handle scene changes
            self._handle_scene_changes(summary, current_time)
        
        # Update scene caption if needed
        if (self.scene_recognizer and 
            self.scene_recognizer.should_update_caption() and
            self.frame_count % (Config.PROCESS_EVERY_N_FRAMES * 3) == 0):
            self._update_scene_caption(frame)
    
    def _handle_scene_changes(self, summary, current_time):
        """Handle detection of new/removed objects"""
        current_objects = set(summary.keys())
        new_objects = current_objects - self.previous_objects
        removed_objects = self.previous_objects - current_objects
        
        if self.audio and self.audio.is_enabled():
            if new_objects or removed_objects:
                self.audio.announce_change(list(new_objects), list(removed_objects))
            
            # Periodic scene description
            if current_time - self.last_description_time >= Config.ANNOUNCEMENT_INTERVAL:
                if summary:
                    description = self._get_scene_description()
                    self.audio.speak(description)
                    self.last_description_time = current_time
        
        self.previous_objects = current_objects
    
    def _get_scene_description(self):
        """Get combined scene description"""
        if (Config.COMBINE_SCENE_AND_OBJECTS and self.scene_recognizer 
            and self.cache['scene_caption']):
            return self.scene_recognizer.generate_detailed_description(
                self.cache['scene_caption'],
                self.cache['scene_context'],
                self.cache['summary']
            )
        return self.detector.generate_description(
            self.cache['summary'],
            self.cache['spatial_info'],
            Config.MAX_OBJECTS_TO_DESCRIBE
        )
    
    def _update_scene_caption(self, frame):
        """Update scene caption"""
        scene_start = time.time()
        caption = self.scene_recognizer.recognize_scene(
            frame, max_length=Config.SCENE_MAX_LENGTH, num_beams=Config.SCENE_NUM_BEAMS
        )
        scene_time = time.time() - scene_start
        
        self.cache['scene_caption'] = caption
        self.cache['scene_context'] = self.scene_recognizer.identify_scene_context(caption)
        self.last_scene_caption_time = time.time()
        
        print(f"Scene: {caption} (took {scene_time:.2f}s)")
            
    
    def render_display_frame(self, frame):
        """Render frame with overlays and information"""
        # Choose base frame
        should_process = self.frame_count % Config.PROCESS_EVERY_N_FRAMES == 0
        if self.show_boxes and self.cache['annotated_frame'] is not None:
            if Config.USE_SIMPLE_BOXES or not Config.SHOW_DISTANCE:
                display_frame = self.cache['annotated_frame'].copy() if should_process else self.cache['annotated_frame']
            else:
                display_frame = self.detector.draw_enhanced_boxes(
                    frame, self.cache['spatial_info'], show_distance=True
                ) if should_process else self.cache['annotated_frame']
        else:
            display_frame = frame
        
        # Add FPS and performance stats
        if Config.SHOW_FPS:
            self._add_performance_overlay(display_frame)
        
        # Add scene caption
        if self.scene_recognizer and self.cache['scene_caption']:
            self._add_caption_overlay(display_frame, self.cache['scene_caption'])
        
        # Add status indicators
        if self.audio and not self.audio.is_enabled():
            cv2.putText(display_frame, "Audio: OFF", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return display_frame
    
    def _add_performance_overlay(self, frame):
        """Add FPS and performance information to frame"""
        fps = self.video_handler.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if self.cache['processing_times']:
            avg_time = sum(self.cache['processing_times']) / len(self.cache['processing_times']) * 1000
            cv2.putText(frame, f"Det: {avg_time:.0f}ms", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _add_caption_overlay(self, frame, caption):
        """Add scene caption overlay to frame"""
        caption_y = frame.shape[0] - 40
        cv2.rectangle(frame, (5, caption_y - 25), (frame.shape[1] - 5, caption_y + 10),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (5, caption_y - 25), (frame.shape[1] - 5, caption_y + 10),
                     (0, 255, 0), 2)
        cv2.putText(frame, f"Scene: {caption[:60]}", (10, caption_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    
    def handle_keyboard(self, key, frame):
        """Handle keyboard input"""
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            if self.audio:
                self.audio.speak_blocking("Scene description system stopped.")
            return False
        
        elif key == ord('d') or key == ord('D'):
            self.show_boxes = not self.show_boxes
            print(f"Bounding boxes: {'ON' if self.show_boxes else 'OFF'}")
        
        elif key == ord('a') or key == ord('A'):
            self._toggle_audio()
        
        elif key == ord('s') or key == ord('S'):
            if self.audio and self.cache['summary']:
                self.audio.speak_blocking(self._get_scene_description())
        
        elif key == ord('c') or key == ord('C'):
            self._generate_caption_on_demand(frame)
        
        elif key == ord('p') or key == ord('P'):
            self._print_detection_summary()
        
        elif key == ord('+') or key == ord('='):
            Config.PROCESS_EVERY_N_FRAMES = min(Config.PROCESS_EVERY_N_FRAMES + 1, 20)
            print(f"Frame skip: {Config.PROCESS_EVERY_N_FRAMES} (Faster)")
        
        elif key == ord('-') or key == ord('_'):
            Config.PROCESS_EVERY_N_FRAMES = max(Config.PROCESS_EVERY_N_FRAMES - 1, 1)
            print(f"Frame skip: {Config.PROCESS_EVERY_N_FRAMES} (More accurate)")
        
        elif key == ord('r') or key == ord('R'):
            Config.RESIZE_FRAME = not Config.RESIZE_FRAME
            print(f"Frame resize: {'ON' if Config.RESIZE_FRAME else 'OFF'}")
        
        return True
    
    def _toggle_audio(self):
        """Toggle audio feedback on/off"""
        if self.audio:
            if self.audio.is_enabled():
                self.audio.disable()
                print("Audio feedback: OFF")
            else:
                self.audio.enable()
                self.audio.speak("Audio feedback enabled")
                print("Audio feedback: ON")
    
    def _generate_caption_on_demand(self, frame):
        """Generate scene caption on demand"""
        if self.scene_recognizer and self.audio:
            print("Generating scene caption...")
            caption = self.scene_recognizer.recognize_scene(
                frame, max_length=Config.SCENE_MAX_LENGTH, num_beams=Config.SCENE_NUM_BEAMS
            )
            self.cache['scene_caption'] = caption
            self.cache['scene_context'] = self.scene_recognizer.identify_scene_context(caption)
            enhanced_caption = self.scene_recognizer.enhance_caption_for_accessibility(caption)
            self.audio.speak_blocking(enhanced_caption)
            print(f"Caption: {caption}")
    
    def _print_detection_summary(self):
        """Print detection summary to console"""
        print("\n" + "="*50)
        print("CURRENT DETECTION SUMMARY")
        print("="*50)
        
        if self.cache['summary']:
            for obj_name, obj_info in self.cache['summary'].items():
                print(f"{obj_name}: {obj_info['count']} "
                      f"(confidence: {obj_info['avg_confidence']:.2f})")
        else:
            print("No objects detected")
        
        if self.scene_recognizer and self.cache['scene_caption']:
            print("\nScene Caption:")
            print(f"  {self.cache['scene_caption']}")
            print(f"\nScene Context:")
            ctx = self.cache['scene_context']
            print(f"  Location: {ctx.get('location_type', 'unknown')}")
            print(f"  Room: {ctx.get('room_type', 'unknown')}")
            print(f"  Confidence: {ctx.get('confidence', 0.0):.2f}")
        
        print("="*50 + "\n")
    
    def run(self):
        """Main processing loop"""
        self.print_controls()
        
        try:
            while True:
                # Read frame
                ret, frame = self.video_handler.read_frame()
                if not ret:
                    print("Failed to grab frame. Attempting to reconnect...")
                    if not self.video_handler.reconnect():
                        print("Could not reconnect. Exiting.")
                        break
                    continue
                
                # Resize frame if needed
                if Config.RESIZE_FRAME and frame.shape[1] > Config.FRAME_WIDTH:
                    frame = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT),
                                     interpolation=cv2.INTER_LINEAR)
                
                self.frame_count += 1
                
                # Process frame
                self.process_frame(frame)
                
                # Render display frame
                display_frame = self.render_display_frame(frame)
                
                # Display
                cv2.imshow(Config.WINDOW_NAME, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_keyboard(key, frame):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        if self.video_handler:
            self.video_handler.release()
        cv2.destroyAllWindows()
        print("Scene description system stopped.")


def main():
    """Main function for scene description system"""
    system = SceneDescriptionSystem()
    if system.initialize_components():
        system.run()


if __name__ == "__main__":
    main()
