"""
Video Handler Module
Manages video capture from various sources (IP camera, webcam, video file)
"""

import cv2
import time


class VideoHandler:
    """
    Handles video capture and frame processing
    Supports IP cameras, webcams, and video files
    """
    
    def __init__(self, source, buffer_size=1):
        """
        Initialize video handler
        
        Args:
            source (str or int): Video source 
                                 - URL string for IP camera
                                 - Integer for webcam (0, 1, etc.)
                                 - File path for video file
            buffer_size (int): Number of frames to buffer
        """
        self.source = source
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        self.width = 0
        self.height = 0
        
        # Open video capture
        self.open()
        
    def open(self):
        """Open the video capture"""
        print(f"Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        
        if self.cap.isOpened():
            self.is_opened = True
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Video opened successfully!")
            print(f"Resolution: {self.width}x{self.height}")
            print(f"Source FPS: {self.source_fps}")
        else:
            self.is_opened = False
            print(f"Failed to open video source: {self.source}")
    
    def read_frame(self):
        """
        Read a frame from the video source
        
        Returns:
            tuple: (success, frame)
        """
        if not self.is_opened:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self._update_fps()
        
        return ret, frame
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        time_diff = current_time - self.fps_start_time
        
        if time_diff >= 1.0:  # Update every second
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def get_resolution(self):
        """Get video resolution"""
        return (self.width, self.height)
    
    def release(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("Video capture released")
    
    def is_open(self):
        """Check if video capture is open"""
        return self.is_opened and self.cap.isOpened()
    
    def reconnect(self):
        """Reconnect to video source"""
        print("Attempting to reconnect...")
        self.release()
        time.sleep(1)
        self.open()
        return self.is_opened
    
    @staticmethod
    def list_available_cameras():
        """
        List available camera indices
        
        Returns:
            list: List of available camera indices
        """
        available = []
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
