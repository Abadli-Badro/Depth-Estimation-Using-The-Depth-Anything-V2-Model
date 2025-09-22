import numpy as np
import time

class FPSCounter:
    """Simple FPS counter"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        
    def update(self) -> float:
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
        if len(self.frame_times) < 2:
            return 0.0
            
        # Calculate FPS
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
            
        return (len(self.frame_times) - 1) / time_diff