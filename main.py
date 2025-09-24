import os
from depth_anythingV2 import DepthAnythingV2
from utils import FPSCounter
import cv2
import numpy as np
import time
from typing import Tuple
import argparse


class CameraDepthEstimator:
    """
    OPTIMIZED: Real-time depth estimation from camera feed
    """
    
    def __init__(self, camera_id: int = 0, model_size: str = "small"):
        """
        Initialize camera and depth model
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            model_size: Size of depth model ('small', 'base', 'large')
        """
        self.camera_id = camera_id
        self.cap = None
        self.depth_model = DepthAnythingV2(model_size=model_size)
        self.fps_counter = FPSCounter()
        
        # OPTIMIZATION: Pre-allocate arrays for better performance
        self.depth_colored_cache = None
        
    def initialize_camera(self) -> bool:
        """OPTIMIZED: Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # OPTIMIZATION: Camera settings for best performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # OPTIMIZATION: Try to set camera to fastest format
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        print(f"Camera initialized: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """OPTIMIZED: Process single frame"""
        # Predict depth
        depth_map = self.depth_model.predict_depth(frame)
        
        # OPTIMIZATION: Faster depth visualization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
            
        # OPTIMIZATION: Use PLASMA colormap (faster than INFERNO)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_PLASMA)
        
        return frame, depth_colored
    
    def run(self, save_output: bool = False, output_path: str = "depth_output.avi"):
        """OPTIMIZED: Run real-time depth estimation"""
        if not self.initialize_camera():
            return
            
        print("Starting depth estimation... Press 'q' to quit, 's' to save screenshot")
        print("OPTIMIZATION TIPS:")
        print("- Ensure good lighting for better depth estimation")
        
        # Video writer for saving (optional)
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Better compression
            fps = 15  # Reasonable fps for saving
            frame_size = (980, 480)  # Side by side
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        # OPTIMIZATION: Skip frame processing to maintain display smoothness
        frame_skip = 2 if self.depth_model.model_size != "small" else 1
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # OPTIMIZATION: Process every nth frame for better performance
                if frame_count % frame_skip == 0:
                    start_time = time.time()
                    original_frame, depth_frame = self.process_frame(frame)
                    processing_time = time.time() - start_time
                else:
                    original_frame = frame
                    # Use cached depth frame
                    if hasattr(self, 'last_depth_frame'):
                        depth_frame = self.last_depth_frame
                    else:
                        # First frame
                        start_time = time.time()
                        original_frame, depth_frame = self.process_frame(frame)
                        processing_time = time.time() - start_time
                    
                # Cache the depth frame
                if frame_count % frame_skip == 0:
                    self.last_depth_frame = depth_frame.copy()
                
                # Update FPS
                fps = self.fps_counter.update()
                
                # OPTIMIZATION: Efficient text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(original_frame, f"FPS: {fps:.1f}", (10, 30), font, 0.7, (0, 255, 0), 2)
                if frame_count % frame_skip == 0:
                    cv2.putText(original_frame, f"Process: {processing_time*1000:.0f}ms", (10, 60), font, 0.7, (0, 255, 0), 2)
                cv2.putText(original_frame, f"Model: {self.depth_model.model_size}", (10, 90), font, 0.7, (0, 255, 0), 2)
                
                # Create side-by-side display
                display_frame = np.hstack([original_frame, depth_frame])
                
                # Save frame if recording
                if writer is not None and frame_count % 2 == 0:  # Save every 2nd frame
                    writer.write(display_frame)
                
                # Display
                cv2.imshow('Depth Anything V2 - Original | Depth', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    cv2.imwrite(f'depth_screenshot_{timestamp}.jpg', display_frame)
                    print(f"Screenshot saved: depth_screenshot_{timestamp}.jpg")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description="Real-time depth estimation using Depth Anything V2")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="small", 
                       help="Model size (default: small for fastest performance)")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--output", type=str, default="depth_output.avi", help="Output video path")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Depth Anything V2 Real-time Demo")
    print("="*50)
    print(f"Camera ID: {args.camera}")
    print(f"Model size: {args.model_size}")
    print(f"Save output: {args.save}")
    print("="*50)
    
    # Create and run estimator
    estimator = CameraDepthEstimator(
        camera_id=args.camera,
        model_size=args.model_size
    )
    
    estimator.run(save_output=args.save, output_path=args.output)


if __name__ == "__main__":
    main()