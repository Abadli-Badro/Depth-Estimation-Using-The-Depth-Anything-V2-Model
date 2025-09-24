import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
from PIL import Image

class DepthAnythingV2:
    """
    Depth Anything V2 model wrapper for real-time depth estimation - OPTIMIZED
    """
    
    def __init__(self, model_size: str = "small", device: str = "auto"):
        """
        Initialize Depth Anything V2 model
        
        Args:
            model_size: Model size ('small', 'base', 'large') - small is fastest
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        self.device = self._get_device(device)
        self.model_size = model_size
        self.model = None
        self.transform = None
        self._load_model()
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the Depth Anything V2 model"""
        try:
            # Try to load from transformers first (recommended)
            from transformers import pipeline
            model_name = f"depth-anything/Depth-Anything-V2-{self.model_size.capitalize()}-hf"
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            self.use_transformers = True
            print(f"Loaded Depth Anything V2 ({self.model_size}) using transformers")
            
        except ImportError:
            # Fallback to direct torch hub loading
            print("Transformers not available, using torch hub...")
            model_name = f"LiheYoung/depth-anything-{self.model_size}-v2"
            self.model = torch.hub.load("LiheYoung/Depth-Anything-V2", "depth_anything_v2_{0}".format(self.model_size), pretrained=True)
            self.model.to(self.device).eval()
            
            # Setup transforms
            self.transform = Compose([
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.use_transformers = False
            print(f"Loaded Depth Anything V2 ({self.model_size}) using torch hub")
            
        # OPTIMIZATION: Warm up model for better first frame performance
        print("Warming up model...")
        self._warmup_model()
        print(f"Model ready on {self.device}")
    
    def _warmup_model(self):
        """Warm up the model with dummy data"""
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _ = self.predict_depth(dummy_frame)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """OPTIMIZED: Preprocess frame for model input - faster version"""
        height, width = frame.shape[:2]
        
        # OPTIMIZATION: Use smaller target size for much better performance
        target_size = 256 if self.model_size == "small" else 384
        
        # OPTIMIZATION: Direct resize without aspect ratio preservation for speed
        frame_resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # OPTIMIZATION: Faster tensor conversion
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float().div_(255.0)  # In-place division
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        if not self.use_transformers:
            frame_tensor = self.transform(frame_tensor)
            
        # OPTIMIZATION: Non-blocking transfer to GPU
        return frame_tensor.to(self.device, non_blocking=True), (height, width)
    
    def predict_depth(self, frame: np.ndarray) -> np.ndarray:
        """OPTIMIZED: Predict depth map from frame"""
        if self.use_transformers:
            # OPTIMIZATION: Resize input for faster transformers processing
            h, w = frame.shape[:2]
            target_size = 256 if self.model_size == "small" else 384
            frame_small = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            result = self.pipe(pil_image)
            depth = np.array(result["depth"])
            
            # OPTIMIZATION: Fast resize back to original size
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            return depth
        else:
            # Using torch hub model - optimized
            frame_tensor, original_size = self.preprocess_frame(frame)
            
            with torch.no_grad():
                depth = self.model(frame_tensor)
                
            # OPTIMIZATION: Direct interpolation to original size
            depth = F.interpolate(
                depth.unsqueeze(1), 
                size=original_size, 
                mode="bilinear", 
                align_corners=False
            )
            depth = depth.squeeze().cpu().numpy()
            
            return depth

if __name__ == "__main__":
    image_path = "examples/nature.jpg"
    frame = cv2.imread(image_path)
    model = DepthAnythingV2(model_size="small")
    
    depth_map = model.predict_depth(frame)
        
    # Normalize depth for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    # resize for better visualization
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
    
    combined = np.hstack((frame, depth_colored))
    cv2.imwrite("examples/depth_output.jpg", combined)
    cv2.imshow("Depth Anything V2", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()