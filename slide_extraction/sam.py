import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Optional
import matplotlib.pyplot as plt

class ScreenDetector:
    """Detect screens/slides in frames using SAM2"""
    
    def __init__(self, model_name: str = "facebook/sam2.1-hiera-large", device: str = "cuda"):
        """
        Initialize the screen detector with SAM2 from cloud weights
        
        Args:
            model_name: Pretrained model name from HuggingFace
                Options: 
                - "facebook/sam2.1-hiera-tiny"
                - "facebook/sam2.1-hiera-small"
                - "facebook/sam2.1-hiera-base-plus"
                - "facebook/sam2.1-hiera-large"
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.model_name = model_name
        
        # Initialize SAM2 from pretrained weights
        print(f"Loading SAM2 model: {model_name}")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name, device=device)
        print("Model loaded successfully!")
        
    def detect_screen_auto(self, image: np.ndarray, num_points: int = 5) -> Optional[np.ndarray]:
        """
        Automatically detect screen/slide using grid-based point prompts
        
        Args:
            image: Input frame (BGR or RGB)
            num_points: Number of points to sample in grid
            
        Returns:
            Binary mask of detected screen or None if not found
        """
        h, w = image.shape[:2]
        
        # Use inference mode and autocast for optimal performance
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            # Set the image for prediction
            self.predictor.set_image(image)
            
            # Create a grid of points biased toward center (screens usually central)
            grid_points = []
            labels = []
            
            # Center point (most likely to be on screen)
            grid_points.append([w//2, h//2])
            labels.append(1)  # Positive point
            
            # Add points in a cross pattern
            for offset in [0.2, 0.3, 0.4]:
                grid_points.extend([
                    [int(w * (0.5 + offset)), h//2],
                    [int(w * (0.5 - offset)), h//2],
                    [w//2, int(h * (0.5 + offset))],
                    [w//2, int(h * (0.5 - offset))]
                ])
                labels.extend([1, 1, 1, 1])
            
            # Add negative points at corners (usually not screen)
            corner_offset = 50
            grid_points.extend([
                [corner_offset, corner_offset],
                [w - corner_offset, corner_offset],
                [corner_offset, h - corner_offset],
                [w - corner_offset, h - corner_offset]
            ])
            labels.extend([0, 0, 0, 0])  # Negative points
            
            # Convert to numpy arrays
            point_coords = np.array(grid_points[:num_points + 4])
            point_labels = np.array(labels[:num_points + 4])
            
            # Predict masks
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
        
        # Select best mask based on score and rectangular properties
        best_mask = self._select_best_screen_mask(masks, scores)
        
        return best_mask
    
    def detect_screen_with_prompt(self, image: np.ndarray, 
                                 point_coords: Optional[np.ndarray] = None,
                                 box: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Detect screen/slide using manual prompts
        
        Args:
            image: Input frame
            point_coords: Points on the screen (Nx2 array)
            box: Bounding box around screen [x1, y1, x2, y2]
            
        Returns:
            Binary mask of detected screen
        """
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(image)
            
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=np.ones(len(point_coords)) if point_coords is not None else None,
                box=box,
                multimask_output=True
            )
        
        # Select best mask
        best_mask = self._select_best_screen_mask(masks, scores)
        
        return best_mask
    
    def _select_best_screen_mask(self, masks: np.ndarray, scores: np.ndarray) -> Optional[np.ndarray]:
        """
        Select the mask most likely to be a screen/slide
        
        Args:
            masks: Array of predicted masks
            scores: Confidence scores for each mask
            
        Returns:
            Best mask or None
        """
        best_score = -1
        best_mask = None
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # Calculate rectangularity score
            rect_score = self._calculate_rectangularity(mask)
            
            # Combine SAM score with rectangularity
            combined_score = score * rect_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_mask = mask
        
        # Filter out if score too low
        if best_score < 0.5:
            return None
            
        return best_mask
    
    def _calculate_rectangularity(self, mask: np.ndarray) -> float:
        """
        Calculate how rectangular a mask is (screens are usually rectangular)
        
        Args:
            mask: Binary mask
            
        Returns:
            Rectangularity score (0-1)
        """
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box_area = rect[1][0] * rect[1][1]
        
        # Calculate contour area
        contour_area = cv2.contourArea(largest_contour)
        
        # Rectangularity is ratio of contour area to bounding box area
        if box_area > 0:
            rectangularity = contour_area / box_area
        else:
            rectangularity = 0
        
        # Check aspect ratio (screens typically have standard ratios)
        aspect_ratio = max(rect[1]) / (min(rect[1]) + 1e-6)
        
        # Common screen aspect ratios: 16:9, 16:10, 4:3
        common_ratios = [16/9, 16/10, 4/3, 9/16, 10/16, 3/4]
        ratio_score = max([1 - abs(aspect_ratio - ratio) / ratio for ratio in common_ratios])
        
        # Combine rectangularity and aspect ratio scores
        return rectangularity * ratio_score
    
    def extract_screen_region(self, image: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the screen region from the image using the mask
        
        Args:
            image: Original image
            mask: Binary mask of the screen
            
        Returns:
            Cropped and perspective-corrected screen region
        """
        if mask is None:
            return None
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we got 4 corners, do perspective transform
        if len(approx) == 4:
            # Get corners
            corners = approx.reshape(4, 2).astype(np.float32)
            
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(corners)
            
            # Calculate output dimensions
            width = max(
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[3])
            )
            height = max(
                np.linalg.norm(corners[3] - corners[0]),
                np.linalg.norm(corners[2] - corners[1])
            )
            
            # Define destination points
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Get perspective transform
            M = cv2.getPerspectiveTransform(corners, dst)
            
            # Warp image
            warped = cv2.warpPerspective(image, M, (int(width), int(height)))
            
            return warped
        else:
            # Just crop to bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y+h, x:x+w]
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left
        
        Args:
            pts: 4x2 array of corner points
            
        Returns:
            Ordered corners
        """
        # Sort by y-coordinate
        sorted_y = pts[np.argsort(pts[:, 1])]
        
        # Top two points
        top = sorted_y[:2]
        # Bottom two points
        bottom = sorted_y[2:]
        
        # Sort top points by x (left to right)
        top = top[np.argsort(top[:, 0])]
        # Sort bottom points by x (left to right)
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def visualize_detection(self, image: np.ndarray, mask: np.ndarray):
        """
        Visualize the detected screen
        
        Args:
            image: Original image
            mask: Detected screen mask
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Detected Screen Mask")
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[2].imshow(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
        axes[2].set_title("Detection Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
def main():
    # Initialize detector with cloud-based model weights
    # Available models:
    # - "facebook/sam2.1-hiera-tiny" (fastest, least accurate)
    # - "facebook/sam2.1-hiera-small"
    # - "facebook/sam2.1-hiera-base-plus"
    # - "facebook/sam2.1-hiera-large" (slowest, most accurate)
    
    detector = ScreenDetector(
        model_name="facebook/sam2.1-hiera-large",  # Choose model variant
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load image
    image_path = "Fintech/clip2_slides/slide_001.jpg"  # Your image path
    image = cv2.imread(image_path)
    
    # Method 1: Automatic detection
    print("Running automatic screen detection...")
    mask = detector.detect_screen_auto(image)
    
    if mask is not None:
        print("Screen detected!")
        
        # Visualize
        detector.visualize_detection(image, mask)
        
        # Extract screen region
        screen_region = detector.extract_screen_region(image, mask)
        if screen_region is not None:
            cv2.imwrite("extracted_screen.jpg", screen_region)
            print("Extracted screen saved to 'extracted_screen.jpg'")
    else:
        print("No screen detected")
    
    # Method 2: With manual bounding box prompt
    # Define approximate bounding box around screen
    # box = np.array([100, 50, 800, 600])  # [x1, y1, x2, y2]
    # mask = detector.detect_screen_with_prompt(image, box=box)
    
    # Method 3: With point prompts
    # Click on a few points on the screen
    # points = np.array([[400, 300], [500, 350], [350, 250]])  # Example points
    # mask = detector.detect_screen_with_prompt(image, point_coords=points)


if __name__ == "__main__":
    main()