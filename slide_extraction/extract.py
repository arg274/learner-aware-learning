import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
import json
from typing import List, Dict
from .sam import ScreenDetector
import torch

class SlideExtractor:
    def __init__(self, video_path: str, pdf_path: str, 
                 sample_interval: float = 1.0):
        """
        Initialize the slide extractor.
        
        Args:
            video_path: Path to the video file
            pdf_path: Path to the PDF with reference slides
            sample_interval: Time interval (seconds) between frame samples
        """
        self.video_path = video_path
        self.pdf_path = pdf_path
        self.sample_interval = sample_interval
        self.reference_slides = []
        
    def load_pdf_slides(self) -> List[np.ndarray]:
        """Load PDF pages as images."""
        print(f"Loading PDF slides from {self.pdf_path}...")
        
        pages = convert_from_path(self.pdf_path, dpi=150)
        
        slides = []
        for i, page in enumerate(pages):
            slide = np.array(page)
            slide = cv2.cvtColor(slide, cv2.COLOR_RGB2BGR)
            # Resize to standard size for comparison
            slide = cv2.resize(slide, (1280, 720), interpolation=cv2.INTER_AREA)
            slides.append(slide)
            print(f"  Loaded slide {i + 1}/{len(pages)}")
        
        self.reference_slides = slides
        return slides
    
    def compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute color histogram for an image.
        More robust to annotations/drawings than pixel-by-pixel comparison.
        """
        # Convert to HSV for better color matching
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Compute histogram
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                           [0, 180, 0, 256, 0, 256])
        
        # Normalize histogram
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def compute_edge_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity based on edge detection.
        Helps with slides that have drawings on them.
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # Compare edge maps
        intersection = np.logical_and(edges1, edges2).sum()
        union = np.logical_or(edges1, edges2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_closest_slide(self, frame: np.ndarray) -> int:
        """
        Find the closest matching slide for a frame.
        Uses both histogram and edge similarity.
        """
        if not self.reference_slides:
            raise ValueError("Reference slides not loaded")
        
        # Resize frame to standard size
        frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        
        # Compute frame histogram
        frame_hist = self.compute_histogram(frame_resized)
        
        best_idx = 0
        best_score = -1
        
        for idx, ref_slide in enumerate(self.reference_slides):
            # Compute histogram similarity
            ref_hist = self.compute_histogram(ref_slide)
            hist_similarity = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)
            
            # Compute edge similarity
            edge_similarity = self.compute_edge_similarity(frame_resized, ref_slide)
            
            # Combined score (weighted average)
            combined_score = 0.45 * hist_similarity + 0.55 * edge_similarity
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        
        return best_idx
    
    def apply_sequential_constraint(self, slide_sequence: List[int]) -> List[int]:
        """
        Apply sequential constraint: fill in missing slides.
        If we jump from slide 2 to slide 5, we must have gone through 3 and 4.
        """
        if not slide_sequence:
            return []
        
        result = [slide_sequence[0]]
        
        for i in range(1, len(slide_sequence)):
            prev_slide = result[-1]
            curr_slide = slide_sequence[i]
            
            if curr_slide == prev_slide:
                result.append(curr_slide)
            elif curr_slide > prev_slide:
                # Moving forward: add all intermediate slides
                for s in range(prev_slide + 1, curr_slide + 1):
                    result.append(s)
            else:
                # Moving backward: add all intermediate slides in reverse
                for s in range(prev_slide - 1, curr_slide - 1, -1):
                    result.append(s)
        
        return result
    
    def condense_to_ranges(self, timestamps: List[float], 
                          slides: List[int]) -> List[Dict]:
        """
        Condense consecutive identical slides into time ranges.
        """
        if not timestamps or not slides:
            return []
        
        ranges = []
        current_slide = slides[0]
        start_time = timestamps[0]
        
        for i in range(1, len(slides)):
            if slides[i] != current_slide:
                # End of current range
                ranges.append({
                    'slide_number': current_slide + 1,  # 1-indexed
                    'start_time': start_time,
                    'end_time': timestamps[i],
                    'duration': timestamps[i] - start_time
                })
                
                # Start new range
                current_slide = slides[i]
                start_time = timestamps[i]
        
        # Add final range
        ranges.append({
            'slide_number': current_slide + 1,
            'start_time': start_time,
            'end_time': timestamps[-1],
            'duration': timestamps[-1] - start_time
        })
        
        return ranges
    
    def extract_slides(self, output_dir: str = "extracted_slides") -> List[Dict]:
        """
        Extract slides from video and map them to timestamps.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load reference slides
        if not self.reference_slides:
            self.load_pdf_slides()
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print("\nVideo info:")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sampling every {self.sample_interval} seconds\n")
        
        # Sample frames and match to slides
        timestamps = []
        matched_slides = []
        sample_every_n_frames = int(fps * self.sample_interval)
        
        print("Matching frames to slides...")
        frame_count = 0

        detector = ScreenDetector(
            model_name="facebook/sam2.1-hiera-large",  # Choose model variant
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_every_n_frames == 0:
                timestamp = frame_count / fps

                # Use SAM to extract slide from frame
                mask = detector.detect_screen_auto(frame)
                screen_region = None
                if mask is not None:
                    screen_region = detector.extract_screen_region(frame, mask)
                if screen_region is None:
                    continue
                
                slide_idx = self.find_closest_slide(screen_region)
                
                timestamps.append(timestamp)
                matched_slides.append(slide_idx)
                
                if frame_count % (sample_every_n_frames * 10) == 0:
                    print(f"  Processed {timestamp:.1f}s / {duration:.1f}s "
                          f"(matched to slide {slide_idx + 1})")
            
            frame_count += 1
        
        cap.release()
        
        # Add final timestamp
        timestamps.append(duration)
        matched_slides.append(matched_slides[-1] if matched_slides else 0)
        
        print(f"\nMatched {len(timestamps)} frames")
        print("Applying sequential constraint...")
        
        # Apply sequential constraint
        constrained_slides = self.apply_sequential_constraint(matched_slides)
        
        print("Condensing to time ranges...")
        
        # Condense to ranges
        results = self.condense_to_ranges(timestamps, constrained_slides)
        
        # Save results
        results_path = output_path / "slide_timestamps.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Extract representative frames
        print("\nExtracting representative frames...")
        cap = cv2.VideoCapture(self.video_path)
        
        for slide_info in results:
            # Extract frame at midpoint of range
            mid_time = (slide_info['start_time'] + slide_info['end_time']) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()
            
            if ret:
                # Use SAM to extract slide from frame
                mask = detector.detect_screen_auto(frame)
                screen_region = None
                if mask is not None:
                    screen_region = detector.extract_screen_region(frame, mask)
                if screen_region is None:
                    screen_region = frame

                frame_path = output_path / f"slide_{slide_info['slide_number']:03d}.jpg"
                cv2.imwrite(str(frame_path), screen_region)
        
        cap.release()
        
        print(f"\nResults saved to {results_path}")
        print(f"Extracted {len(results)} slide segments\n")
        
        # Print summary
        print("="*60)
        print("SLIDE TIMELINE")
        print("="*60)
        for slide_info in results:
            print(f"Slide {slide_info['slide_number']:2d}: "
                  f"{slide_info['start_time']:7.2f}s - {slide_info['end_time']:7.2f}s "
                  f"(duration: {slide_info['duration']:6.2f}s)")
        
        return results


def main():
    """Example usage."""
    # Configuration
    video_path = "Fintech/Lecture1_clip-2.mp4"  # Path to your video file
    pdf_path = "Fintech/clip2_slides.pdf"          # Path to your PDF with slides
    output_dir = "Fintech/clip2_slides"  # Output directory
    
    extractor = SlideExtractor(
        video_path=video_path,
        pdf_path=pdf_path,
        sample_interval=1.0  # Sample every 1 second
    )
    
    extractor.extract_slides(output_dir=output_dir)



if __name__ == "__main__":
    main()