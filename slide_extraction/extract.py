import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path
import json
from typing import List, Dict
from skimage.metrics import structural_similarity as ssim


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

    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute SSIM (Structural Similarity Index) between two frames.
        Used to detect if consecutive frames are nearly identical.
        """
        # Convert to grayscale for faster comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Ensure same size
        if gray1.shape != gray2.shape:
            gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
        
        # Compute SSIM
        score = ssim(gray1, gray2)
        return score

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

    def find_closest_slide(self, frame: np.ndarray, confidence_threshold: float = 0.3) -> tuple[int | None, float]:
        """
        Find the closest matching slide for a frame.
        Returns (slide_idx, confidence) or (None, confidence) if below threshold.
        """
        if not self.reference_slides:
            raise ValueError("Reference slides not loaded")

        frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        frame_hist = self.compute_histogram(frame_resized)

        best_idx = 0
        best_score = -1

        for idx, ref_slide in enumerate(self.reference_slides):
            ref_hist = self.compute_histogram(ref_slide)
            hist_similarity = cv2.compareHist(frame_hist, ref_hist, cv2.HISTCMP_CORREL)
            edge_similarity = self.compute_edge_similarity(frame_resized, ref_slide)
            combined_score = 0.45 * hist_similarity + 0.55 * edge_similarity

            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        # Return None if confidence is too low
        if best_score < confidence_threshold:
            return None, best_score
        
        return best_idx, best_score


    def apply_smoothing(self, slide_sequence: List[int | None], 
                        timestamps: List[float], 
                        window_size: int = 5) -> tuple[List[int | None], List[float]]:
        """
        Apply median filtering to smooth out noisy detections.
        Much more reasonable than sequential constraint.
        """
        if len(slide_sequence) < window_size:
            return slide_sequence, timestamps
        
        smoothed = []
        for i in range(len(slide_sequence)):
            start = max(0, i - window_size // 2)
            end = min(len(slide_sequence), i + window_size // 2 + 1)
            window = [s for s in slide_sequence[start:end] if s is not None]
            
            if window:
                # Use mode (most common) instead of median for slide numbers
                smoothed.append(max(set(window), key=window.count))
            else:
                smoothed.append(None)
        
        return smoothed, timestamps


    def condense_to_ranges(self, timestamps: List[float],
                        slides: List[int | None]) -> List[Dict]:
        """
        Condense consecutive identical slides into time ranges.
        Handles None values for low-confidence matches.
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
                    'slide_number': current_slide + 1 if current_slide is not None else None,
                    'start_time': start_time,
                    'end_time': timestamps[i],
                    'duration': timestamps[i] - start_time
                })

                current_slide = slides[i]
                start_time = timestamps[i]

        # Add final range
        ranges.append({
            'slide_number': current_slide + 1 if current_slide is not None else None,
            'start_time': start_time,
            'end_time': timestamps[-1],
            'duration': timestamps[-1] - start_time
        })

        return ranges

    def extract_slides(self, output_dir: str = "extracted_slides", 
                  confidence_threshold: float = 0.3,
                  smoothing_window: int = 5,
                  ssim_threshold: float = 0.98) -> List[Dict]:
        """
        Extract slides from video and map them to timestamps.
        
        Args:
            output_dir: Directory to save results
            confidence_threshold: Minimum confidence score to accept a match (0-1)
            smoothing_window: Window size for median filtering noise
            ssim_threshold: SSIM threshold to consider frames identical (0-1)
        
        Returns:
            List of slide timing information
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

        print("\n" + "="*60)
        print("VIDEO INFORMATION")
        print("="*60)
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sampling interval: {self.sample_interval}s")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  SSIM threshold: {ssim_threshold}")
        print(f"  Smoothing window: {smoothing_window}")
        print("="*60 + "\n")

        # Sample frames and match to slides
        timestamps = []
        matched_slides = []
        confidences = []
        sample_every_n_frames = int(fps * self.sample_interval)

        # Track previous frame and slide for SSIM comparison
        prev_frame = None
        prev_slide = None
        ssim_skips = 0  # Counter for SSIM-based skips

        print("Matching frames to slides...")
        frame_count = 0
        samples_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_every_n_frames == 0:
                timestamp = frame_count / fps
                
                # Check SSIM with previous frame first
                if prev_frame is not None:
                    ssim_score = self.compute_ssim(prev_frame, frame)
                    
                    if ssim_score >= ssim_threshold:
                        # Frame is nearly identical to previous, reuse previous slide
                        slide_idx = prev_slide
                        confidence = 1.0  # High confidence since it's same as previous
                        ssim_skips += 1
                        
                        timestamps.append(timestamp)
                        matched_slides.append(slide_idx)
                        confidences.append(confidence)
                        samples_processed += 1
                        
                        # Progress update every 10 samples
                        if samples_processed % 10 == 0:
                            slide_str = f"slide {slide_idx + 1}" if slide_idx is not None else "None (low confidence)"
                            print(f"  [{timestamp:7.1f}s / {duration:7.1f}s] → {slide_str:20s} (SSIM: {ssim_score:.3f}, skipped)")
                        
                        # Update previous frame but keep processing
                        prev_frame = frame.copy()
                        frame_count += 1
                        continue
                
                # SSIM check failed or no previous frame, do full matching
                slide_idx, confidence = self.find_closest_slide(frame, confidence_threshold)

                timestamps.append(timestamp)
                matched_slides.append(slide_idx)
                confidences.append(confidence)
                samples_processed += 1

                # Progress update every 10 samples
                if samples_processed % 10 == 0:
                    slide_str = f"slide {slide_idx + 1}" if slide_idx is not None else "None (low confidence)"
                    print(f"  [{timestamp:7.1f}s / {duration:7.1f}s] → {slide_str:20s} (conf: {confidence:.3f})")

                # Update previous frame and slide
                prev_frame = frame.copy()
                prev_slide = slide_idx

            frame_count += 1

        cap.release()

        # Add final timestamp
        if timestamps:
            timestamps.append(duration)
            matched_slides.append(matched_slides[-1])
            confidences.append(confidences[-1])

        print(f"\n✓ Matched {len(timestamps)} frames")
        print(f"✓ SSIM optimization skipped {ssim_skips} expensive comparisons ({100*ssim_skips/max(1, len(timestamps)):.1f}%)\n")

        # Statistics before smoothing
        none_count = sum(1 for s in matched_slides if s is None)
        valid_count = len(matched_slides) - none_count
        print(f"Raw matching statistics:")
        print(f"  Valid matches: {valid_count}/{len(matched_slides)} ({100*valid_count/len(matched_slides):.1f}%)")
        print(f"  Low confidence: {none_count}/{len(matched_slides)} ({100*none_count/len(matched_slides):.1f}%)")
        if confidences:
            valid_confidences = [c for c, s in zip(confidences, matched_slides) if s is not None]
            if valid_confidences:
                print(f"  Avg confidence: {np.mean(valid_confidences):.3f}")
        print()

        # Apply smoothing
        print("Applying median smoothing to reduce noise...")
        smoothed_slides, smoothed_timestamps = self.apply_smoothing(
            matched_slides, timestamps, window_size=smoothing_window
        )
        
        # Statistics after smoothing
        none_count_smooth = sum(1 for s in smoothed_slides if s is None)
        valid_count_smooth = len(smoothed_slides) - none_count_smooth
        print(f"After smoothing:")
        print(f"  Valid matches: {valid_count_smooth}/{len(smoothed_slides)} ({100*valid_count_smooth/len(smoothed_slides):.1f}%)")
        print()

        # Condense to ranges
        print("Condensing to time ranges...")
        results = self.condense_to_ranges(smoothed_timestamps, smoothed_slides)

        # Save detailed results with confidence scores
        results_path = output_path / "slide_timestamps.json"
        detailed_results = {
            'metadata': {
                'video_path': self.video_path,
                'pdf_path': self.pdf_path,
                'duration': duration,
                'sample_interval': self.sample_interval,
                'confidence_threshold': confidence_threshold,
                'ssim_threshold': ssim_threshold,
                'smoothing_window': smoothing_window,
                'total_slides_in_pdf': len(self.reference_slides),
                'ssim_skips': ssim_skips,
                'ssim_skip_percentage': round(100 * ssim_skips / max(1, len(timestamps)), 2)
            },
            'segments': results
        }
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Save raw matching data for debugging
        debug_path = output_path / "debug_matching.json"
        debug_data = []
        for i, (ts, slide, conf) in enumerate(zip(timestamps, matched_slides, confidences)):
            debug_data.append({
                'timestamp': round(ts, 2),
                'slide': slide + 1 if slide is not None else None,
                'confidence': round(conf, 4)
            })
        
        with open(debug_path, 'w') as f:
            json.dump(debug_data, f, indent=2)

        # Extract representative frames
        print("Extracting representative frames for each segment...")
        cap = cv2.VideoCapture(self.video_path)
        extracted_count = 0

        for slide_info in results:
            if slide_info['slide_number'] is None:
                continue  # Skip segments with no confident match
                
            # Extract frame at midpoint of range
            mid_time = (slide_info['start_time'] + slide_info['end_time']) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()

            if ret:
                frame_path = output_path / f"slide_{slide_info['slide_number']:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1

        cap.release()

        print(f"✓ Extracted {extracted_count} frames\n")
        print(f"Results saved to {results_path}")
        print(f"Debug data saved to {debug_path}\n")

        # Print summary
        print("="*60)
        print("SLIDE TIMELINE SUMMARY")
        print("="*60)
        
        for slide_info in results:
            slide_num = slide_info['slide_number']
            slide_str = f"Slide {slide_num:2d}" if slide_num else "No Match"
            print(f"{slide_str}: "
                f"{slide_info['start_time']:7.2f}s - {slide_info['end_time']:7.2f}s "
                f"(duration: {slide_info['duration']:6.2f}s)")
        
        print("="*60)
        
        # Summary statistics
        total_segments = len(results)
        matched_segments = sum(1 for r in results if r['slide_number'] is not None)
        unmatched_segments = total_segments - matched_segments
        
        print(f"\nTotal segments: {total_segments}")
        print(f"  Matched: {matched_segments}")
        print(f"  Unmatched: {unmatched_segments}")
        
        unique_slides = set(r['slide_number'] for r in results if r['slide_number'] is not None)
        print(f"  Unique slides shown: {len(unique_slides)} / {len(self.reference_slides)}")

        return results


def main():
    """Example usage."""
    # Configuration
    video_path = "output/LaP0Ut84GzI.mp4"  # Path to your video file
    pdf_path = "data/MIT15-S08S20_class4.pdf"          # Path to your PDF with slides
    output_dir = "output/slide_4_ssim"  # Output directory

    extractor = SlideExtractor(
        video_path=video_path,
        pdf_path=pdf_path,
        sample_interval=1.0  # Sample every 1 second
    )

    extractor.extract_slides(
        output_dir=output_dir,
        ssim_threshold=0.99  # Adjust this threshold as needed
    )


if __name__ == "__main__":
    main()
