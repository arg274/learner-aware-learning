import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional
from pdf2image import convert_from_path
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from sentence_transformers import SentenceTransformer, util
import torch
import re


class HybridSlideExtractor:
    def __init__(self, 
                 video_path: str, 
                 pdf_path: str,
                 fps: float = 1.0,
                 phash_high_threshold: float = 0.7,
                 phash_gap_threshold: float = 0.05,
                 ocr_batch_size: int = 16,
                 ocr_similarity_threshold: float = 0.7):
        """
        Initialize hybrid slide extractor.
        
        Args:
            video_path: Path to video file
            pdf_path: Path to PDF with reference slides
            fps: Frames per second to sample (default: 1.0)
            phash_high_threshold: High confidence threshold for pHash (default: 0.7)
            phash_gap_threshold: Minimum gap between top 2 matches (default: 0.05)
            ocr_batch_size: Batch size for DeepSeek OCR (default: 16)
            ocr_similarity_threshold: Threshold for sentence similarity (default: 0.7)
        """
        self.video_path = video_path
        self.pdf_path = pdf_path
        self.fps = fps
        self.phash_high_threshold = phash_high_threshold
        self.phash_gap_threshold = phash_gap_threshold
        self.ocr_batch_size = ocr_batch_size
        self.ocr_similarity_threshold = ocr_similarity_threshold
        
        self.reference_slides = []
        self.reference_phashes = []
        self.reference_ocr_texts = []
        
        # Will be initialized lazily
        self.llm = None
        self.sentence_model = None
        self.reference_embeddings = None
    
    def load_pdf_slides(self):
        """Load PDF slides and compute pHashes."""
        print(f"Loading PDF slides from {Path(self.pdf_path).name}...")
        
        pages = convert_from_path(self.pdf_path, dpi=150)
        if not pages:
            raise ValueError("No pages found in PDF")
        
        for page in tqdm(pages, desc="  Processing PDF pages"):
            slide = np.array(page)
            slide = cv2.cvtColor(slide, cv2.COLOR_RGB2BGR)
            
            # Store slide
            self.reference_slides.append(slide)
            
            # Compute pHash
            pil_image = Image.fromarray(cv2.cvtColor(slide, cv2.COLOR_BGR2RGB))
            phash = imagehash.phash(pil_image, hash_size=16)
            self.reference_phashes.append(phash)
        
        print(f"  ✓ Loaded {len(self.reference_slides)} slides with pHashes")
    
    def initialize_ocr_models(self):
        """Initialize DeepSeek OCR and sentence transformer models."""
        if self.llm is None:
            print("\nInitializing DeepSeek OCR model...")
            self.llm = LLM(
                model="deepseek-ai/DeepSeek-OCR",
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor]
            )
            print("  ✓ DeepSeek OCR initialized")
        
        if self.sentence_model is None:
            print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  ✓ Sentence transformer loaded")
    
    def extract_ocr_from_slides(self):
        """Extract OCR text from all reference slides using DeepSeek."""
        print("\nExtracting OCR from reference slides...")
        self.initialize_ocr_models()
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )
        
        # Prepare batches
        batches = []
        for i in range(0, len(self.reference_slides), self.ocr_batch_size):
            batch = self.reference_slides[i:i + self.ocr_batch_size]
            batches.append(batch)
        
        self.reference_ocr_texts = []
        
        for batch in tqdm(batches, desc="  Processing reference slides"):
            # Prepare batch input
            batch_inputs = []
            for slide in batch:
                pil_image = Image.fromarray(cv2.cvtColor(slide, cv2.COLOR_BGR2RGB))
                batch_inputs.append({
                    "prompt": "<image>\nFree OCR.",
                    "multi_modal_data": {"image": pil_image}
                })
            
            # Run batch inference
            outputs = self.llm.generate(batch_inputs, sampling_params)
            
            # Extract texts
            for output in outputs:
                text = output.outputs[0].text
                self.reference_ocr_texts.append(text)
        
        # Compute embeddings for reference texts
        print("  Computing embeddings for reference slides...")
        self.reference_embeddings = self.sentence_model.encode(
            self.reference_ocr_texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        print(f"  ✓ Extracted OCR from {len(self.reference_ocr_texts)} reference slides")
    
    def compute_phash(self, image: np.ndarray) -> imagehash.ImageHash:
        """Compute perceptual hash for an image."""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return imagehash.phash(pil_image, hash_size=16)
    
    def find_closest_slide_phash(self, frame: np.ndarray) -> Tuple[Optional[int], float, bool]:
        """
        Find closest slide using pHash.
        
        Returns:
            (slide_idx, confidence, is_confident_match)
            - slide_idx: Index of best matching slide (or None)
            - confidence: Similarity score of best match
            - is_confident_match: True if match passes confidence criteria
        """
        if not self.reference_phashes:
            raise ValueError("Reference slides not loaded")
        
        frame_phash = self.compute_phash(frame)
        max_hash_distance = 16 * 16
        
        # Calculate similarities for all slides
        similarities = []
        for idx, ref_phash in enumerate(self.reference_phashes):
            distance = frame_phash - ref_phash
            similarity = 1.0 - (distance / max_hash_distance)
            similarities.append((idx, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        best_idx, best_score = similarities[0]
        second_score = similarities[1][1] if len(similarities) > 1 else 0.0
        
        # Check confidence criteria
        is_confident = False
        if best_score >= self.phash_high_threshold:
            # High confidence match
            is_confident = True
        elif best_score - second_score >= self.phash_gap_threshold:
            # Significant gap between top 2
            is_confident = True
        
        if is_confident:
            return best_idx, best_score, True
        else:
            return None, best_score, False
    
    def process_ocr_batch(self, frames: List[np.ndarray]) -> List[str]:
        """Process a batch of frames through DeepSeek OCR."""
        if not frames:
            return []
        
        self.initialize_ocr_models()
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )
        
        # Prepare batch input
        batch_inputs = []
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch_inputs.append({
                "prompt": "<image>\nFree OCR.",
                "multi_modal_data": {"image": pil_image}
            })
        
        # Run batch inference
        outputs = self.llm.generate(batch_inputs, sampling_params)
        
        # Extract texts
        texts = [output.outputs[0].text for output in outputs]
        return texts
    
    def find_closest_slide_ocr(self, frame_text: str) -> Tuple[Optional[int], float]:
        """
        Find closest slide using OCR text similarity.
        
        Returns:
            (slide_idx, similarity_score)
        """
        if self.reference_embeddings is None:
            raise ValueError("Reference OCR embeddings not computed")
        
        # Encode frame text
        frame_embedding = self.sentence_model.encode(
            frame_text,
            convert_to_tensor=True
        )
        
        # Compute cosine similarities
        similarities = util.cos_sim(frame_embedding, self.reference_embeddings)[0]
        
        # Find best match
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()
        
        if best_score >= self.ocr_similarity_threshold:
            return best_idx, best_score
        else:
            return None, best_score
    
    def extract_slides(self, output_dir: str = "extracted_slides"):
        """
        Extract and match slides from video using hybrid approach.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load PDF slides
        if not self.reference_slides:
            self.load_pdf_slides()
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps
        
        print("\n" + "="*70)
        print("VIDEO INFORMATION")
        print("="*70)
        print(f"  Video FPS: {video_fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Sampling rate: {self.fps} fps")
        print(f"  pHash high threshold: {self.phash_high_threshold}")
        print(f"  pHash gap threshold: {self.phash_gap_threshold}")
        print(f"  OCR batch size: {self.ocr_batch_size}")
        print(f"  OCR similarity threshold: {self.ocr_similarity_threshold}")
        print("="*70 + "\n")
        
        # Calculate sampling interval
        sample_every_n_frames = int(video_fps / self.fps)
        expected_samples = total_frames // sample_every_n_frames
        
        # Phase 1: pHash matching
        print("PHASE 1: pHash Matching")
        print("-"*70)
        
        timestamps = []
        matched_slides = []
        confidences = []
        match_methods = []
        
        low_confidence_queue = []  # Queue for OCR processing
        low_confidence_indices = []  # Track indices for later update
        
        frame_count = 0
        phash_matches = 0
        
        with tqdm(total=expected_samples, desc="  Matching frames with pHash") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_every_n_frames == 0:
                    timestamp = frame_count / video_fps
                    
                    # Try pHash matching
                    slide_idx, confidence, is_confident = self.find_closest_slide_phash(frame)
                    
                    timestamps.append(timestamp)
                    
                    if is_confident:
                        # High confidence match
                        matched_slides.append(slide_idx)
                        confidences.append(confidence)
                        match_methods.append('phash')
                        phash_matches += 1
                    else:
                        # Low confidence - queue for OCR
                        matched_slides.append(None)  # Placeholder
                        confidences.append(confidence)
                        match_methods.append('pending_ocr')
                        
                        low_confidence_queue.append(frame.copy())
                        low_confidence_indices.append(len(matched_slides) - 1)
                    
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        # Add final timestamp
        if timestamps:
            timestamps.append(duration)
            matched_slides.append(matched_slides[-1])
            confidences.append(confidences[-1])
            match_methods.append(match_methods[-1])
        
        print(f"\n  Phase 1 complete:")
        print(f"    pHash matches: {phash_matches}/{len(matched_slides)-1}")
        print(f"    Low confidence: {len(low_confidence_queue)}/{len(matched_slides)-1}")
        
        # Phase 2: OCR matching for low confidence frames
        if low_confidence_queue:
            print(f"\nPHASE 2: OCR Matching ({len(low_confidence_queue)} frames)")
            print("-"*70)
            
            # Extract OCR from reference slides if not done yet
            if not self.reference_ocr_texts:
                self.extract_ocr_from_slides()
            
            # Process low confidence frames in batches
            num_batches = (len(low_confidence_queue) + self.ocr_batch_size - 1) // self.ocr_batch_size
            ocr_matches = 0
            ocr_failures = 0
            
            for batch_idx in tqdm(range(num_batches), desc="  Processing OCR batches"):
                batch_start = batch_idx * self.ocr_batch_size
                batch_end = min(batch_start + self.ocr_batch_size, len(low_confidence_queue))
                
                batch_frames = low_confidence_queue[batch_start:batch_end]
                batch_indices = low_confidence_indices[batch_start:batch_end]
                
                # Extract OCR from batch
                batch_texts = self.process_ocr_batch(batch_frames)
                
                # Match each frame
                for frame_text, result_idx in zip(batch_texts, batch_indices):
                    slide_idx, similarity = self.find_closest_slide_ocr(frame_text)
                    
                    if slide_idx is not None:
                        matched_slides[result_idx] = slide_idx
                        confidences[result_idx] = similarity
                        match_methods[result_idx] = 'ocr'
                        ocr_matches += 1
                    else:
                        # Still no match
                        match_methods[result_idx] = 'no_match'
                        ocr_failures += 1
            
            print(f"\n  Phase 2 complete:")
            print(f"    OCR matches: {ocr_matches}/{len(low_confidence_queue)}")
            print(f"    No match: {ocr_failures}/{len(low_confidence_queue)}")
        
        # Generate summary statistics
        print(f"\n{'='*70}")
        print("MATCHING SUMMARY")
        print(f"{'='*70}")
        
        total_samples = len(matched_slides) - 1  # Exclude final duplicate
        phash_count = sum(1 for m in match_methods[:-1] if m == 'phash')
        ocr_count = sum(1 for m in match_methods[:-1] if m == 'ocr')
        no_match_count = sum(1 for m in match_methods[:-1] if m == 'no_match')
        
        print(f"  Total samples: {total_samples}")
        print(f"  pHash matches: {phash_count} ({100*phash_count/total_samples:.1f}%)")
        print(f"  OCR matches: {ocr_count} ({100*ocr_count/total_samples:.1f}%)")
        print(f"  No match: {no_match_count} ({100*no_match_count/total_samples:.1f}%)")
        
        # Condense to time ranges
        results = self.condense_to_ranges(timestamps, matched_slides, match_methods)
        
        # Save results
        results_path = output_path / "slide_timestamps.json"
        detailed_results = {
            'metadata': {
                'video_path': self.video_path,
                'pdf_path': self.pdf_path,
                'duration': duration,
                'sampling_fps': self.fps,
                'phash_high_threshold': self.phash_high_threshold,
                'phash_gap_threshold': self.phash_gap_threshold,
                'ocr_batch_size': self.ocr_batch_size,
                'ocr_similarity_threshold': self.ocr_similarity_threshold,
                'total_slides_in_pdf': len(self.reference_slides),
                'method': 'hybrid (pHash + DeepSeek OCR)'
            },
            'statistics': {
                'total_samples': total_samples,
                'phash_matches': phash_count,
                'ocr_matches': ocr_count,
                'no_matches': no_match_count
            },
            'segments': results
        }
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_path}")
        
        # Save debug info
        debug_path = output_path / "debug_matching.json"
        debug_data = []
        for ts, slide, conf, method in zip(timestamps[:-1], matched_slides[:-1], 
                                           confidences[:-1], match_methods[:-1]):
            debug_data.append({
                'timestamp': round(ts, 2),
                'slide': slide + 1 if slide is not None else None,
                'confidence': round(conf, 4),
                'method': method
            })
        
        with open(debug_path, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        print(f"✓ Debug info saved to {debug_path}")
        
        # Extract representative frames
        print(f"\nExtracting representative frames...")
        cap = cv2.VideoCapture(self.video_path)
        extracted_count = 0
        
        for segment in results:
            if segment['slide_number'] is None:
                continue
            
            mid_time = (segment['start_time'] + segment['end_time']) / 2
            cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
            ret, frame = cap.read()
            
            if ret:
                frame_path = output_path / f"slide_{segment['slide_number']:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                extracted_count += 1
        
        cap.release()
        print(f"✓ Extracted {extracted_count} representative frames")
        
        return detailed_results
    
    def condense_to_ranges(self, timestamps: List[float], 
                          slides: List[Optional[int]],
                          methods: List[str]) -> List[Dict]:
        """Condense consecutive identical slides into time ranges."""
        if not timestamps or not slides:
            return []
        
        ranges = []
        current_slide = slides[0]
        current_method = methods[0]
        start_time = timestamps[0]
        
        for i in range(1, len(slides)):
            if slides[i] != current_slide:
                ranges.append({
                    'slide_number': current_slide + 1 if current_slide is not None else None,
                    'start_time': start_time,
                    'end_time': timestamps[i],
                    'duration': timestamps[i] - start_time,
                    'match_method': current_method
                })
                
                current_slide = slides[i]
                current_method = methods[i]
                start_time = timestamps[i]
        
        # Add final segment
        ranges.append({
            'slide_number': current_slide + 1 if current_slide is not None else None,
            'start_time': start_time,
            'end_time': timestamps[-1],
            'duration': timestamps[-1] - start_time,
            'match_method': current_method
        })
        
        return ranges


def extract_class_number(filename: str) -> Optional[int]:
    """
    Extract class number from filename.
    
    For videos: expects format like "01_<id>.mp4" (class number as prefix)
    For PDFs: expects format like "<id>_class1.pdf" (class number as suffix)
    
    Returns:
        Class number as integer, or None if not found
    """
    # Try video format: 01_something.mp4
    match = re.match(r'^(\d+)_.*\.mp4$', filename)
    if match:
        return int(match.group(1))
    
    # Try PDF format: something_class1.pdf
    match = re.search(r'_class(\d+)\.pdf$', filename)
    if match:
        return int(match.group(1))
    
    return None


def discover_course_pairs(course_dir: Path) -> List[Dict[str, any]]:
    """
    Discover video-PDF pairs in course directory.
    
    Args:
        course_dir: Path to course directory containing 'cropped_videos' and 'pdf' subdirs
    
    Returns:
        List of dictionaries with 'class_num', 'video_path', and 'pdf_path'
    """
    videos_dir = course_dir / "cropped_videos"
    pdf_dir = course_dir / "pdf"
    
    if not videos_dir.exists():
        raise ValueError(f"Videos directory not found: {videos_dir}")
    
    if not pdf_dir.exists():
        raise ValueError(f"PDF directory not found: {pdf_dir}")
    
    # Find all videos
    video_files = {}
    for video_file in videos_dir.glob("*.mp4"):
        class_num = extract_class_number(video_file.name)
        if class_num is not None:
            video_files[class_num] = video_file
    
    # Find all PDFs
    pdf_files = {}
    for pdf_file in pdf_dir.glob("*.pdf"):
        class_num = extract_class_number(pdf_file.name)
        if class_num is not None:
            pdf_files[class_num] = pdf_file
    
    # Match pairs
    pairs = []
    for class_num in sorted(set(video_files.keys()) & set(pdf_files.keys())):
        pairs.append({
            'class_num': class_num,
            'video_path': str(video_files[class_num]),
            'pdf_path': str(pdf_files[class_num])
        })
    
    # Report unmatched files
    unmatched_videos = set(video_files.keys()) - set(pdf_files.keys())
    unmatched_pdfs = set(pdf_files.keys()) - set(video_files.keys())
    
    if unmatched_videos:
        print(f"⚠ Warning: Videos without matching PDFs for classes: {sorted(unmatched_videos)}")
    
    if unmatched_pdfs:
        print(f"⚠ Warning: PDFs without matching videos for classes: {sorted(unmatched_pdfs)}")
    
    return pairs


def process_course(course_dir: str,
                  output_dir: str,
                  fps: float = 1.0,
                  phash_high_threshold: float = 0.7,
                  phash_gap_threshold: float = 0.05,
                  ocr_batch_size: int = 16,
                  ocr_similarity_threshold: float = 0.7):
    """
    Process entire course directory.
    
    Args:
        course_dir: Path to course directory
        output_dir: Output directory for results
        Other args: Same as HybridSlideExtractor
    """
    course_path = Path(course_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COURSE SLIDE EXTRACTION")
    print("="*70)
    print(f"Course directory: {course_path}")
    print(f"Output directory: {output_path}")
    print("="*70 + "\n")
    
    # Discover video-PDF pairs
    print("Discovering video-PDF pairs...")
    pairs = discover_course_pairs(course_path)
    
    if not pairs:
        print("❌ Error: No matching video-PDF pairs found!")
        return
    
    print(f"✓ Found {len(pairs)} matching pairs:")
    for pair in pairs:
        print(f"  Class {pair['class_num']:02d}: {Path(pair['video_path']).name} + {Path(pair['pdf_path']).name}")
    print()
    
    # Process each pair
    course_results = []
    
    for i, pair in enumerate(pairs, 1):
        class_num = pair['class_num']
        video_path = pair['video_path']
        pdf_path = pair['pdf_path']
        
        print("\n" + "="*70)
        print(f"PROCESSING CLASS {class_num} ({i}/{len(pairs)})")
        print("="*70)
        
        # Create output directory for this class
        class_output_dir = output_path / f"class_{class_num:02d}"
        
        try:
            # Create extractor
            extractor = HybridSlideExtractor(
                video_path=video_path,
                pdf_path=pdf_path,
                fps=fps,
                phash_high_threshold=phash_high_threshold,
                phash_gap_threshold=phash_gap_threshold,
                ocr_batch_size=ocr_batch_size,
                ocr_similarity_threshold=ocr_similarity_threshold
            )
            
            # Extract slides
            result = extractor.extract_slides(output_dir=str(class_output_dir))
            
            # Store result
            course_results.append({
                'class_num': class_num,
                'video_file': Path(video_path).name,
                'pdf_file': Path(pdf_path).name,
                'output_dir': str(class_output_dir),
                'statistics': result['statistics'],
                'metadata': result['metadata']
            })
            
            print(f"\n✓ Class {class_num} complete")
            
        except Exception as e:
            print(f"\n❌ Error processing class {class_num}: {e}")
            import traceback
            traceback.print_exc()
            
            course_results.append({
                'class_num': class_num,
                'video_file': Path(video_path).name,
                'pdf_file': Path(pdf_path).name,
                'error': str(e)
            })
    
    # Save course-wide summary
    print("\n" + "="*70)
    print("GENERATING COURSE SUMMARY")
    print("="*70)
    
    summary_path = output_path / "course_summary.json"
    summary = {
        'course_dir': str(course_path),
        'total_classes': len(pairs),
        'processing_params': {
            'fps': fps,
            'phash_high_threshold': phash_high_threshold,
            'phash_gap_threshold': phash_gap_threshold,
            'ocr_batch_size': ocr_batch_size,
            'ocr_similarity_threshold': ocr_similarity_threshold
        },
        'classes': course_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Course summary saved to {summary_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("COURSE SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in course_results if 'error' not in r)
    failed = len(course_results) - successful
    
    print(f"Total classes: {len(pairs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print("\nPer-class statistics:")
        for result in course_results:
            if 'error' not in result:
                stats = result['statistics']
                class_num = result['class_num']
                total = stats['total_samples']
                phash = stats['phash_matches']
                ocr = stats['ocr_matches']
                print(f"  Class {class_num:02d}: "
                      f"{total} samples, "
                      f"{phash} pHash ({100*phash/total:.1f}%), "
                      f"{ocr} OCR ({100*ocr/total:.1f}%)")
    
    if failed > 0:
        print("\nFailed classes:")
        for result in course_results:
            if 'error' in result:
                print(f"  Class {result['class_num']:02d}: {result['error']}")
    
    print("\n" + "="*70)
    print("✓ COURSE PROCESSING COMPLETE")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract slides from course videos using hybrid pHash + DeepSeek OCR matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire course
  python extract_slides_course.py /path/to/course --output results/

  # Custom sampling rate and thresholds
  python extract_slides_course.py /path/to/course -o results/ --fps 2.0 --phash-threshold 0.75

  # Adjust OCR batch size
  python extract_slides_course.py /path/to/course -o results/ --ocr-batch-size 32

Directory structure expected:
  course/
    ├── cropped_videos/
    │   ├── 01_video.mp4
    │   ├── 02_video.mp4
    │   └── ...
    └── pdf/
        ├── slides_class1.pdf
        ├── slides_class2.pdf
        └── ...

Matching strategy:
  1. Videos are matched to PDFs by class number
  2. Each video-PDF pair is processed independently
  3. Results are organized by class in separate subdirectories
        """
    )
    
    parser.add_argument(
        'course_dir',
        type=str,
        help='Path to course directory containing "cropped_videos" and "pdf" subdirectories'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second to sample from videos (default: 1.0)'
    )
    
    parser.add_argument(
        '--phash-threshold',
        type=float,
        default=0.7,
        help='High confidence threshold for pHash matching (default: 0.7)'
    )
    
    parser.add_argument(
        '--gap-threshold',
        type=float,
        default=0.05,
        help='Minimum gap between top 2 pHash matches (default: 0.05)'
    )
    
    parser.add_argument(
        '--ocr-batch-size',
        type=int,
        default=16,
        help='Batch size for DeepSeek OCR processing (default: 16)'
    )
    
    parser.add_argument(
        '--ocr-similarity-threshold',
        type=float,
        default=0.7,
        help='Threshold for OCR sentence similarity (default: 0.7)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    course_path = Path(args.course_dir)
    if not course_path.exists():
        print(f"❌ Error: Course directory not found: {args.course_dir}")
        return 1
    
    if not course_path.is_dir():
        print(f"❌ Error: Course path is not a directory: {args.course_dir}")
        return 1
    
    try:
        process_course(
            course_dir=args.course_dir,
            output_dir=args.output,
            fps=args.fps,
            phash_high_threshold=args.phash_threshold,
            phash_gap_threshold=args.gap_threshold,
            ocr_batch_size=args.ocr_batch_size,
            ocr_similarity_threshold=args.ocr_similarity_threshold
        )
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
