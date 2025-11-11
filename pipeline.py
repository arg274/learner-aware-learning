from slide_extraction.extract import SlideExtractor
from vlm.vlm import VLM
import argparse
import os

# Testing command:
# python3 pipeline.py --input-pdf Fintech/clip1_slides.pdf --input-video Fintech/Lecture1_clip-1.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-pdf", "-p", type=str, required=True, help="Path to the input PDF file containing slides.")
    parser.add_argument("--input-video", "-v", type=str, required=True, help="Path to the input lecture video file.")
    parser.add_argument("--output-dir", "-o", type=str, default="./output", help="Directory to save the outputs.")
    args = parser.parse_args()
    
    # Step 1: Extract slides from the video
    print("Starting slide extraction process...")
    extractor = SlideExtractor(pdf_path=args.input_pdf, video_path=args.input_video, sample_interval=1.0)
    video_name = args.input_video.split("/")[-1].split(".")[0]
    slides_path = args.output_dir + "/" + video_name
    os.makedirs(slides_path, exist_ok=True)
    extractor.extract_slides(output_dir=slides_path)
    print(f"Slides extracted and saved to {slides_path}")
    
    # Step 2: Use VLM to extract text from slides
    print(f"Starting VLM text extraction...")
    vlm = VLM()
    vlm.inference(slides_path=slides_path)
    print(f"Text extraction completed. Results saved in {slides_path}/{video_name}.json")