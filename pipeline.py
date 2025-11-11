from slide_extraction.extract import SlideExtractor
import argparse
import os

# python3 pipeline.py --input-pdf Fintech/clip1_slides.pdf --input-video Fintech/Lecture1_clip-1.mp4
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-pdf", "-p", type=str, required=True, help="Path to the input PDF file containing slides.")
    parser.add_argument("--input-video", "-v", type=str, required=True, help="Path to the input lecture video file.")
    parser.add_argument("--output-dir", "-o", type=str, default="./output", help="Directory to save the outputs.")
    args = parser.parse_args()
    
    print("Starting slide extraction process...")
    extractor = SlideExtractor(pdf_path=args.input_pdf, video_path=args.input_video, sample_interval=1.0)
    
    video_name = args.input_video.split("/")[-1].split(".")[0]
    slides_path = args.output_dir + "/" + video_name + "_slides"
    os.makedirs(slides_path, exist_ok=True)
    extractor.extract_slides(output_dir=slides_path)
    print(f"Slides extracted and saved to {slides_path}")