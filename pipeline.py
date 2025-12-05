from slide_extraction.extract_ssim import SlideExtractor
from vlm.vlm import VLM
import argparse
import os
import shutil
from glob import glob
import json
import webvtt
from utils.time import format_time

# Testing command:
# python3 pipeline.py --input-pdf Fintech/MIT15-S08S20_class1.pdf --input-video https://www.youtube.com/watch?v=90JWoR9MfYU
# python3 pipeline.py --input-pdf Fintech/MIT15-S08S20_class11.pdf --input-video https://www.youtube.com/watch?v=iahUTx27HUg
if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pdf", "-p", type=str, required=True, help="Path to the input PDF file containing slides.")
    parser.add_argument("--input-video", "-v", type=str, required=True, help="YouTube URL of the input lecture video.")
    parser.add_argument("--output-dir", "-o", type=str, default="./output", help="Directory to save the outputs.")
    args = parser.parse_args()

    # Step 1: Download video and subtitles
    print("\n[1/5] Downloading video and subtitles...")
    os.system(rf'yt-dlp --write-subs --sub-langs "en.*" -f mp4 -o "./tmp/%(title)s.%(ext)s" {args.input_video}')
    video_path = glob('./tmp/*.mp4')[0]
    subtitle_path = glob('./tmp/*.vtt')[0]
    print(f"Video downloaded and saved to {video_path}")
    print(f"Subtitles downloaded and saved to {subtitle_path}")
    
    # Step 2: Run SAM3 slide segmentation
    print("\n[2/5] Running SAM3 slide segmentation...")
    seg_path = 'tmp/segmented'
    os.system(rf'python3 slide_extraction/sam_next.py "{video_path}" "{seg_path}"')
    print(f"SAM3 slide segmentation completed. Results saved in {seg_path}")

    # Step 3: Extract slides from the video
    print("\n[3/5] Starting slide extraction process...")
    video_name = video_path.split("/")[-1].split(".")[0]
    video_seg_path = f"{seg_path}/{video_name}.mp4"
    extractor = SlideExtractor(pdf_path=args.input_pdf, video_path=video_seg_path, sample_interval=1.0)
    slides_path = args.output_dir + "/" + video_name
    os.makedirs(slides_path, exist_ok=True)
    extractor.extract_slides(output_dir=slides_path)
    print(f"Slides extracted and saved to {slides_path}")
    
    # Step 4: Use VLM to extract text from slides
    print("\n[4/5] Starting VLM text extraction...")
    vlm = VLM()
    vlm.inference(slides_path=slides_path)
    print(f"Text extraction completed. Results saved in {slides_path}/{video_name}.json")
    
    # Step 5: Add subtitles to JSON
    # print("\n[5/5] Extracting subtitles...")
    # subtitles = webvtt.read(subtitle_path)
    # with open(f"{slides_path}/slide_timestamps.json", 'r') as f:
    #     slide_timestamps = json.load(f)
    # with open(f"{slides_path}/{video_name}.json", 'r') as f:
    #     slides = json.load(f)
        
    # for timestamp in slide_timestamps["segments"]:
    #     start = format_time(timestamp["start_time"])
    #     end = format_time(timestamp["end_time"])
    #     slide_number = timestamp["slide_number"]
    #     if "captions" not in slides[slide_number - 1]:
    #         slides[slide_number - 1]["captions"] = []
        
    #     for caption in subtitles.iter_slice(start, end):
    #         slides[slide_number - 1]["captions"].append(caption.text)
    # with open(f"{slides_path}/{video_name}.json", 'w') as f:
    #     json.dump(slides, f, ensure_ascii=False, indent=4)
    # print(f"Subtitle extraction completed. Results saved in {slides_path}/{video_name}.json")

    # Cleanup
    shutil.rmtree('./tmp/')
