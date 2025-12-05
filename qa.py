import argparse
from utils.time import convert_time_to_seconds, format_time
from pathlib import Path
import json
from glob import glob
import webvtt

# time in seconds
# scope can be 'local', 'global', or 'both'
def get_context(class_path: Path, time: float, scope: str):
    context = {}
    
    # Load slide timestamps, transcriptions, and subtitles
    class_name = class_path.name
    with open(class_path.parent / "cleaned" / f"{class_name}.json", 'r') as f:
        slide_timestamps = json.load(f)
    with open(class_path / f"{class_name}.json", 'r') as f:
        slides = json.load(f)
    subtitle_path = glob(str(class_path / "*.vtt"))[0]
    subtitles = webvtt.read(subtitle_path)
    
    if scope in ['local', 'both']:
        # Get current slide based on time
        current_slide_number = None
        for slide in slide_timestamps["segments"]:
            start = slide["start_time"]
            end = slide["end_time"]
            if start <= time <= end:
                current_slide_number = slide["slide_number"]
                break
        
        # Find the slide transcription
        if current_slide_number is None:
            current_slide = None
        else:
            for slide in slides:
                if slide["slide_number"] == current_slide_number:
                    current_slide = slide
                    break
                
        # Get last minute of subtitles
        current_slide["captions"] = []
        for caption in subtitles.iter_slice(format_time(max(0, time - 60)), format_time(time)):
            current_slide["captions"].append(caption.text)
            
        # Add to context
        context["local"] = current_slide
        
    return json.dumps(context, ensure_ascii=False, indent=4)
        
# Testing command:
# python3 qa.py --class-path Fintech/MIT_15.S08/class_11 --time 16:45
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-path", "-p", type=str, required=True, help="Path to the class directory.")
    parser.add_argument("--time", "-t", type=str, required=True, help="Timestamp in h:mm:ss format.")
    args = parser.parse_args()
    
    context = get_context(Path(args.class_path), convert_time_to_seconds(args.time), scope='both')
    print(context)