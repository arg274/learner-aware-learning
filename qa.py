import argparse
from utils.time import convert_time_to_seconds, format_time
from pathlib import Path
import json
from glob import glob
import webvtt
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    
    # Get local scope (current slide transcription and last minute of subtitles)
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
        current_slide["slide_title"] = current_slide.pop("title")
        current_slide["slide_text"] = current_slide.pop("text")
                
        # Get last minute of subtitles
        current_slide["captions"] = []
        for caption in subtitles.iter_slice(format_time(max(0, time - 60)), format_time(time)):
            current_slide["captions"].append(caption.text)
            
        # Add to context
        context["local"] = current_slide
        
    # Get global scope (entire slide transcriptions)
    if scope in ['global', 'both']:
        with open(class_path / "summary.txt", 'r') as f:
            summary = f.read()
            
        context["global"] = {
            "video_summary": summary
        }
        
    return json.dumps(context, ensure_ascii=False, indent=4)
        
# Testing commands:
# python3 qa.py --class-path Fintech/MIT_15.S08/class_01 --time 38:45 --question "Why does FinTech create concerns about privacy?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_02 --time 35:45 --question "What kind of factors will encourage people to use chatbots?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_03 --time 21:40 --question "Why is machine learning considered essential in the modern finance technology stack?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_04 --time 28:45 --question "Why would financial institutions want to open their platforms through different types of APIs if doing so could also create competition for their own services?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_05 --time 38:28 --question "Why does Bitcoin make up such a large share of the total cryptocurrency market cap?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_06 --time 1:09:30 --question "What is the main difference between retail stablecoins like Tether and wholesale stablecoin projects like JPM Coin?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_07 --time 37:10 --question "How are new companies able to compete in the mortgage origination market?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_08 --time 35:50 --question "Why have challenger banks grown so quickly compared to traditional banks?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_09 --time 1:05:30 --question "What would happen if the big banks started holding more crypto on their balance sheets?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_10 --time 16:58 --question "What does underwriting mean in the insurance process?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_11 --time 16:45 --question "Why did problems in the U.S. housing market spread to other countries?"
# python3 qa.py --class-path Fintech/MIT_15.S08/class_12 --time 42:07 --question "Why do regulators give startups more flexibility than big finance or big tech firms?"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-path", "-p", type=str, required=True, help="Path to the class directory.")
    parser.add_argument("--time", "-t", type=str, required=True, help="Timestamp in h:mm:ss format.")
    parser.add_argument("--question", "-q", type=str, required=True, help="Question to ask.")
    args = parser.parse_args()
    
    context = get_context(Path(args.class_path), convert_time_to_seconds(args.time), scope='local')
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for scope in ['none', 'local', 'global', 'both']:
        context = get_context(Path(args.class_path), convert_time_to_seconds(args.time), scope=scope)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers student questions based on the provided context from a lecture video."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{args.question}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"--- Context scope: {scope} ---")
        print(response)
        print()
