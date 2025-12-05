from transformers import AutoModelForCausalLM, AutoTokenizer
import webvtt
import json
from utils.time import format_time
from glob import glob
import argparse

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get 2-minute blocks of captions
def get_captions_blocks(captions_path, video_length):
    captions = webvtt.read(captions_path)
    blocks = []
    
    time = 0
    while time < video_length:
        blocks.append([])
        for caption in captions.iter_slice(format_time(time), format_time(min(time + 120, video_length))):
            blocks[-1].append(caption.text)
        time += 120
    return blocks

def summarize(text_list, summarized_before):
    if summarized_before:
        system_prompt = "Summarize the following list of summaries together."
    else:
        system_prompt = "Summarize the following list of captions from a 2-minute clip of a lecture video."
    prompt = json.dumps(text_list, ensure_ascii=False, indent=4)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
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
    return response

# Testing command:
# python3 summarize.py --course-path Fintech/MIT_15.S08
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course-path", "-p", type=str, required=True, help="Path to the course directory.")
    args = parser.parse_args()
    
    classes = sorted(glob(args.course_path + "/class*"))
    for class_path in classes:
        class_name = class_path.split("/")[-1]
        print(f"Processing {class_name}...")
        
        # Load slide timestamps to get video length
        with open(class_path + "/slide_timestamps.json", 'r') as f:
            slide_timestamps = json.load(f)
        video_length = slide_timestamps["metadata"]["duration"]
        
        # Get captions blocks
        subtitle_path = glob(class_path + "/*.vtt")[0]
        caption_blocks = get_captions_blocks(subtitle_path, video_length)
        
        # Summarize each block
        two_minute_summaries = []
        for i, block in enumerate(caption_blocks):
            print(f"  Summarizing 2-minute block {i+1}/{len(caption_blocks)}...")
            summary = summarize(block, summarized_before=False)
            two_minute_summaries.append(summary)
            
        # Summarize two-minute summaries into ten-minute summaries
        ten_minute_summaries = []
        for i in range(0, len(two_minute_summaries), 5):
            print(f"  Summarizing 10-minute block {i//5+1}/{(len(two_minute_summaries)+4)//5}...")
            summary = summarize(two_minute_summaries[i:i+5], summarized_before=True)
            ten_minute_summaries.append(summary)
        
        # Summarize all summaries
        print(f"  Summarizing all blocks together...")
        final_summary = summarize(ten_minute_summaries, summarized_before=True)
        
        # Save final summary
        with open(class_path + "/summary.txt", 'w') as f:
            f.write(final_summary)
        print(f"  Final summary saved to {class_path}/summary.txt")    
    