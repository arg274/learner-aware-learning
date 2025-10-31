from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import argparse
import json
import regex
from glob import glob
import torch

parser = argparse.ArgumentParser()

# Create a mutually exclusive group for input type
input_type_group = parser.add_mutually_exclusive_group(required=True)
input_type_group.add_argument('-v', '--video', action='store_const', const='video', dest='input_type', help='Process lecture video input')
input_type_group.add_argument('-m', '--multi-image', action='store_const', const='multi-image', dest='input_type', help='Process slide images all at once')
input_type_group.add_argument('-s', '--single-image', action='store_const', const='single-image', dest='input_type', help='Process a single image per prompt')

args = parser.parse_args()

model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

if args.input_type == 'video':
    video_prompt = """You are given a video of a slideshow.
For each slide, do the following:
1. Identify the main heading or title of the slide and put it in a "title" field.  
2. Transcribe all other visible text and annotations exactly as they appear.  
- Each distinct line of text should be a separate string in the "text" array.  
- Include labels on diagrams or arrows as separate lines.  
- Ignore decorative elements with no text.  
3. Do not summarize or paraphrase — copy the text faithfully.  

Only output one JSON object per slide you receive.
Return the result as a JSON array in the same order as the slides, using this format:

[
    {
        "title": "Introduction to Machine Learning",
        "text": [
            "Basics and Applications",
            "What is ML?",
            "Why it matters"
        ],
        "slide_number": 1
    },
    {
        "title": "Supervised Learning",
        "text": [
            "y = f(x) + ε",
            "Training data with labels",
            "Prediction on new data"
        ],
        "slide_number": 2
    }
]
"""

    messages = [[
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "path": "./Fintech/Lecture1_clip-2.mp4"
                },
                {"type": "text", "text": video_prompt},
            ],
        }
    ]]

elif args.input_type == 'multi-image':
    multi_prompt = """You are given a sequence of images, each representing one slide in a slideshow.  
For each slide, do the following:
1. Identify the main heading or title of the slide and put it in a "title" field.  
2. Transcribe all other visible text and annotations exactly as they appear.  
- Each distinct line of text should be a separate string in the "text" array.  
- Include labels on diagrams or arrows as separate lines.  
- Ignore decorative elements with no text.  
3. Do not summarize or paraphrase — copy the text faithfully.  

Only output one JSON object per slide you receive.
Return the result as a JSON array in the same order as the slides, using this format:

[
    {
        "title": "Introduction to Machine Learning",
        "text": [
            "Basics and Applications",
            "What is ML?",
            "Why it matters"
        ],
        "slide_number": 1
    },
    {
        "title": "Supervised Learning",
        "text": [
            "y = f(x) + ε",
            "Training data with labels",
            "Prediction on new data"
        ],
        "slide_number": 2
    }
]
"""

    messages = [[
        {
            "role": "user",
            "content": [
                {"type": "image", "path": filename} for filename in sorted(glob('./Fintech/clip2_slides/*.jpg'))                
            ] + [
                {"type": "text", "text": multi_prompt}
            ]
        }
    ]]

else:
    single_prompt = """You are given an image of a slide in a slideshow. Do the following:
1. Identify the main heading or title of the slide and put it in a "title" field.  
2. Transcribe all other visible text and annotations exactly as they appear.  
   - Each distinct line of text should be a separate string in the "text" array.  
   - Include labels on diagrams or arrows as separate lines.  
   - Ignore decorative elements with no text.  
3. Do not summarize or paraphrase — copy the text faithfully.  

Return the result as a properly-formatted JSON object, using this format:

{
    "title": "Introduction to Machine Learning",
    "text": [
        "Basics and Applications",
        "What is ML?",
        "Why it matters"
    ]
}
"""

    messages = [[
        {
            "role": "user",
            "content": [
                {"type": "image", "path": filename},
                {"type": "text", "text": single_prompt},
            ],
        },
    ] for filename in sorted(glob('./Fintech/clip2_slides/*.jpg'))]

inputs = processor.apply_chat_template(
    messages,
    num_frames=8,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    padding=True,
    return_tensors="pt"
).to(model.device, torch.float16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=512)
generate_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
]
output_texts = processor.batch_decode(generate_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

if args.input_type == 'single-image':
    extractions = []
    for i, response in enumerate(output_texts, start=1):
        print('--- RESPONSE ---\n', response)
        json_string = max(regex.findall(r'\{(?:[^{}]|(?R))*\}', response), key=len)
        print('--- JSON ---\n', json_string)
        data = json.loads(json_string)
        data['slide_number'] = i
        extractions.append(data)
else:
    response = output_texts[0]
    print('--- RESPONSE ---\n', response)
    json_string = max(regex.findall(r'\[(?:[^[\]]|(?R))*\]', response), key=len)
    print('--- JSON ---\n', json_string)
    extractions = json.loads(json_string)

with open(f'vlm_results/llava_{args.input_type}.json', 'w', encoding='utf-8') as f:
    json.dump(extractions, f, ensure_ascii=False, indent=4)