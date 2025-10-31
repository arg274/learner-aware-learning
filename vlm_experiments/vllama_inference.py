import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import argparse
from glob import glob
import json
import regex

parser = argparse.ArgumentParser()

# Create a mutually exclusive group for input type
input_type_group = parser.add_mutually_exclusive_group(required=True)
input_type_group.add_argument('-v', '--video', action='store_const', const='video', dest='input_type', help='Process lecture video input')
input_type_group.add_argument('-m', '--multi-image', action='store_const', const='multi-image', dest='input_type', help='Process slide images all at once')
input_type_group.add_argument('-s', '--single-image', action='store_const', const='single-image', dest='input_type', help='Process a single image per prompt')

args = parser.parse_args()

model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

if args.input_type == 'single-image':
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
    
    extractions = []
    for i, filename in enumerate(sorted(glob('./Fintech/clip2_slides/*.jpg')), start=1):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": filename}},
                    {"type": "text", "text": single_prompt},
                ],
            },
        ]
        
        inputs = processor(conversation=message, return_tensors="pt", padding=True)
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=512) # type: ignore
        output_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
        
        response = output_texts[0]
        print('--- RESPONSE ---\n', response)
        json_string = max(regex.findall(r'\{(?:[^{}]|(?R))*\}', response), key=len)
        print('--- JSON ---\n', json_string)
        data = json.loads(json_string)
        data['slide_number'] = i
        extractions.append(data)
        
else:
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

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {"video_path": "./Fintech/Lecture1_clip-2.mp4", "fps": 1, "max_frames": 16}
                    },
                    {"type": "text", "text": video_prompt},
                ],
            }
        ]

    else:
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

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": filename}} for filename in sorted(glob('./Fintech/clip2_slides/*.jpg'))                
                ] + [
                    {"type": "text", "text": multi_prompt}
                ]
            }
        ]

    inputs = processor(conversation=message, return_tensors="pt", padding=True)
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=512) # type: ignore
    output_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

    response = output_texts[0]
    print('--- RESPONSE ---\n', response)
    json_string = max(regex.findall(r'\[(?:[^[\]]|(?R))*\]', response), key=len)
    print('--- JSON ---\n', json_string)
    extractions = json.loads(json_string)

with open(f'vlm_results/vllama_{args.input_type}.json', 'w', encoding='utf-8') as f:
    json.dump(extractions, f, ensure_ascii=False, indent=4)
