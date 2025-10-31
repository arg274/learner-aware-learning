from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from glob import glob
import regex
import json
import argparse

parser = argparse.ArgumentParser()

# Create a mutually exclusive group for input type
input_type_group = parser.add_mutually_exclusive_group(required=True)
input_type_group.add_argument('-v', '--video', action='store_const', const='video', dest='input_type', help='Process lecture video input')
input_type_group.add_argument('-m', '--multi-image', action='store_const', const='multi-image', dest='input_type', help='Process slide images all at once')
input_type_group.add_argument('-s', '--single-image', action='store_const', const='single-image', dest='input_type', help='Process a single image per prompt')

args = parser.parse_args()

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct-AWQ", min_pixels=min_pixels, max_pixels=max_pixels)

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
                    "video": "./Fintech/Lecture1_clip-2.mp4"
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
                {"type": "image", "image": filename} for filename in sorted(glob('./Fintech/clip2_slides/*.jpg'))                
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
                {"type": "image", "image": filename},
                {"type": "text", "text": single_prompt},
            ],
        },
    ] for filename in sorted(glob('./Fintech/clip2_slides/*.jpg'))]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

if args.input_type == 'single-image':
    extractions = []
    for i, response in enumerate(output_texts, start=1):
        json_string = max(regex.findall(r'\{(?:[^{}]|(?R))*\}', response), key=len)
        print('--- RESPONSE ---\n', response)
        print('--- JSON ---\n', json_string)
        data = json.loads(json_string)
        data['slide_number'] = i
        extractions.append(data)
else:
    response = output_texts[0]
    json_string = max(regex.findall(r'\[(?:[^[\]]|(?R))*\]', response), key=len)
    print('--- RESPONSE ---\n', response)
    print('--- JSON ---\n', json_string)
    extractions = json.loads(json_string)

with open(f'vlm_results/qwen_{args.input_type}.json', 'w', encoding='utf-8') as f:
    json.dump(extractions, f, ensure_ascii=False, indent=4)
