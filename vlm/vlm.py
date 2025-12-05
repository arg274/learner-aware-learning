from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from glob import glob
import regex
import json

class VLM():
    
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        
        self.prompt = """You are given an image of a slide in a slideshow. Do the following:
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
}"""

    def inference(self, slides_path):
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": filename},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ] for filename in sorted(glob(slides_path + "/*.jpg"))]
        
        # Loop through each slide and perform inference
        output_texts = []
        for msg in messages:
            text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(msg)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts += self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        # Extract JSON from outputs
        extractions = []
        for i, response in enumerate(output_texts, start=1):
            try:
                json_string = max(regex.findall(r'\{(?:[^{}]|(?R))*?\}', response), key=len)
            except ValueError:
                print(f"Warning: No JSON found in response for slide {i}. Skipping.")
                print(f"Response was:\n{response}\n--------------------------------")
                continue
            data = json.loads(json_string)
            data['slide_number'] = i
            extractions.append(data)
            
        slides_name = slides_path.split("/")[-1]
        with open(f'{slides_path}/{slides_name}.json', 'w', encoding='utf-8') as f:
            json.dump(extractions, f, ensure_ascii=False, indent=4)
