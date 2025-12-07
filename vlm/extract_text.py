from vlm import VLM
from glob import glob
import argparse

# python3 vlm/extract_text.py --course-path Fintech/MIT_15.S08
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course-path", "-p", type=str, required=True, help="Path to the course directory.")
    args = parser.parse_args()
    
    vlm = VLM()
    
    classes = sorted(glob(args.course_path + "/class*"))
    for class_path in classes:
        class_name = class_path.split("/")[-1]
        print(f"Processing {class_name}...")
        vlm.inference(slides_path=class_path)
