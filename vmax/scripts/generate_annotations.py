import os
import json
import argparse
import torch
import time
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def load_model_processor(checkpoint_path, flash_attn2=False, cpu_only=False):

    print("Loading model and processor...")
    device_map = 'cpu' if cpu_only else 'auto'
    torch_dtype = torch.bfloat16
    attn_implementation = 'flash_attention_2' if flash_attn2 and torch.cuda.is_available() else 'sdpa'
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    print("✅ Model and processor loaded successfully.")
    print(f"   - Model dtype: {model.dtype}")
    print(f"   - Model attention implementation: {model.config._attn_implementation}")
    return model, processor

def run_vlm_inference(model, processor, image_path: str, prompt: str, scene_desc: str):
    messages = [
        {"role": "user", "content": [
            {'type': 'image', 'image': f'file://{image_path}'},
            {'type': 'text', 'text': scene_desc + prompt},
        ]}
    ]
    

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors='pt'
    ).to(model.device)

    # Generation parameters for near-deterministic, fast output
    gen_kwargs = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.001, 'top_p': None, 'top_k': None}
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    
    output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response.strip()

def create_optimized_prompt():

    # return (
    #       "You are an AI autonomous driving assistant. The vehicle is following its planned path. Based on the Birds-Eye-View image, determine a safe action following the road with overall navigation goal in mind. DISREGARD THE EGO's ANGULAR VELOCITY. You may slow down or stop as needed, especially at red lights and to avoid hitting other vehicles. "
    #     "The ego-vehicle is the red car at (0, 0). Based on the vehicle velocity and yaw, consider the current path of the vehicle, and decide the next action with that path in mind; consider this with the context of the navigation goal and the image scene.."
    #     "Output a driving plan with these specs: "
    #     "1. **Trajectory:** 10 waypoints. "
    #     "2. **Time:** Path over the next 1 second. (0.1 second interval for each waypoint)"
    #     "3. **Coordinate System:** Absolute (x, y) coordinate system based on the image. "
    #     "4. **Format:** ONLY a formatted list of lists for [x, y] coordinates in meters. (e.g [[1, 1], [1, 2], [1, 3]])"
    #     "You must put your final waypoints inside of <waypoint></waypoint> tags, and nothing else. Avoid extensive calculations."
    # )

    return (
        "You are an AI autonomous driving navigator. Your job is to determine a safe, efficient path for the ego-vehicle "
        "to take in order to continue on its route based on the birds-eye-view image of the scene and the positions of other actors. The red box at the center of the image is the ego-vehicle."
        "The the numbered blue objects are other cars. All units are in terms of meters. The vehicle is currently following its planned route. Do not make any abrupt changes. You must obey all laws of the road such as stopping for red lights. You will generate a set of 10 waypoints that represent a safe, efficient action and path for the vehicle to take over the next 5 seconds, "
        "with 0.5 seconds between each waypoint. In your response, put the final waypoint output as a list of number coordinates (meters) inside of "
        "<waypoint></waypoint> tags and nothing else, e.g [[x1, y1], [x2, y2]]. When thinking, consider "
        "what action should the vehicle take (e.g merge, stop, continue, turn)? What is the vehicle's current heading, direction, and route? Think step by step, and put your thinking in <think></think> tags. Consider the following as additional context:"
    )


# Get image data and feed proper metadata
def process_all_data(model, processor, args):
 
    processed_files = set()
    output_mode = 'w'  


    if os.path.exists(args.output_file):
        print(f"Output file '{args.output_file}' already exists. Attempting to resume.")
        with open(args.output_file, 'r', encoding='utf-8') as f_out:
            for line in f_out:
                try:
                    data = json.loads(line)
                    if 'file_name' in data:
                        processed_files.add(data['file_name'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode a line in the existing output file. Skipping line.")
        output_mode = 'a'
        print(f"Found {len(processed_files)} previously processed files. Will skip these.")

    with open(args.output_file, output_mode, encoding='utf-8') as f_out:
        with open(args.input_file, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
            

            task_prompt = create_optimized_prompt()

            for line in tqdm(lines, desc="Annotating dataset"):
                try:
                    original_data = json.loads(line)
                    relative_path = original_data.get("file_name")
                    scene_description = (
                        # original_data.get("ego_description") + "The current navigation instruction is to reach the waypoint " + original_data.get("nav_goal") + "." + original_data.get("light_description")
                        original_data.get("ego_description") + "Navigation notes: " + original_data.get("instruction") + "." + original_data.get("light_description") + original_data.get("closest")

                    )
                    if (original_data.get("nav_goal") == '(-1.0, -1.0)'):
                        continue
                    if not relative_path:
                        continue
                    

                    if relative_path in processed_files:
                        continue  # Skip this file as it's already processed

                    full_image_path = os.path.join(args.image_dir, relative_path)
                    if not os.path.exists(full_image_path):
                        continue

                    # Inference
                    vlm_output = run_vlm_inference(model, processor, full_image_path, task_prompt, "Scene description" + scene_description)
                    
                    # Saving
                    original_data['output_reasoning'] = vlm_output
                    print(original_data["file_name"]+"\n")
                    print(vlm_output)
                    f_out.write(json.dumps(original_data) + '\n')

                    f_out.flush()

                except Exception as e:
                    print(f"\n processing line: {line.strip()}. Error: {e}. Skipping.")

    print(f"\n✅ Batch processing complete. Results saved to {args.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Optimized batch processing script for Qwen-VL.")
    parser.add_argument('--input-file', type=str, required=True, help="Path to the input metadata.jsonl file.")
    parser.add_argument('--output-file', type=str, required=True, help="Path to save the output .jsonl file.")
    parser.add_argument('--image-dir', type=str, default='.', help="Base directory where image folders are located.")
    parser.add_argument('--checkpoint-path', type=str, required=True, help="Path to the 32B model checkpoint.")
    parser.add_argument('--flash-attn2', action='store_true', help="Enable Flash Attention 2 (Highly Recommended).")
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()

    model, processor = load_model_processor(
        checkpoint_path=args.checkpoint_path, flash_attn2=args.flash_attn2, cpu_only=args.cpu_only
    )
    
    process_all_data(model, processor, args)

if __name__ == '__main__':
    main()

