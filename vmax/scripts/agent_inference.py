from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from PIL import Image
from datasets import Dataset, load_dataset
from transformers import TextStreamer
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import argparse
import re
import ast

IMAGE_ROOT = "/home/lz457/zhoulab/vlm-ad/vl-finetuning/data/"

def decode_waypoints(vlm_output):
    match = re.search('<waypoints>(.*?)</waypoints>', vlm_output, re.DOTALL)

    waypoints_list = []
    if match:
        # The actual content is in the first captured group
        list_string = match.group(1)
        
        print(f"Extracted string: {list_string}")
        
        list_string_cleaned = list_string.strip()

        try:
            # 3. Safely convert the string to a list
            waypoints_list = ast.literal_eval(list_string_cleaned)
            
            print("✅ Successfully extracted and converted list:")
            print(waypoints_list)
            print(f"\nData type: {type(waypoints_list)}")
            return waypoints_list
            
        except (ValueError, SyntaxError) as e:
            print(f"Error: Could not parse the extracted string. Details: {e}")

    else:
        print("No <waypoint> tags found in the output.")

if True:
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "/home/lz457/zhoulab/vlm-ad/vl-finetuning/scripts/outputs/checkpoint-104",#unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!

try:
    dataset = load_dataset("json", data_files="/home/lz457/zhoulab/vlm-ad/vl-finetuning/annotated_data/annotated_data.jsonl", split="train")
except FileNotFoundError:
    print("="*80)
    print("ERROR: `my_dataset.jsonl` not found.")
    print("Please create a jsonl file with 'image_path' and 'messages' keys.")
    print("See the documentation for the expected format.")
    print("="*80)
    exit()

# --- Function to load images from paths ---
def load_image(example):
    try:
        image = Image.open(IMAGE_ROOT+example["file_name"]).convert("RGB")
        return {"image": image}
    except Exception as e:
        print(f"Error loading image {example['file_name']}: {e}")
        # Return a placeholder or handle the error as needed
        # For simplicity, we'll skip examples with bad images
        return {"image": None}

# Apply the function to load images.
print(dataset[-1]["file_name"])
dataset = dataset.map(load_image, remove_columns=["file_name"])


# Filter out examples where the image failed to load
dataset = dataset.filter(lambda example: example["image"] is not None)

def create_optimized_prompt():
    """
    Creates a shorter, more direct prompt for better and consistent performance.
    """
    return (
        "You are an AI autonomous driving assistant. The vehicle is following its planned path. Based on the Birds-Eye-View image, determine a safe action following the road with overall navigation goal in mind. DISREGARD THE EGO's ANGULAR VELOCITY. You may slow down or stop as needed, especially at red lights and to avoid hitting other vehicles. "
        "The ego-vehicle is the red car at (0, 0). Based on the vehicle velocity and yaw, consider the current path of the vehicle, and decide the next action with that path in mind; consider this with the context of the navigation goal and the image scene.."
        "Output a driving plan with these specs: "
        "1. **Trajectory:** 10 waypoints. "
        "2. **Time:** Path over the next 1 second. (0.1 second interval for each waypoint)"
        "3. **Coordinate System:** Absolute (x, y) coordinate system based on the image. "
        "4. **Format:** ONLY a formatted list of lists for [x, y] coordinates in meters. (e.g [[1, 1], [1, 2], [1, 3]])"
        "You must put your final waypoints inside of <waypoint></waypoint> tags, and nothing else. Avoid extensive calculations."
    )

instruction = create_optimized_prompt()
context = lambda sample: sample["ego_description"] + "The current navigation instruction is to reach the waypoint " + sample["nav_goal"] + "." + sample["light_description"]


if True:
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "image"},
            {"type" : "text",  "text"  : context(dataset[-3]) + instruction}
            ]
        }
    ]
    

image = dataset[-1]["image"]
image.show()


input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")


# text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048,
#                    use_cache = True, temperature = 0.001, min_p = 0.1)

generated_ids = model.generate(
    **inputs, 
    max_new_tokens=2048,
    use_cache=True, 
    temperature=0.001, 
    min_p=0.1
)

# 3. Decode the generated IDs into a string
# We use [0] because generate returns a batch, and we want the first result.
# We also skip the prompt part of the tokens to get only the new generation.
llm_output_string = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


print(llm_output_string)
wp_list = decode_waypoints(llm_output_string)
print(wp_list)
print(wp_list[0])

