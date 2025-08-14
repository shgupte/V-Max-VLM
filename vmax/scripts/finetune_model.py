from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from PIL import Image
from datasets import Dataset, load_dataset
from transformers import TextStreamer
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import argparse

IMAGE_ROOT = "/home/lz457/zhoulab/vlm-ad/vl-finetuning/data/"

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

# #################################################################################
# 3. DATA PREPARATION: Load and format your JSONL dataset
# #################################################################################
print("Step 3: Loading and preparing JSONL dataset...")

# --- Load the JSONL file ---
# !! IMPORTANT !!
# Replace "my_dataset.jsonl" with the actual path to your data file.
# Your image paths inside the jsonl should be accessible from this script.
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
dataset = dataset.map(load_image, remove_columns=["file_name"])

# Filter out examples where the image failed to load
dataset = dataset.filter(lambda example: example["image"] is not None)

print(dataset)

# --- Formatting Function (this remains the same) ---
# def formatting_prompts_func(example):
#     text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
#     return {"text": text}

# # Apply the formatting function
# dataset = dataset.map(formatting_prompts_func)

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

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : context(sample) + instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["output_reasoning"]} ]
        },
    ]
    return { "messages" : conversation }

converted_dataset = [convert_to_conversation(sample) for sample in dataset]


FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = 30,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
