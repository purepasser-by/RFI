from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import os
from _utils import construct_train_test_ds, prepare_hal_train_test_ds, prepare_pair_data_loader, load_flow_model
from tqdm import tqdm
import math
import time
from model import RectifiedFlow, LinearUNet
from wrapper import Wrapper
from copy import deepcopy
import sys
import torch.nn as nn
import random
import numpy as np


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# Set random seed
SEED = 42
set_seed(SEED)

# Model configuration
model_name = "qwen"  # Changed to qwen
model_path = ".../QWEN_MODEL"  # Path to Qwen-VL model
image_path = ".../train2014"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training parameters
num_epochs = 100
token_pos = "last"  # Strategy for data processing
min_lr_scale = 0.7
num_warmup_steps = 100
layers = [15]  # Layers to intervene on
layer = layers[0]

k = 20
alpha = 1.5
ds_name = "rfi"

# Save path configuration
save_nn_name = f"Flow_{model_name}_epoch{num_epochs}"
save_res_name = f"Flow_{model_name}_k{k}_alpha{alpha}_epoch{num_epochs}"

res_dir = f"{model_name}_hal_results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

save_model_path = os.path.join(res_dir, save_nn_name) + f"_{layer}.pth"
save_res_path = os.path.join(res_dir, save_res_name) + f"_{layer}"

# Load dataset
dataset = load_dataset("json", data_files="../data/correct_hal_pairs.jsonl", split="train")
train_test_split = dataset.train_test_split(test_size=0.2)
train_ds = train_test_split["train"]
test_ds = train_test_split["test"]
print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

# Load Qwen-VL model and processor
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    trust_remote_code=True
).eval()

hid_dim = model.config.hidden_size

# Construct training and testing datasets
dataset = construct_train_test_ds(
    train_ds,
    test_ds,
    model_name,
    model_path,
    model,
    tokenizer,
    image_path,
    processor,  # Use processor instead of image_processor
    layers,
    token_pos,
    device=device
)

# Save dataset
os.makedirs(ds_name, exist_ok=True)
dataset.save_to_disk(os.path.join(ds_name, model_name + f"_{layer}"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("hid_dim:", hid_dim)

from model import normalize_to_neg_one_to_one

# Prepare training data
ds_path = os.path.join(ds_name, model_name + f"_{layer}")
train_ds, _ = prepare_hal_train_test_ds(
    tokenizer,
    ds_path,
    model_name,
    model_path,
    image_path,
    model.config,
    device,
    layers,
    processor  # Use processor
)

# Load flow model
flow_model = load_flow_model(hid_dim, device=device)

# Prepare data loaders
train_loader = prepare_pair_data_loader(train_ds, layers, ds_type="train")
val_loader = None  # Can add validation set later

# Optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(flow_model.parameters(), lr=1e-4)
num_training_steps = len(train_loader) * num_epochs


def cosine_schedule_with_warmup(current_step: int):
    """Custom cosine learning rate schedule with warmup."""
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_scale, cosine_decay)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)

# Training loop
start_time = time.time()
train_losses = []

for epoch in range(num_epochs):
    flow_model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training")

    for example in train_bar:
        y_win = example[f"y_win_layer{layer}"]
        y_lose = example[f"y_lose_layer{layer}"]
        y_win, y_lose = y_win.to(device), y_lose.to(device)

        y_win = normalize_to_neg_one_to_one(y_win)
        y_lose = normalize_to_neg_one_to_one(y_lose)

        loss = flow_model(y_win, y_lose, return_loss_breakdown=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        train_bar.set_postfix(train_loss=loss.item())

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation phase (if validation loader is provided)
    if val_loader is not None:
        flow_model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", leave=False)

        with torch.no_grad():
            for example in val_bar:
                y_win = example[f"y_win_layer{layer}"]
                y_lose = example[f"y_lose_layer{layer}"]
                y_win, y_lose = y_win.to(device), y_lose.to(device)

                loss = flow_model(y_win, y_lose, return_loss_breakdown=False)
                val_loss += loss.item()
                val_bar.set_postfix(val_loss=loss.item())

        val_loss /= len(val_loader)

end_time = time.time()
print(f"Training time: {end_time - start_time}")

# Save the trained model
torch.save(flow_model.state_dict(), save_model_path)