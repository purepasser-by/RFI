
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from _utils import construct_train_test_ds, prepare_hal_train_test_ds, prepare_pair_data_loader, prepare_hal_train_ds, load_flow_model
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False    
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 42
set_seed(SEED)


model_name = "llava"
model_path = ".../llava-v1.5-7b"
image_path = ".../train2014"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 25
token_pos = "last"
min_lr_scale = 0.7
num_warmup_steps = 100
layers = [15]
layer = layers[0]
k = 20
alpha = 1.5

ds_name = "rfi"

# Flow Matching Model Path
save_nn_name = f"Flow_{model_name}_epoch{num_epochs}" # save neural network for flow
save_res_name = f"Flow_{model_name}_k{k}_alpha{alpha}_epoch{num_epochs}" # save generation results for flow

res_dir = f"{model_name}_hal_results"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    
save_model_path = os.path.join(res_dir, save_nn_name) + f"_{layer}.pth" 
save_res_path = os.path.join(res_dir, save_res_name) + f"_{layer}"

# Prepare LVLM Hidden States data for Training FLOW MODEL
dataset = load_dataset("json", data_files="../data/correct_hal_pairs.jsonl", split="train")

train_test_split = dataset.train_test_split(test_size=0.2)
train_ds = train_test_split["train"]
test_ds = train_test_split["test"]
print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

# load model and tokenizer
if "llava" in model_name:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    model_path = os.path.expanduser(model_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, get_model_name_from_path(model_path), device=device)


model.eval()
hid_dim = model.config.hidden_size

dataset = construct_train_test_ds(
    train_ds,
    test_ds, 
    model_name, 
    model, 
    tokenizer,
    image_path,
    image_processor,  
    layers, 
    token_pos, 
    device=device 
)

# save dataset
os.makedirs(ds_name, exist_ok=True)
dataset.save_to_disk(os.path.join(ds_name, model_name + f"_{layer}"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


hid_dim = globals().get('hid_dim', 4096)
if 'tokenzier' not in globals():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

ds_path = os.path.join(ds_name, model_name + f"_{layer}")

train_ds, _ = prepare_hal_train_ds(tokenizer, ds_path, model_name, device, layers)

flow_model = load_flow_model(hid_dim, device = device)

train_loader = prepare_pair_data_loader(train_ds, layers, ds_type="train")

val_loader = None

optimizer = torch.optim.AdamW(flow_model.parameters(), lr=1e-4)

num_training_steps = len(train_loader) * num_epochs


def cosine_schedule_with_warmup(current_step: int):
    if current_step < num_warmup_steps:
        # Linear warm-up
        return float(current_step) / float(max(1, num_warmup_steps))
    # Cosine decay after warm-up
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_lr_scale, cosine_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)

start_time = time.time()
train_losses = []
for epoch in range(num_epochs):
    # Training phase
    flow_model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training")
    for example in train_bar:
        y_win = example[f"y_win_layer{layer}"]
        y_lose = example[f"y_lose_layer{layer}"]
        y_win, y_lose = y_win.to(device), y_lose.to(device)
            
        loss = flow_model(y_win, y_lose, return_loss_breakdown = False)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        train_bar.set_postfix(train_loss=loss.item())
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    
    # validation phase
    if val_loader is None:
        continue
    flow_model.eval()
    val_loss = 0
    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", leave=False)
    with torch.no_grad():
        for example in val_bar:
            y_win = example[f"y_win_layer{layer}"]
            y_lose = example[f"y_lose_layer{layer}"]
            print(y_win.shape, y_lose.shape)
            y_win, y_lose = y_win.to(device), y_lose.to(device)
            
            loss = flow_model(y_win, y_lose, return_loss_breakdown = False)  
            val_loss += loss.item()
            val_bar.set_postfix(val_loss=loss.item())
    
    val_loss /= len(val_loader)
    
    
end_time = time.time()
print(f"Training time: {end_time - start_time}")

# save model
torch.save(flow_model.state_dict(), save_model_path)



