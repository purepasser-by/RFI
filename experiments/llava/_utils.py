from tqdm import tqdm
import torch
import math
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import List
from datasets import load_from_disk
import requests
from io import BytesIO
import sys
from pathlib import Path
import os
from PIL import Image, ImageFilter, ImageDraw
import random
import numpy as np

sys.path.append(str(Path(__file__).parent))

from model import LinearUNet, RectifiedFlow, MeanFlow

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
INSTRUCT = "Answer the question concisely. Q: {} A:"

def get_chat(model_name: str, question: str):
    """chat template for LLMs"""
    prompt = INSTRUCT.format(question)

    if "qwen" in model_name:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    elif "llava" in model_name:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs) 
        conv.append_message(conv.roles[1], None)
        # chat = conv.get_prompt()
        return conv        

    return chat

def random_mask(
    image, 
    num_patches=30,         
    patch_size_ratio=0.1,    
    fill_color=(0, 0, 0)     
):

    width, height = image.size
    patch_size = int(min(width, height) * patch_size_ratio)  
    result = image.copy()
    draw = ImageDraw.Draw(result)
    
    for _ in range(num_patches):
        x = random.randint(0, width - patch_size)
        y = random.randint(0, height - patch_size)
        draw.rectangle([x, y, x + patch_size, y + patch_size], fill=fill_color)
    
    return result

def get_avg_random_mask_image(image):
    masked_images = []
    for _ in range(3):
        masked_image = random_mask(image)
        masked_images.append(masked_image)

    np_images = np.stack([np.array(img) for img in masked_images])
    
    avg_image = np.mean(np_images, axis=0).astype(np.uint8)
    
    return Image.fromarray(avg_image)

def load_image(image_file, prefix_path):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(prefix_path + "/" + image_file).convert("RGB")
    return image


def load_images(image_files, prefix_path):
    out = []
    for image_file in image_files:
        image = load_image(image_file, prefix_path)
        out.append(image)
    return out


def construct_train_test_ds(
        train_ds,
        test_ds, 
        model_name, 
        model, 
        tokenizer, 
        image_path,
        image_processor, 
        layers, 
        token_pos, 
        device
    ):
    train_win, train_lose, train_tmp = extract_hq_avg_minus_tqa(train_ds, model_name, model, tokenizer, image_path, image_processor, layers, token_pos, device)
    test_win, test_lose, test_tmp = extract_hq_avg_minus_tqa(test_ds, model_name, model, tokenizer,  image_path, image_processor, layers, token_pos, device)
    train_data_dict = {
        'correct_answers': [x["value"] for x in train_ds],
        'incorrect_answers': [x["h_value"] for x in train_ds],
        'question': [x["question"] for x in train_ds],
        'image': [x["image"] for x in train_ds],
        'template_q': train_tmp,
    }

    test_data_dict = {
        'correct_answers': [x["value"] for x in test_ds],
        'incorrect_answers': [x["h_value"] for x in test_ds],
        'question': [x["question"] for x in test_ds],
        'image': [x["image"] for x in test_ds],
        'template_q': test_tmp,
    }
        
    for j, layer in enumerate(layers):
        train_data_dict[f"y_win_layer{layer}"] = [train_win[i][:, j, :] for i in range(len(train_win))]
        train_data_dict[f"y_lose_layer{layer}"] = [train_lose[k][:, j, :] for k in range(len(train_lose))]

    for j, layer in enumerate(layers):
        test_data_dict[f"y_win_layer{layer}"] = [test_win[i][:, j, :] for i in range(len(test_win))]
        test_data_dict[f"y_lose_layer{layer}"] = [test_lose[k][:, j, :] for k in range(len(test_lose))]
        
    from datasets import Dataset, DatasetDict
    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    return dataset

def get_y_win_lose(chat, 
    image_file, image_path, image_processor, 
    p_ans, n_ans, 
    model_name, model, tokenizer, 
    layer, 
    device = "cuda:0", token_pos = "ans_avg"):
    
    # Get 
    # 1. y_lose( Original Question Hiddenstates )
    # 2. y_win ( Direction = Positive - Negative ) (via Append P/N Answers)

    if "llava" in model_name:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token, process_images
        conv = chat.copy() # conv
        chat = chat.get_prompt() 
        template_q = chat

        input_ids = tokenizer_image_token(chat, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        image_files = [image_file]
        images = load_images(image_files, image_path)
        images_tensor = process_images(
            images,
            image_processor, # clip
            model.config
        ).to(model.device, dtype=torch.float16)


        random_avg_mask_image = get_avg_random_mask_image(images[0])
        random_avg_mask_images_tensors = process_images(
            [random_avg_mask_image],
            image_processor, # clip
            model.config
        ).to(model.device, dtype=torch.float16)
        

        question_token_length = input_ids.shape[1]

    else:
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        template_q = formatted_chat
        tokenized_format_chat = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        question_token_length = tokenized_format_chat["input_ids"].shape[1]
    
    # hq
    with torch.no_grad():
        if "llava" in model_name:
            outputs = model(
                input_ids,
                images=images_tensor,
                output_hidden_states=True
            )
        else:
            outputs = model(**tokenized_format_chat.to(device), output_hidden_states=True)

        # last token hidden states
        hq = outputs.hidden_states[layer][0, -1, :]
        hq = [hq.cpu()]
    
    hqs = torch.stack(hq)  
    
    y_lose = hqs.unsqueeze(0)

    
    if "llava" in model_name:
        inc_conv = conv.copy()
        c_conv = conv.copy()

        inc_conv.append_message(conv.roles[1], n_ans)
        c_conv.append_message(conv.roles[1], p_ans)

        inc_input_ids = tokenizer_image_token(inc_conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        c_input_ids = tokenizer_image_token(c_conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    else:

        inc_chat = chat + [{"role": "assistant", "content": n_ans}]
        c_chat = chat + [{"role": "assistant", "content": p_ans}]
    
        inc_formatted_chat = tokenizer.apply_chat_template(inc_chat, tokenize=False, add_generation_prompt=False)
        c_formatted_chat = tokenizer.apply_chat_template(c_chat, tokenize=False, add_generation_prompt=False)

        tokenzied_inc_chat = tokenizer(inc_formatted_chat, return_tensors="pt", add_special_tokens=False)
        tokenized_c_chat = tokenizer(c_formatted_chat, return_tensors="pt", add_special_tokens=False)
    
    with torch.no_grad():
        if "llava" in model_name:
            c_outputs = model(
                c_input_ids,
                images=images_tensor,
                output_hidden_states=True
            )
            inc_outputs = model(
                inc_input_ids,
                # images=images_tensor,
                images = random_avg_mask_images_tensors,
                output_hidden_states=True
            )
        else:
            c_outputs = model(**tokenized_c_chat.to(device), output_hidden_states=True)
            inc_outputs = model(**tokenzied_inc_chat.to(device), output_hidden_states=True)
    
    
    if token_pos == "qa_avg":
        # average over all Q and A tokens
        hc = c_outputs.hidden_states[layer][0, :, :].mean(dim=0)
        hi = inc_outputs.hidden_states[layer][0, :, :].mean(dim=0)
    elif token_pos == "ans_avg":
        # average over all answer tokens
        hc = c_outputs.hidden_states[layer][0, question_token_length:, :].mean(dim=0)
        hi = inc_outputs.hidden_states[layer][0, question_token_length:, :].mean(dim=0)
    elif token_pos == "last":
        # last token
        hc = c_outputs.hidden_states[layer][0, -1, :]
        hi = inc_outputs.hidden_states[layer][0, -1, :]
    else:
        raise ValueError("Invalid setting.")

    direction = [(hc - hi).cpu()] 
    
    y_win = torch.stack(direction).unsqueeze(0)

    return y_win, y_lose, template_q



def extract_hq_avg_minus_tqa(
        dataset, 
        model_name, 
        model, 
        tokenizer,
        image_path,
        image_processor, 
        layers, 
        token_pos, 
        device
    ):
    """extract query last token hidden states as y_lose, and (correct answer average hidden states - incorrect answer average hidden states) as y_win."""
    
    template_q = [] # chat template with only the question
    y_win_set, y_lose_set = [], [] # y_lose -- hq, y_win -- hc - hi
    
    for data in tqdm(dataset):
        # General : describe in details
        qs = get_chat(model_name, data["question"])
        image_file = data["image"]  
        p_ans = data["value"]
        n_ans = data["h_value"]
        y_win, y_lose, tq = get_y_win_lose(
            qs, 
            image_file, image_path, image_processor,
            p_ans, n_ans, 
            model_name, model, 
            tokenizer, layers[0], 
            device,
            token_pos
        )
            
        y_win_set.append(y_win)
        y_lose_set.append(y_lose)
        template_q.append(tq)


    return y_win_set, y_lose_set, template_q


def prepare_hal_train_test_ds(tokenizer, ds_name, model_name, image_path, config, device="cuda:1", layers:List[int]=[20], image_processor=None):
    ds = load_from_disk(ds_name)
    train_ds = ds["train"]
    test_ds = ds["test"] 
    def prepare_inputs(example):
        if 'llava' in model_name:
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import tokenizer_image_token, process_images
            prompt = example["template_q"]
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device)
            
            image_file = example["image"]
            image_files = [image_file]
            images = load_images(image_files, image_path)
            images_tensor = process_images(
                images,
                image_processor, # clip
                config # model.config
            ).to(device, dtype=torch.float16)    
            
            return {"input_ids": input_ids.unsqueeze(0), "images_tensor": images_tensor}
        else:
            return tokenizer(example["template_q"], return_tensors="pt", add_special_tokens=False)  
    
    attr_list = [f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers]
    train_ds.set_format(type='torch', columns=attr_list)
    
    test_ds = test_ds.map(prepare_inputs)

    if "llava" in model_name:
        test_ds.set_format(type='torch', columns=attr_list + ['question', 'template_q', 'input_ids', 'images_tensor', 'correct_answers', 'incorrect_answers'])
    else:
        test_ds.set_format(type='torch', columns=attr_list + ['question', 'template_q', 'input_ids', 'correct_answers', 'incorrect_answers'])
    
    return train_ds, test_ds


def prepare_pair_data_loader(ds, layers:List[int], ds_type:str="train", batch_size=136):
    y_win_set = [[] for _ in range(len(layers))]
    y_lose_set = [[] for _ in range(len(layers))]
    for example in ds:
        for idx, layer in enumerate(layers):
            y_win = example[f"y_win_layer{layer}"]
            y_lose = example[f"y_lose_layer{layer}"]

            y_win_pair = y_win.repeat(1, y_lose.shape[0]).reshape(-1, y_win.shape[1])
            y_lose_pair = y_lose.tile((y_win.shape[0], 1))
            y_win_set[idx].append(y_win_pair)
            y_lose_set[idx].append(y_lose_pair)
        
    y_win_set = [torch.cat(y_win_per_layer) for y_win_per_layer in y_win_set]
    y_lose_set = [torch.cat(y_lose_per_layer) for y_lose_per_layer in y_lose_set]
        
    data_dict = {
        **{f"y_win_layer{layers[idx]}": y_win for idx, y_win in enumerate(y_win_set)},
        **{f"y_lose_layer{layers[idx]}": y_lose for idx, y_lose in enumerate(y_lose_set)}
    }
    dataset = Dataset.from_dict(data_dict)
    attr_list = [f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers]
    dataset.set_format(type='torch', columns=attr_list)
    
    if ds_type == "train":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif ds_type == "validate":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif ds_type == "test":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Invalid dataset type.")
    
    return data_loader

def load_flow_model(hid_dim, device='cuda:0', model_path = None, flow_type='rectified'):

    unet = LinearUNet(
        hid_dim=hid_dim,
        depth=4,
        feature_scale=0.5,
        time_embedding_dim=128,
    ).to(device)
    # flow_model = RectifiedFlow(unet, data_shape=(hid_dim,))
    if flow_type == 'mean':
        flow_model = MeanFlow(unet, data_shape=(hid_dim,))
    else:
        flow_model = RectifiedFlow(unet, data_shape=(hid_dim,))

    if model_path is not None:
        flow_model.load_state_dict(torch.load(model_path, map_location=device))

    return flow_model