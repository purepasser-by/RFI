from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from typing import List
from datasets import load_from_disk
import requests
from model import LinearUNet,RectifiedFlow
import os
from functools import partial



def construct_train_test_ds(
        train_ds,
        test_ds, 
        model_name, 
        model_path,
        model, 
        tokenizer, 
        image_path,
        image_processor, 
        layers, 
        token_pos, 
        device
    ):
    train_win, train_lose, train_tmp = extract_hq_avg_minus_tqa(train_ds, model_name, model_path, model, tokenizer, image_path, image_processor, layers, token_pos, device)
    test_win, test_lose, test_tmp = extract_hq_avg_minus_tqa(test_ds, model_name, model_path, model, tokenizer,  image_path, image_processor, layers, token_pos, device)
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

# Origin
def extract_hq_avg_minus_tqa(
    dataset, 
    model_name,
    model_path,
    model, 
    tokenizer,
    image_path,
    image_processor, 
    layers, 
    token_pos, 
    device
):
    from PIL import Image
    import os
    """extract query last token hidden states as y_lose, and (correct answer average hidden states - incorrect answer average hidden states) as y_win."""
    print("hello", flush=True)
    print("model_name:", model_name)
    template_q = [] # chat template with only the question
    y_win_set, y_lose_set = [], [] # y_lose -- hq, y_win -- hc - hi

    
    for data in tqdm(dataset):
        image_file = data["image"]
        full_image_path = os.path.join(image_path, image_file)
        
        if "qwen" in model_name:
            query = tokenizer.from_list_format([
                {'image': full_image_path},
                {'text': data["question"]}  
            ])
            # print("query:", query)
            template_q.append(query)  
            inputs = tokenizer(query, return_tensors='pt').to(device)
        
        # hq (using original image)
        hq_list = []
        with torch.no_grad():
            outputs = model(**inputs.to(device), output_hidden_states=True)
            for layer in layers:
                hq = outputs.hidden_states[layer][0, -1, :]  
                
                hq_list.append(hq.cpu())
                
        
        hqs = torch.stack(hq_list)  #* (num_layers, hid_dim)  
    
        y_lose_set.append(hqs.unsqueeze(0))  # (1, 1, 4096)

        if "qwen" in model_name:
            c_query = tokenizer.from_list_format([
                {'image': full_image_path},
                {'text': data["value"]}  
            ])
            c_inputs = tokenizer(c_query, return_tensors='pt').to(device)
        

            inc_query = tokenizer.from_list_format([
                {'image': full_image_path},
                {'text': data["h_value"]} 
            ])
            inc_inputs = tokenizer(inc_query, return_tensors='pt').to(device)
            
        
        with torch.no_grad():
            if "qwen" in model_name:
                c_outputs = model(**c_inputs.to(device), output_hidden_states=True)
                inc_outputs = model(**inc_inputs.to(device), output_hidden_states=True)
            else:
                c_outputs = model(**tokenized_c_chat.to(device), output_hidden_states=True)
                inc_outputs = model(**tokenzied_inc_chat.to(device), output_hidden_states=True)
        
        avg_hc_minus_hi = []
        for layer in layers:
            if token_pos == "qa_avg":
                hc = c_outputs.hidden_states[layer][0, :, :].mean(dim=0)  
                hi = inc_outputs.hidden_states[layer][0, :, :].mean(dim=0)
            elif token_pos == "ans_avg":
                num_image_tokens = 256  
                hc = c_outputs.hidden_states[layer][0, num_image_tokens:, :].mean(dim=0)
                hi = inc_outputs.hidden_states[layer][0, num_image_tokens:, :].mean(dim=0)
            elif token_pos == "last":
                hc = c_outputs.hidden_states[layer][0, -1, :]  
                hi = inc_outputs.hidden_states[layer][0, -1, :]
            else:
                raise ValueError("Invalid setting.")
                
            avg_hc_minus_hi.append((hc - hi).cpu())
            
        avgs = torch.stack(avg_hc_minus_hi) 
        y_win_set.append(avgs.unsqueeze(0))  

    return y_win_set, y_lose_set, template_q


def prepare_hal_train_test_ds(tokenizer, ds_name, model_name, model_path, image_path, config, device="cuda:0", layers:List[int]=[20], image_processor=None):
    ds = load_from_disk(ds_name)
    train_ds = ds["train"]
    test_ds = ds["test"] 
    
    def prepare_inputs(example, tokenizer):
        if 'qwen' in model_name:
            image_file = example["image"]
            full_image_path = os.path.join(image_path, image_file)
            
            query = tokenizer.from_list_format([
                {'image': full_image_path},
                {'text': example["template_q"]}
            ])
            inputs = tokenizer(query, return_tensors='pt').to(device)
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "images_tensor": inputs.get("pixel_values", None)
            }
        else:
            return tokenizer(example["template_q"], return_tensors="pt", add_special_tokens=False)  
    
    attr_list = [f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers]
    train_ds.set_format(type='torch', columns=attr_list)
    
    train_ds = train_ds.map(partial(prepare_inputs, tokenizer=tokenizer))
    test_ds = test_ds.map(partial(prepare_inputs, tokenizer=tokenizer))
    
    if "qwen" in model_name:
        test_ds.set_format(type='torch', columns=attr_list + ['question', 'template_q', 'input_ids', 'images_tensor', 'correct_answers', 'incorrect_answers'])
    else:
        test_ds.set_format(type='torch', columns=attr_list + ['question', 'template_q', 'input_ids', 'correct_answers', 'incorrect_answers'])
    
    return train_ds, test_ds
    

def prepare_pair_data_loader(ds, layers:List[int], ds_type:str="train", batch_size=136, seed=42):
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

    generator = torch.Generator()
    generator.manual_seed(seed) 
    
    if ds_type == "train":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator = generator)
    elif ds_type == "validate":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif ds_type == "test":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Invalid dataset type.")
    
    return data_loader

def load_flow_model(hid_dim, device='cuda:0', model_path = None,):
    unet = LinearUNet(
        hid_dim=hid_dim,
        depth=4,
        feature_scale=0.5,
        time_embedding_dim=128,
    ).to(device)
    flow_model = RectifiedFlow(unet, data_shape=(hid_dim,))

    if model_path is not None:
        flow_model.load_state_dict(torch.load(model_path, map_location=device))

    return flow_model