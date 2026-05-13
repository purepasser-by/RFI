import argparse
import torch
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image

import re

from transformers import set_seed, AutoConfig

from copy import deepcopy
from model import RectifiedFlow, LinearUNet
from wrapper import Wrapper
from _utils import prepare_hal_train_test_ds

def recorder(out):
    word_list = re.split(r'[^\w]+', out.lower())
    if "yes" in word_list:
        return "Yes"
    else:
        return "No"
    


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
    

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device="cuda:0")

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.flow:
        hid_dim = model.config.hidden_size
        layer = args.flow_layer
        layers = [layer]
        image_path = args.flow_image_path
        ds_path = args.dataset_path
        train_ds, _ = prepare_hal_train_test_ds(tokenizer, ds_path, model_name, image_path, model.config, model.device, layers, image_processor)
        ## flow model
        unet = LinearUNet(
            hid_dim=hid_dim,
            depth=4,
            feature_scale=0.5,
            time_embedding_dim=128,
        ).to(model.device)
        flow_model = RectifiedFlow(unet, data_shape=(hid_dim,))
        flow_model.load_state_dict(torch.load(args.save_model_path, map_location=model.device))
        flow_model.eval()
        wrapper = Wrapper

    if args.flow:  
        layer = args.flow_layer
        hs_mat = torch.cat([train_ds[i][f"y_win_layer{layer}"] for i in range(len(train_ds))], dim=0)
        _, _, v = torch.svd(hs_mat)
    
        original_layer = deepcopy(model.model.layers[layer])
    

    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs + " Answer with yes or no.") 
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(0)
        image_files = [image_file]
        images = load_images(image_files, args.image_folder)
        images_tensor = process_images(
            images,
            image_processor, # clip
            model.config
        ).to(model.device, dtype=torch.float16)
            
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        
        if args.flow:  
            with torch.no_grad():
                outputs = model(
                    input_ids = input_ids,
                    images= images_tensor,
                    output_hidden_states=True
                )
            hs = outputs.hidden_states[layer][:, -1, :]
            hs_flow = flow_model.sample(hidden_states=hs)
            model.model.layers[layer] = wrapper(model.model.layers[layer], hs_flow[0], v.to(model.device), k=args.k, alpha=args.alpha)

        output_dict = model.generate(
            input_ids,
            images=images_tensor,
            max_new_tokens=3,
            output_hidden_states=True,
            return_dict_in_generate=True,
            stopping_criteria=[stopping_criteria]
        )

        if args.flow:
            model.model.layers[layer] = original_layer

        output_ids = output_dict.sequences
        outputs = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()


        ans_file.write(json.dumps(
            {
                "question_id": idx,
                "prompt": cur_prompt,
                "answer": recorder(outputs),
                "model_id": model_name,
                "image": image_file,
                "metadata": {
                } 
            }
        ) + "\n")

        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=".../llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=".../val2014")
    parser.add_argument("--question-file", type=str, default=".../aokvqa_pope_seem_random.txt")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--answers-file", type=str, default="./llava_flow_0_answers.json")
    parser.add_argument("--flow_layer", type=int, default=15)
    parser.add_argument("--flow", action="store_true", default=True, help="whether to use flow")
    parser.add_argument("--alpha", type=float, default=3)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--dataset_path", type=str, default=".../rfi/llava_15")
    parser.add_argument("--flow_image_path", type=str, default=".../train2014")
    parser.add_argument("--save_model_path", type=str, default=".../Flow_llava_epoch25_15.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    eval_model(args)

