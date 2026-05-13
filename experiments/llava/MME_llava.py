import argparse
import torch
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO
import glob

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
from transformers import set_seed
from copy import deepcopy
from model import RectifiedFlow, LinearUNet
from wrapper import Wrapper
from _utils import prepare_hal_train_test_ds


def recorder(out):
    """
    Extract words and return 'Yes' if 'yes' appears in the output, otherwise 'No'.
    Case-insensitive and handles punctuation.
    """
    word_list = re.split(r'[^\w]+', out.lower())
    return "Yes" if "yes" in word_list else "No"


def load_image(image_file, prefix_path):
    """Load image from URL or local path."""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(os.path.join(prefix_path, image_file)).convert("RGB")
    return image


def load_images(image_files, prefix_path):
    """Load multiple images from a list of file names."""
    out = []
    for image_file in image_files:
        image = load_image(image_file, prefix_path)
        out.append(image)
    return out


def eval_model(args):
    """Evaluate the LLaVA model with optional flow-based intervention."""
    # Initialize model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device="cuda:0"
    )

    # Load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Initialize flow model if enabled
    if args.flow:
        hid_dim = model.config.hidden_size
        layer = args.flow_layer
        layers = [layer]
        image_path = args.flow_image_path
        ds_path = args.dataset_path

        # Prepare training dataset to extract SVD components
        train_ds, _ = prepare_hal_train_test_ds(
            tokenizer, ds_path, model_name, image_path, model.config,
            model.device, layers, image_processor
        )

        # Build and load flow model
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

        # Compute SVD basis from training hidden states
        hs_mat = torch.cat([train_ds[i][f"y_win_layer{layer}"] for i in range(len(train_ds))], dim=0)
        _, _, v = torch.svd(hs_mat.float())

        # Save original layer for restoration
        original_layer = deepcopy(model.model.layers[layer])

    # Process each question
    for line in tqdm(questions, desc="Evaluating"):
        idx = line["question_id"]
        image_file = line["question_id"]
        qs = line["question"]
        answer_gt = line["answer"]
        cur_prompt = qs

        # Format prompt with image token
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # Select conversation template
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

        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(0)

        # Load and process image
        images = load_images([image_file], args.image_folder)
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        # Define stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        # Apply flow-based intervention if enabled
        if args.flow:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    output_hidden_states=True
                )
            hs = outputs.hidden_states[layer][:, -1, :]  # Last token hidden state
            hs_flow = flow_model.sample(hidden_states=hs, steps=args.k)
            model.model.layers[layer] = wrapper(
                model.model.layers[layer],
                hs_flow[0],
                v.to(model.device),
                k=args.k,
                alpha=args.alpha
            )

        # Generate response
        output_dict = model.generate(
            input_ids,
            images=images_tensor,
            max_new_tokens=3,
            output_hidden_states=True,
            return_dict_in_generate=True,
            stopping_criteria=[stopping_criteria]
        )

        # Restore original layer
        if args.flow:
            model.model.layers[layer] = original_layer

        # Decode output
        output_ids = output_dict.sequences
        outputs = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        print(f"ID: {idx}, Pred: {outputs}, GT: {answer_gt}")

        # Save result
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "answer": recorder(outputs),
            "model_id": model_name,
            "image": image_file,
            "metadata": {

            }
        }) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLaVA model with optional Rectified Flow intervention.")

    parser.add_argument("--model-path", type=str, default=".../llava-v1.5-7b",
                        help="Path to the pretrained LLaVA model.")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model name (e.g., lmsys/vicuna-7b-v1.5).")
    parser.add_argument("--image-folder", type=str, default=".../val2014",
                        help="Directory containing images.")
    parser.add_argument("--question-file", type=str,
                        default=".../aokvqa_pope_seem_adversarial.json",
                        help="Path to the question file (JSON).")
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="Conversation mode (e.g., llava_v1).")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks to split the dataset into.")
    parser.add_argument("--chunk-idx", type=int, default=0,
                        help="Index of the chunk to process.")
    parser.add_argument("--temperature", type=float, default=-1,
                        help="Sampling temperature. -1 means greedy decoding.")
    parser.add_argument("--answers-file", type=str, default="./llava_flow_0_answers.json",
                        help="Output path for saving answers.")
    parser.add_argument("--flow_layer", type=int, default=15,
                        help="Transformer layer index to apply intervention (0-based).")
    parser.add_argument("--flow", action="store_true", default=True,
                        help="Whether to enable flow-based intervention.")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Scaling factor for flow adjustment.")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of sampling steps in the flow model.")
    parser.add_argument("--dataset_path", type=str, default=".../rfi/llava_15",
                        help="Path to the dataset for extracting training hidden states.")
    parser.add_argument("--flow_image_path", type=str, default=".../train2014",
                        help="Image folder used during flow model training.")
    parser.add_argument("--save_model_path", type=str,
                        default=".../Flow_llava_epoch25_15.pth",
                        help="Path to the trained flow model checkpoint.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()
    set_seed(args.seed)

    # Define directories
    text_dir = ".../MME_dataset/Text/"
    image_base_dir = ".../MME_dataset/extracted_images/"
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)

    # Get all task files
    txt_files = glob.glob(os.path.join(text_dir, "*.txt"))

    for txt_path in txt_files:
        category_name = os.path.basename(txt_path).replace(".txt", "")
        image_folder = image_base_dir

        if not os.path.exists(image_folder):
            print(f"[Warning] Image folder {image_folder} does not exist. Skipping.")
            continue

        # Update arguments for current task
        args.question_file = txt_path
        args.image_folder = image_folder
        args.answers_file = os.path.join(result_dir, f"{category_name}_answers.json")

        print(f"Processing task: {category_name}")
        eval_model(args)