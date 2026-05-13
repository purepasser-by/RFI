import argparse
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import RectifiedFlow, LinearUNet
from wrapper import Wrapper
from datasets import load_from_disk
import re
from copy import deepcopy
import numpy as np
import glob
import requests
from io import BytesIO


def recorder(out):
    """Extract words and return 'Yes' if 'yes' is in the output, else 'No'."""
    word_list = re.split(r'[^\w]+', out.lower())
    return "Yes" if "yes" in word_list else "No"


def load_image(image_file, prefix_path):
    """Load image from local path or URL."""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(os.path.join(prefix_path, image_file)).convert("RGB")
    return image


def prepare_svd_components(dataset_path, layer, device):
    """Extract SVD components from the training dataset (corrected version)."""
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]

    # Process all items into 2D tensors (num_samples, hid_dim)
    hs_list = []
    for i in range(len(train_ds)):
        item = train_ds[i][f"y_win_layer{layer}"]

        if isinstance(item, list):
            arr = np.array(item)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            tensor_item = torch.tensor(arr)
        else:
            tensor_item = item.clone().detach()

        if tensor_item.dim() == 2 and tensor_item.size(0) == 1:
            tensor_item = tensor_item.squeeze(0)

        hs_list.append(tensor_item)

    hs_mat = torch.stack(hs_list).to(device)

    if hs_mat.dim() == 3:
        hs_mat = hs_mat.view(-1, hs_mat.shape[-1])

    print(f"Corrected HS mat shape: {hs_mat.shape}")

    _, _, v = torch.svd(hs_mat.float())
    print(f"SVD v shape: {v.shape}")
    return v


def eval_model(args):
    """Evaluate the model with optional flow-based intervention."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    # Load questions
    questions = []
    question_file = args.question_file
    if question_file.endswith('.txt'):
        with open(question_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in {question_file}: {e}")
    else:  # Assume JSON file
        questions = json.load(open(question_file, "r"))

    # Create output directory
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    # Initialize Flow model if enabled
    if args.flow:
        hid_dim = model.config.hidden_size

        unet = LinearUNet(
            hid_dim=hid_dim,
            depth=4,
            feature_scale=0.5,
            time_embedding_dim=128
        ).to(model.device)

        flow_model = RectifiedFlow(unet, data_shape=(hid_dim,))
        flow_model.load_state_dict(torch.load(args.save_model_path, map_location=model.device))
        flow_model.eval()

        # Compute SVD components from training data
        v = prepare_svd_components(args.dataset_path, args.flow_layer, model.device)

    # Open output file for writing
    with open(args.answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(questions, desc="Processing questions"):
            data = {
                "question_id": line["question_id"],
                "prompt": line["question"],
                "image": line.get("image", None),
                "metadata": {
                    "flow_layer": args.flow_layer if args.flow else None,
                    "alpha": args.alpha if args.flow else None,
                    "k": args.k if args.flow else None
                }
            }

            try:
                # Construct image path
                image_id = line["question_id"]
                image_path = os.path.join(args.image_folder, image_id)

                # Build multimodal input using tokenizer format
                query = tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': data["prompt"]}
                ])

                # Apply Flow-based intervention if enabled
                if args.flow:
                    inputs = tokenizer(query, return_tensors='pt').to(model.device)
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        hs = outputs.hidden_states[args.flow_layer + 1][0, -1, :]  # Last token

                        # Sample from flow model
                        hs_flow = flow_model.sample(
                            hidden_states=hs.unsqueeze(0),
                            steps=args.k
                        ).squeeze(0)

                    # Wrap the target transformer layer
                    wrapped_layer = Wrapper(
                        model.transformer.h[args.flow_layer],
                        hs_flow,
                        v,
                        k=args.k,
                        alpha=args.alpha
                    ).to(model.device)

                    # Replace layer temporarily
                    original_layer = deepcopy(model.transformer.h[args.flow_layer])
                    model.transformer.h[args.flow_layer] = wrapped_layer

                # Generate response using chat interface
                response, history = model.chat(tokenizer, query=query, history=None)
                data["answer"] = response.strip()

                if args.flow:
                    # Restore original layer
                    model.transformer.h[args.flow_layer] = original_layer

                print("Answer:", data["answer"])

            except Exception as e:
                print(f"Error processing {data['question_id']}: {str(e)}")
                data["answer"] = "Error"

            # Write result to file
            ans_file.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Qwen model with optional Rectified Flow intervention.")
    parser.add_argument("--model-path", type=str, default="/gz-data/QWEN_model",
                        help="Path to the pretrained Qwen model.")
    parser.add_argument("--image-folder", type=str, default="/gz-data/MME_dataset/extracted_images",
                        help="Directory containing images.")
    parser.add_argument("--question-file", type=str, default="/gz-data/MME_dataset/Text",
                        help="Path to the question file (txt or json), or directory containing txt files.")
    parser.add_argument("--answers-file", type=str, default="./result/qwen_flow_answers.json",
                        help="Output path for saving answers.")

    # Flow intervention parameters
    parser.add_argument("--flow", action="store_true", default=True,
                        help="Whether to enable flow-based intervention.")
    parser.add_argument("--flow-layer", type=int, default=15,
                        help="Transformer layer index to intervene on (0-based).")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Alpha parameter for directional adjustment.")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of sampling steps in flow model.")
    parser.add_argument("--dataset_path", type=str, default="/gz-data/QWEN/rfi/qwen_15",
                        help="Path to the dataset for extracting SVD components.")
    parser.add_argument("--save_model_path", type=str,
                        default="/gz-data/QWEN/qwen_hal_results/Flow_qwen_epoch40_15.pth",
                        help="Path to the trained flow model checkpoint.")

    args = parser.parse_args()

    # Validate input paths
    assert os.path.exists(args.dataset_path), f"Dataset path not found: {args.dataset_path}"
    assert os.path.exists(args.save_model_path), f"Flow model checkpoint not found: {args.save_model_path}"

    print("Alpha:", args.alpha)
    print("Flow enabled:", args.flow)
    print("Flow model path:", args.save_model_path)
    print("Question file:", args.question_file)

    # Support processing multiple files in a directory
    if os.path.isdir(args.question_file):
        txt_files = glob.glob(os.path.join(args.question_file, "*.txt"))
        print(f"Found {len(txt_files)} task files in directory.")
        for txt_path in txt_files:
            category_name = os.path.basename(txt_path).replace(".txt", "")
            args.question_file = txt_path
            args.answers_file = f"./result/qwen_flow_{category_name}_answers.json"
            eval_model(args)
    else:
        eval_model(args)