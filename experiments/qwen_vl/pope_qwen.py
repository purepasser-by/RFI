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


def recorder(out):
    """
    Extract words from output and check if 'yes' is present.
    Returns "Yes" if 'yes' is found, otherwise "No".
    """
    word_list = re.split(r'[^\w]+', out.lower())
    return "Yes" if "yes" in word_list else "No"


def prepare_svd_components(dataset_path, layer, device):
    """
    Load training dataset and compute SVD components from hidden states.
    This function ensures the hidden states are properly reshaped before SVD.
    
    Args:
        dataset_path (str): Path to the saved dataset (Hugging Face Dataset format).
        layer (int): Layer index for extracting hidden states (e.g., y_win_layer{layer}).
        device (torch.device): Device to load tensors on (e.g., cuda).
    
    Returns:
        v (torch.Tensor): Right singular vectors from SVD, shape [hid_dim, hid_dim].
    """
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]

    hs_list = []
    for i in range(len(train_ds)):
        item = train_ds[i][f"y_win_layer{layer}"]

        if isinstance(item, list):
            # Convert list to tensor
            arr = np.array(item)
            if arr.ndim == 2 and arr.shape[0] == 1:
                # Squeeze redundant batch dimension [1, hid_dim] -> [hid_dim]
                arr = arr.squeeze(0)
            tensor_item = torch.tensor(arr)
        else:
            # Handle tensor data
            tensor_item = item.clone().detach()

        # Ensure 1D vector: squeeze if shape is [1, hid_dim]
        if tensor_item.dim() == 2 and tensor_item.size(0) == 1:
            tensor_item = tensor_item.squeeze(0)

        hs_list.append(tensor_item)

    # Stack all hidden states into a matrix: [num_samples, hid_dim]
    hs_mat = torch.stack(hs_list).to(device)

    # Flatten higher-dimensional tensors (e.g., [a, b, c] -> [a*b, c])
    if hs_mat.dim() == 3:
        hs_mat = hs_mat.view(-1, hs_mat.shape[-1])

    print(f"Corrected HS mat shape: {hs_mat.shape}")  # Expected: [80, 4096]

    # Perform SVD: hs_mat = U @ S @ V^T
    _, _, v = torch.svd(hs_mat.float())
    print(f"SVD v shape: {v.shape}")  # Expected: [4096, 4096]
    return v


def eval_model(args):
    """
    Evaluate the model on a question-answering task with optional Rectified Flow intervention.
    
    Args:
        args: Parsed command-line arguments.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    # Load questions from file
    questions = [json.loads(q) for q in open(args.question_file, "r")]
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    # Initialize Flow model if enabled
    if args.flow:
        hid_dim = model.config.hidden_size

        # Define UNet for Rectified Flow
        unet = LinearUNet(
            hid_dim=hid_dim,
            depth=4,
            feature_scale=0.5,
            time_embedding_dim=128
        ).to(model.device)

        # Initialize Rectified Flow model
        flow_model = RectifiedFlow(unet, data_shape=(hid_dim,))
        flow_model.load_state_dict(torch.load(args.save_model_path, map_location=model.device))
        flow_model.eval()

        # Compute SVD components from training data for directional projection
        v = prepare_svd_components(args.dataset_path, args.flow_layer, model.device)

    # Open output file for writing answers
    with open(args.answers_file, "w") as ans_file:
        for line in tqdm(questions):
            data = {
                "question_id": line["question_id"],
                "prompt": line["text"],
                "image": line["image"],
                "metadata": {
                    "flow_layer": args.flow_layer if args.flow else None,
                    "alpha": args.alpha if args.flow else None,
                    "k": args.k if args.flow else None
                }
            }

            try:
                # Construct multimodal input with image and text
                image_path = os.path.join(args.image_folder, line["image"])
                query = tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': line["text"] + " Please only answer yes or no. "}
                ])

                # Apply Flow-based intervention if enabled
                if args.flow:
                    # Tokenize input and get hidden states
                    inputs = tokenizer(query, return_tensors='pt').to(model.device)
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        # Extract hidden state at specified layer and last token position
                        hs = outputs.hidden_states[args.flow_layer + 1][0, -1, :]  # [hid_dim]

                        # Generate refined hidden state using Rectified Flow
                        hs_flow = flow_model.sample(
                            hidden_states=hs.unsqueeze(0),  # Add batch dim
                            steps=args.k
                        ).squeeze(0)  # Remove batch dim

                    # Wrap the target transformer layer with intervention
                    wrapped_layer = Wrapper(
                        model.transformer.h[args.flow_layer],
                        hs_flow,
                        v,
                        k=args.k,
                        alpha=args.alpha
                    ).to(model.device)

                    # Backup original layer and replace with wrapped version
                    original_layer = deepcopy(model.transformer.h[args.flow_layer])
                    model.transformer.h[args.flow_layer] = wrapped_layer

                # Generate response using model's chat interface
                response, history = model.chat(tokenizer, query=query, history=None)
                data["answer"] = response

                # Restore original transformer layer after generation
                if args.flow:
                    model.transformer.h[args.flow_layer] = original_layer

                print("answer:", data["answer"])

            except Exception as e:
                print(f"Error processing {line['question_id']}: {str(e)}")
                data["answer"] = "Error"

            # Write result to output file
            ans_file.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and data paths
    parser.add_argument("--model-path", type=str, default="/gz-data/QWEN_MODEL")
    parser.add_argument("--image-folder", type=str, default="/gz-data/val2014")
    parser.add_argument("--question-file", type=str, default="/gz-data/QWEN/COCO_POPE/coco_pope_adversarial.txt")
    parser.add_argument("--answers-file", type=str, default="./qwen_flow_answers.json")

    # Flow intervention hyperparameters
    parser.add_argument("--flow", action="store_true", default=True,
                        help="Whether to enable Rectified Flow intervention.")
    parser.add_argument("--flow-layer", type=int, default=15,
                        help="Transformer layer index to intervene on (0-based).")
    parser.add_argument("--alpha", type=float, default=2.16,
                        help="Scaling factor for the intervention direction.")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of sampling steps in the flow model.")
    parser.add_argument("--dataset_path", type=str, default="/gz-data/QWEN/rfi/qwen_15",
                        help="Path to the training dataset for SVD computation.")
    parser.add_argument("--save_model_path", type=str, default="/gz-data/QWEN/qwen_hal_results/Flow_qwen_epoch40_15.pth",
                        help="Path to the trained Rectified Flow model checkpoint.")

    args = parser.parse_args()

    # Start evaluation
    eval_model(args)