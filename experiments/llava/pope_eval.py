import os
import json
from tqdm import tqdm
import pandas as pd

class Args:
    def __init__(self):
        self.gt_files = ".../aokvqa_pope_seem_random.txt"
        self.gen_files = "./llava_flow_0_answers.json"

args = Args()

with open(os.path.expanduser(args.gt_files), "r") as f:
    gt_files = []
    for line in f:
        if line.strip():
            gt_files.append(json.loads(line.strip()))

with open(os.path.expanduser(args.gen_files), "r") as f:
    gen_files = [json.loads(line) if line.strip() else None for line in f if line.strip()]

assert len(gt_files) == len(gen_files), "Length of GT and generated files must match"

true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    assert idx == gen_files[index]["question_id"], f"Mismatch at index {index}"
    gen_answer = gen_files[index]["answer"]

    gt_answer = gt_answer.lower().strip()
    gen_answer = gen_answer.lower().strip()

    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
        else:
            false_pos += 1
            yes_answers += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1

precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (true_pos + true_neg) / total_questions
yes_proportion = yes_answers / total_questions
unknown_prop = unknown / total_questions

output_path = "./evaluation_results.txt"

output_content = (
    f"gt_files: {args.gt_files}\n"
    f"TP\tFP\tTN\tFN\n"
    f"{true_pos}\t{false_pos}\t{true_neg}\t{false_neg}\n\n"
    f"Accuracy: {accuracy:.16f}\n"
    f"Precision: {precision:.16f}\n"
    f"Recall: {recall:.2f}\n"
    f"F1 score: {f1:.16f}\n"
    f"Yes ratio: {yes_proportion:.16f}\n"
    f"==================================================================\n"
    f"\n"
)

with open(output_path, 'a') as f:
    f.write(output_content)

print(f"Results saved to {output_path}")