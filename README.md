_**For your convenience, a quick-start notebook is available at: [quick_start.ipynb](./quick_start.ipynb)**_.

# Setup

```bash
conda create -n rfi python==3.11.9
conda activate rfi
pip install -r requirements.txt
```

# Dataset


For training, we use ​​100 selected correct-hallucinated image-text pairs​​ from COCO_train2014. For evaluation, we employ three datasets: ​​COCO_val2014​​, ​​GQA and MME​​.

​​COCO​​: https://cocodataset.org

​​GQA​​: https://huggingface.co/datasets/lmms-lab/GQA

MME: https://huggingface.co/datasets/lmms-lab/MME


# Configuration Parameters

### Model & Data Paths
- `--model-path` (str): Path to LVLMs weights  
- `--image-folder` (str): Directory containing COCO val2014 images  
- `--question-file` (str): Path to AOKVQA/POPE formatted questions  
- `--answers-file` (str): Output path for generated answers (JSON)  
- `--dataset_path` (str): Preprocessed RFI dataset location  
- `--flow_image_path` (str): Training images path (COCO train2014)  
- `--save_model_path` (str): Output path for trained Flow model  


### Flow Parameters
- `--flow_layer` (int): Layer index for flow integration (default: 15)  
- `--flow` (flag): Enable/disable flow module (default: True)  
- `--alpha` (float): Flow intensity coefficient (default: 3.0)  
- `--k` (int): Top-k principal truth directions of SVD (default: 20)  

# Train 
After preparing the dataset and model, run the following command to train the rectified flow and save the results:
```bash
python train_flow.py
```


# Evaluation

To reproduce **POPE** benchmark results on LLaVA, run:

```bash
python pope_llava.py
```
Allow 25 minutes for result generation. Then start evaluation of POPE: 

```bash
python pope_eval.py
```


To reproduce **MME** benchmark results, run:

```bash
python MME_llava.py
```

# GPU Requirements​

The full experiment has been validated on NVIDIA RTX A5000 (24GB).

# Acknowledgements

Our work builds upon [TruthFlow](https://github.com/wwwhy725/TruthFlow). We sincerely thank the authors for open-sourcing their work.
