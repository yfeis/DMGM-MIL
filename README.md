# DMGM-MIL: Discriminative Memory-Guided Mamba for Tumor Mutational Burden Prediction from Whole-Slide Images 

This repository contains the official PyTorch implementation of **DMGM-MIL: Discriminative Memory-Guided Mamba for Tumor Mutational Burden Prediction from Whole-Slide Images**. 

## Prepare Patch Features 
To preprocess WSIs, we used [CLAM](https://github.com/mahmoodlab/CLAM). UNI model and weight can also be found in [this](https://github.com/mahmoodlab/CLAM).
```bash
# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --step_size 256 --preset tcga.csv --seg --patch --stitch --patch_level 1

# Feature Extraction
export UNI_CKPT_PATH=checkpoints/uni/pytorch_model.bin

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY csv_path  CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --model_name uni_v1 --batch_size 512 --slide_ext .svs
```

## Installation 
- Ubuntu 20.04
- Python 3.10
- CUDA 11.8
- NVIDIA GPU (RTX 4090)
- PyTorch 2.0.1

You can refer to the following instructions.
```bash
# Create the conda environment
conda create -n dmgmmil python=3.10 -y
conda activate dmgmmil

# Install PyTorch
## Please install the appropriate versions according to your CUDA and PyTorch versions.
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install torch-scatter torch-sparse torch-geometric

# Install Mamba
pip install mamba-ssm==1.1.2 causal-conv1d==1.1.1

# Install the remaining dependencies
pip install -r requirements.txt
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --embed_dim 1024 --drop_out 0.25 --early_stopping --lr 1e-4 --k 5 --opt sgd --lr_scheduler cosine --label_frac 1 --split_dir SPLIT_DIR --exp_code SAVE_DIR --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type dmgmmil --log_data --data_root_dir DATA_ROOT_DIR
```

## Eval and Test
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --drop_out 0.25 --k 5 --models_exp_code MODEL_CKPT --save_exp_code SAVE_DIR --splits_dir SPLITS_DIR --task task_1_tumor_vs_normal --model_type dmgmmil --results_dir results --data_root_dir DATA_ROOT_DIR
```
