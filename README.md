<div align="center">
  <h1>WAVE-0: Bridging VLA Policies and WAMs with Lightweight Latent Future Prediction</h1>

  <!-- TODO: Replace the paper placeholder with the published URL. -->
  <a href="#"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b?logo=arXiv" alt="Paper: Coming Soon"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://hmwang2002.github.io/release/wave0/"><img src="https://img.shields.io/badge/Project-Page-green" alt="Project Page"></a> &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/collections/KiyotakaWang/wave-0"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-HF-orange" alt="Models on Hugging Face"></a>
</div>

## 📚 Introduction

We introduce **WAVE-0 (World-model-Augmented VLA Execution)**, a framework that bridges **vision–language–action (VLA) policies** and **world action models (WAMs)** through action-grounded latent future prediction. Built on π0.5, WAVE-0 unifies latent planning, future prediction, and action generation, allowing anticipated action consequences to directly guide robotic manipulation.

- **🌊 Lightweight future-conditioned action generation** — A **257M-parameter latent world model** transforms VLM-generated latent action plans into predicted future visual representations that condition the action expert during both training and inference, without reconstructing future images.
- **🎯 Action-grounded two-stage training** — We first learn action-dependent latent dynamics directly from ground-truth robot actions with a counterfactual ranking objective, then integrate the frozen world model into a pretrained VLA to guide action generation with predicted future representations, without requiring additional large-scale video pretraining.
- **🤖 Strong performance and generalization** — WAVE-0 achieves **91.47%** success on **RoboTwin 2.0** and **98.7%** on **LIBERO**, with improved generalization under visual, spatial, and language distribution shifts on **LIBERO-Plus**, **LIBERO-Pro**, and **LIBERO-Para**. Across three real-world manipulation tasks, it improves the average success rate of π0.5 from **38.3% to 73.3%**.

## 🔥 News

- **[2026-10-14]** 👋 Upload paper and init project. [Read]()

## 📝 Open-Source Plan

- [ ] Paper
- [x] Model weights
- [x] Training scripts

## 📌 Quick Start

### ⚙️ Installation

WAVE-0 is built upon the π0.5 implementation in [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/v0.4.4).

```bash
conda create -n wave0 -c conda-forge python=3.10 'ffmpeg>=7,<8' -y
conda activate wave0
pip install --upgrade pip

# Select the CUDA build explicitly before installing the remaining dependencies.
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-wave0.txt
pip check
```

### 🚀 Train

#### LIBERO

##### 1. Download the model and dataset

Download the [π0.5 base model](https://huggingface.co/lerobot/pi05_base) and the standard 40-task [LIBERO dataset](https://huggingface.co/datasets/HuggingFaceVLA/libero). Both stages use the mean/std statistics in the dataset's `meta/stats.json`; no quantile computation is needed. Run the following commands from the repository root.

```bash
hf download lerobot/pi05_base --local-dir /path/to/models/pi05_base
hf download HuggingFaceVLA/libero --repo-type dataset --local-dir /path/to/data/libero
```

##### 2. Train Stage 1

Train the latent world model for 20,000 steps:

```bash
DATASET_ROOT=/path/to/data/libero \
PRETRAINED_MODEL_PATH=/path/to/models/pi05_base \
OUTPUT_DIR=/path/to/outputs/wave0_libero_stage1 \
bash scripts/train/libero/train_libero_stage1.sh
```

##### 3. Train Stage 2

Initialize the policy from π0.5 and load the Stage-1 checkpoint to train for 15,000 steps:

```bash
DATASET_ROOT=/path/to/data/libero \
POLICY_BASE_PATH=/path/to/models/pi05_base \
STAGE1_CHECKPOINT_PATH=/path/to/outputs/wave0_libero_stage1/checkpoints/020000/pretrained_model \
OUTPUT_DIR=/path/to/outputs/wave0_libero_stage2 \
bash scripts/train/libero/train_libero_stage2.sh
```

Both scripts default to 8 GPUs with a per-GPU batch size of 64 (global batch 512). The action chunk length is 32, with 10 actions executed per prediction and a future prediction horizon of 9 frames. Override `NUM_GPUS` and `BATCH_SIZE` as needed. We use the Stage-2 checkpoint at 10,000 steps for the LIBERO results reported in our paper.

#### RoboTwin2.0

##### 1. Download the model and dataset

Before training, we need to download the [π0.5 base model](https://huggingface.co/lerobot/pi05_base) and the [RoboTwin dataset](https://huggingface.co/datasets/lerobot/robotwin_unified).

```bash
hf download lerobot/pi05_base --local-dir /path/to/models/pi05_base
hf download lerobot/robotwin_unified --repo-type dataset --local-dir /path/to/data/robotwin_unified
```

##### 2. Prepare quantile statistics

Both stages use quantile normalization for actions and robot states, requiring the 1st and 99th percentiles (`q01` and `q99`). We provide precomputed statistics for the full RoboTwin dataset (6,075,103 frames) in [scripts/train/robotwin/robotwin_stats.json](scripts/train/robotwin/robotwin_stats.json). The training commands below use this file via `DATASET_STATS_PATH`, so recomputation is not required for this dataset.

To compute statistics yourself, run the following command once on the complete training dataset. It reads only the action/state columns from Parquet files, without decoding videos or uploading to Hugging Face, and saves a new statistics file without modifying the original dataset. For a modified dataset, recompute the statistics instead of using the bundled RoboTwin file, and use the same statistics for both stages.

```bash
python scripts/train/robotwin/compute_quantile_stats.py \
    --root /path/to/data/robotwin_unified \
    --output /path/to/data/robotwin_unified/meta/stats.quantiles.json
```

If you compute your own statistics, replace `DATASET_STATS_PATH` in both commands below with the generated file path. Run these commands from the repository root.

##### 3. Train Stage 1

Train the latent world model using ground-truth robot actions:

```bash
DATASET_ROOT=/path/to/data/robotwin_unified \
DATASET_STATS_PATH=scripts/train/robotwin/robotwin_stats.json \
PRETRAINED_MODEL_PATH=/path/to/models/pi05_base \
OUTPUT_DIR=/path/to/outputs/wave0_stage1 \
bash scripts/train/robotwin/train_robotwin_stage1.sh
```

##### 4. Train Stage 2

After Stage 1 finishes, initialize the policy from the π0.5 base model and load the Stage-1 checkpoint:

```bash
DATASET_ROOT=/path/to/data/robotwin_unified \
DATASET_STATS_PATH=scripts/train/robotwin/robotwin_stats.json \
POLICY_BASE_PATH=/path/to/models/pi05_base \
STAGE1_CHECKPOINT_PATH=/path/to/outputs/wave0_stage1/checkpoints/050000/pretrained_model \
OUTPUT_DIR=/path/to/outputs/wave0_stage2 \
bash scripts/train/robotwin/train_robotwin_stage2.sh
```

### 📊 Evaluation

#### LIBERO

For LIBERO evaluation, follow the [official LIBERO guide](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the simulator and task assets into the `wave0` environment.

```bash
POLICY_PATH=/path/to/wave0/checkpoint \
LIBERO_SRC=/path/to/LIBERO \
bash scripts/eval/libero/eval_libero.sh
```

By default, the script evaluates all 40 tasks across the four original LIBERO suites, with 50 episodes per task and seed 42.

#### RoboTwin2.0

For RoboTwin 2.0 evaluation, follow the [official LeRobot RoboTwin guide](https://huggingface.co/docs/lerobot/robotwin) for simulator installation, asset downloads, and environment setup. Refer to [requirements-eval.txt](scripts/eval/robotwin/requirements-eval.txt) for evaluation dependencies.

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
POLICY_PATH=/path/to/checkpoint/pretrained_model \
NUM_GPUS=8 \
WORKERS_PER_GPU=4 \
bash scripts/eval/robotwin/eval_robotwin_all.sh
```

## 📜 Acknowledgements

Our implementation builds on [LeRobot](https://github.com/huggingface/lerobot) and [OpenPI](https://github.com/Physical-Intelligence/openpi). We thank their contributors for sharing their code and supporting open-source robotics research.

## 📖 Citation

```BibTex

```
