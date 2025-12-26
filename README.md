# DARLING
This is the official implementation of the paper [Jointly Reinforcing Diversity and Quality of Language Model Generations](https://arxiv.org/abs/2509.02534).

DARLING uses the [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning) framework to jointly optimize for both diversity and quality in language model generations through reinforcement learning.

## Getting Started for Training

### Environment Setup
Create conda environment and install dependencies:

    conda create -n verlenv python=3.10
    conda activate verlenv

Install PyTorch (tested on CUDA 12.8):

    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

Install verl and other dependencies:

    cd verl
    pip install -e ./
    # This code only uses FSDP. If you need to use Megatron, remove USE_MEGATRON=0
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
    pip install vllm==0.11.0
    pip install flash-attn --no-build-isolation

To use Wandb for experiment tracking:

    export WANDB_API_KEY=<your_api_key>


## Training Scripts
- **Verifiable tasks (math)**: `verl/math_scripts/`
- **Non-verifiable tasks (creative writing)**: `verl/wildchat_scripts/`

Each directory contains:
- `darling.batch`: DARLING training with diversity rewards
- `grpo_baseline.batch`: GRPO baseline for comparison

## Running DARLING

### 1. Serve the Partition Classifier
First, serve the partition classifier used for diversity rewards:

    bash verl/serve_classifier.sh <PATH_TO_CLASSIFIER_HF>

This will serve 8 instances of the classifier on ports 8000-8007.

### 2. Set the Server Hostname
Set the hostname where the classifier is running:

    export VLLM_SERVER_HOSTNAME=<your_hostname>

Alternatively, you can manually edit `verl/verl/utils/reward_score/partition_reward_vllm_serve.py`.

### 3. Launch Training
For **math tasks** (Qwen-4B-Base):

    # Edit verl/math_scripts/darling.batch to configure your cluster settings
    sbatch verl/math_scripts/darling.batch

For **creative writing tasks** (Llama-3.1-8B-Instruct):

    # Edit verl/wildchat_scripts/darling.batch to configure your cluster settings
    sbatch verl/wildchat_scripts/darling.batch

## Configuring Hyperparameters
Hyperparameters can be configured by editing the script variables or passing command-line arguments:

**Key Parameters:**
- `B`: Training batch size (e.g., 256 for math, 64 for wildchat)
- `N`: Number of samples per prompt (default: 8)
- `L`: Maximum response length (e.g., 8192 for math, 1024 for wildchat)
- `actor_rollout_ref.actor.optim.lr`: Learning rate (default: 1e-6)
- `actor_rollout_ref.rollout.temperature`: Sampling temperature
- `trainer.total_epochs`: Total training epochs

For the full list of available hyperparameters, see the training scripts in `verl/math_scripts/` and `verl/wildchat_scripts/`.

## Evaluation

The `evals/` directory contains benchmarks for evaluating model outputs:

### Math Evaluation (`evals/math_evaluation/`)
Evaluates mathematical reasoning on standard benchmarks. See [evals/math_evaluation/README.md](evals/math_evaluation/README.md) for setup and usage.

### NoveltyBench (`evals/novelty-bench/`)
Evaluates the diversity and novelty of model generations. This benchmark:
- Generates multiple responses from models
- Groups semantically similar responses using partitioning
- Scores response quality
- Provides a diversity-quality tradeoff analysis

See [evals/novelty-bench/README.md](evals/novelty-bench/README.md) for details and [project webpage](https://novelty-bench.github.io/).

### Creative Writing Benchmark (`evals/creative-writing-bench/`)
Evaluates creative writing capabilities using the EQ-Bench v3 system with hybrid rubric and Elo scoring. See [evals/creative-writing-bench/README.md](evals/creative-writing-bench/README.md) for details.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation
If you find DARLING useful, please consider citing:
```bibtex
@article{tianjian2025jointlyreinforcingdiversityquality,
	title        = {Jointly Reinforcing Diversity and Quality in Language Model Generations},
	author       = {Tianjian Li and Yiming Zhang and Ping Yu and Swarnadeep Saha and Daniel Khashabi and Jason Weston and Jack Lanchantin and Tianlu Wang},
	year         = 2025,
	journal      = {arXiv preprint arXiv:2509.02534},
	url          = {https://arxiv.org/abs/2509.02534},
	eprint       = {2509.02534},
	archiveprefix = {arXiv},
	primaryclass = {cs.CL},
}
```
