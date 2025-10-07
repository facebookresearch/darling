#!/bin/bash

#SBATCH --nodes=3
#SBATCH --mem-per-cpu=0g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=8
#SBATCH --account=ram
# #SBATCH --account=ram_external
#SBATCH --qos=ram_high
# #SBATCH --qos=alignment_shared
# #SBATCH --qos=ram_external_high
#SBATCH --signal=USR1@120
#SBATCH --open-mode=append
#SBATCH --job-name=rlhi_dpo
#SBATCH --output=slurm/slurm_%A.out
#SBATCH --error=slurm/slurm_%A.err


# Setup environment
export FI_EFA_USE_HUGE_PAGE=0
eval "$(conda shell.bash hook)"
source /fsx-ram/$USER/miniconda3/bin/activate
conda activate fairseq2

CONFIG="/home/chuanyang/dpo/dpo_config.yaml"

# PROJECT="rip_0710"
# DATANAME="Wildchat-RIP-Filtered-by-8b-Llama"

PROJECT="math_0907"
DATANAME="prm800k_phase2_processed_final" # no ".jsonl"

# PROJECT="mixed_0710"
# DATANAME="mixed"

DATASET="/home/chuanyang/math/${DATANAME}.jsonl"


BETA=0.1
LR=7e-7
FINAL_LR=1e-15
WARMUP=30
EPS=1e-8
RUNNAME="${PROJECT}_${DATANAME}_LR${LR}_FLR${FINAL_LR}_BT${BETA}_WP${WARMUP}_EPS${EPS}"
# bsz = 2 * 4 * 1 * 8 = 32

# srun fairseq2 lm preference_finetune --no-sweep-dir /fsx-ram/jingxu23/projects/gen_better_prompts/checkpoints/dpo/${RUNNAME} \
# --config-file ${CONFIG} --config dataset=${DATASET} criterion_config.beta=${BETA} optimizer_config.lr=${LR} lr_scheduler_config.final_lr=${LR} mixed_precision="static" checkpoint_every_n_steps=50 lr_scheduler_config.num_warmup_steps=${WARMUP} wandb_project=${PROJECT} wandb_run_name=${RUNNAME} gradient_accumulation=4 optimizer_config.eps=${EPS}

# srun fairseq2 lm preference_finetune --no-sweep-dir /checkpoint/ram/chuanyang/dpo/${RUNNAME} \
# --config-file ${CONFIG} --config dataset.path=${DATASET} criterion.config.beta=${BETA} optimizer.config.lr=${LR} lr_scheduler.config.final_lr=${LR} regime.checkpoint_every_n_steps=200 lr_scheduler.config.num_warmup_steps=${WARMUP} trainer.gradient_accumulation=4 optimizer.config.eps=${EPS}

export WANDB_BASE_URL="https://fairwandb.org"
export WANDB_API_KEY="local-94bc1b16aff0febd7d5abee21969c9eb2451a048"
export WANDB_ENTITY="chuanyang"

srun --export=ALL fairseq2 lm preference_finetune --no-sweep-dir /checkpoint/ram/chuanyang/math/${RUNNAME} \
--config-file ${CONFIG} --config dataset.path=${DATASET} criterion.config.beta=${BETA} optimizer.config.lr=${LR} regime.num_steps=2000 regime.checkpoint_every_n_steps=500 regime.checkpoint_every_n_data_epochs=100 lr_scheduler.config.num_warmup_steps=${WARMUP} lr_scheduler.config.final_lr=${FINAL_LR} trainer.grad_accumulation.num_batches=4 optimizer.config.eps=${EPS} common.metric_recorders.wandb.enabled=True common.metric_recorders.wandb.project=${PROJECT} common.metric_recorders.wandb.run_name=${RUNNAME}

# fairseq2 lm preference_finetune --dump-config
# cd /checkpoint/ram/benjaminliueecs/work/projects/personalized_language_model
# jq -c '.[]' processbench_synthetic_data_merged_v2.json > ~/data/processbench_synthetic_data_merged_v2.jsonl