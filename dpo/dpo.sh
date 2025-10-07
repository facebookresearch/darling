#!/bin/bash

#SBATCH --nodes=4
#SBATCH --mem-per-cpu=0g
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --gpus-per-node=8
#SBATCH --account=ram
#SBATCH --qos=ram_high
#SBATCH --signal=USR1@120
#SBATCH --open-mode=append
#SBATCH --job-name=[diversity]-divpo
#SBATCH --output=slurm/slurm_%A.out
#SBATCH --error=slurm/slurm_%A.err

source ~/.bashrc
source ~/.zshrc
conda init
conda activate /home/chuanyang/miniconda3/envs/fairseq2
CONFIG="/home/tianjian/darling/dpo/dpo_config.yaml"

PROJECT="divpo"
DATANAME="dpo_0.2_new" # no ".jsonl"
DATASET="/checkpoint/ram/tianjian/${DATANAME}.jsonl"


BETA=0.1
LR=5e-7
FINAL_LR=1e-15
WARMUP=30
EPS=1e-8
RUNNAME="${PROJECT}_${DATANAME}_LR${LR}_FLR${FINAL_LR}_BT${BETA}_WP${WARMUP}_EPS${EPS}_new"
# bsz = 2 * 4 * 1 * 8 = 32


srun --export=ALL fairseq2 lm preference_finetune --no-sweep-dir /checkpoint/ram/tianjian/divpo_ckpts/${RUNNAME} \
--config-file ${CONFIG} --config dataset.path=${DATASET} criterion.config.beta=${BETA} optimizer.config.lr=${LR} regime.checkpoint_every_n_steps=100 lr_scheduler.config.num_warmup_steps=${WARMUP} lr_scheduler.config.final_lr=${FINAL_LR} trainer.grad_accumulation.num_batches=4 optimizer.config.eps=${EPS}

