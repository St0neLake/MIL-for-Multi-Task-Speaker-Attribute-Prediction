#!/bin/bash

cd .. # Navigate to the root of the RL-MIL-main-snellius directory
source venv/bin/activate

# ---- Multi-Task MIL Sweep Configuration ----
# This script is for pre-tuning the base MIL model (e.g., MeanMLP) for MTL.
# The goal is to generate the best_model_config.json for this base MTL MIL model.

BASELINE_TO_TUNE="MeanMLP" # The base MIL model you want to pre-tune

# Multi-Task Labels (ensure your run_mil.py can handle this)
TARGET_LABELS_MTL="age gender party"

# Dataset and Embedding Details (Match your RL-MIL setup)
DATASET="political_data_with_age"
DATA_EMBEDDED_COLUMN_NAME="text"
EMBEDDING_MODEL="roberta-base"
AUTOENCODER_LAYER_SIZES="768,256,768" # Must match what RL-MIL expects for state_dim

BAG_SIZE=20
TASK_TYPE="classification" # All three tasks are classification

# WandB Config
WANDB_ENTITY="stonelake-university-of-amsterdam" # Your W&B entity
WANDB_PROJECT="RL_MIL_MTL_Test_MIL_PreTune"  # Project for MIL pre-tuning sweeps

# GPU for this sweep agent. If running multiple agents, assign different GPUs.
GPU_ID=0

# Random seed for data splitting for this set of sweep runs
# This seed should match the seed you intend to use for the subsequent RL-MIL run
# so that the pre-tuned model is on the same data split.
# Your RL-MIL run used seed 44.
SWEEP_RUN_RANDOM_SEED=44


echo "---------------------------------------------------------------------"
echo "🚀 Starting Multi-Task MIL Pre-Tuning Sweep for: $BASELINE_TO_TUNE"
echo "Labels: $TARGET_LABELS_MTL"
echo "Dataset: $DATASET"
echo "GPU ID for CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "W&B Project: $WANDB_PROJECT"
echo "Sweep Random Seed (for data split): $SWEEP_RUN_RANDOM_SEED"
echo "---------------------------------------------------------------------"

PYTHON_CMD_MIL_SWEEP="python3 run_mil.py \
    --run_sweep \
    --baseline \"$BASELINE_TO_TUNE\" \
    --label $TARGET_LABELS_MTL \
    --bag_size $BAG_SIZE \
    --embedding_model \"$EMBEDDING_MODEL\" \
    --dataset \"$DATASET\" \
    --autoencoder_layer_sizes \"$AUTOENCODER_LAYER_SIZES\" \
    --data_embedded_column_name \"$DATA_EMBEDDED_COLUMN_NAME\" \
    --task_type \"$TASK_TYPE\" \
    --random_seed $SWEEP_RUN_RANDOM_SEED \
    --wandb_entity \"$WANDB_ENTITY\" \
    --wandb_project \"$WANDB_PROJECT\" \
    --gpu $GPU_ID"
    # Add other fixed arguments run_mil.py might need, e.g., --balance_dataset if desired

# Run in a screen session
SESSION_NAME="mtl_mil_sweep_${DATASET}_${BASELINE_TO_TUNE}"
FULL_COMMAND_MIL_SWEEP="CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD_MIL_SWEEP"

echo "Executing W&B sweep agent launch in screen session '$SESSION_NAME':"
echo "$FULL_COMMAND_MIL_SWEEP"
echo "This command will initialize a sweep with W&B and print a SWEEP_ID."
echo "You will then need to run 'wandb agent YOUR_SWEEP_ID' (possibly multiple times for parallel agents)."
echo "This script itself will just print the command to start the agent based on run_mil.py's sweep setup."
echo "---------------------------------------------------------------------"

screen -dmS "$SESSION_NAME" bash -c "
  $FULL_COMMAND_MIL_SWEEP; \
  echo ''; \
  echo '--------------------------------------------------'; \
  echo 'MTL MIL Pre-Tuning Sweep setup script finished.'; \
  echo 'Check W&B for the SWEEP_ID if printed by run_mil.py.'; \
  echo 'Then run: wandb agent YOUR_SWEEP_ID'; \
  echo 'Screen session $SESSION_NAME will remain active.'; \
  echo 'To detach: Ctrl+A then D'; \
  echo '--------------------------------------------------'; \
  exec bash"

echo "✅ MTL MIL Pre-Tuning Sweep setup launched in screen session: $SESSION_NAME"
echo "   To attach to the session, type: screen -r $SESSION_NAME"
echo "   Follow instructions printed by run_mil.py or W&B to start sweep agents."
echo "---------------------------------------------------------------------"