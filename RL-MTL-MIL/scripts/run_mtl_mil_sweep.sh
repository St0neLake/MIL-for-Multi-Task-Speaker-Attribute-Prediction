#!/bin/bash

# Navigate to the root of the RL-MIL-main-snellius directory
cd ..
source venv/bin/activate

# ---- Multi-Task MIL Hyperparameter Sweep Configuration ----
# This script is for pre-tuning the base MIL model (e.g., MeanMLP) for MTL.
# The goal is to find the best hyperparameters and generate a best_model_config.json
# which can then be used by the full RL-MTL framework.

# --- Fixed Parameters for this Sweep ---
BASELINE_TO_TUNE="MaxMLP" # The base MIL model architecture to be tuned

# Multi-Task Labels should be space-separated
TARGET_LABELS_MTL="age gender party"

# Dataset and Embedding Details
DATASET="political_data_with_age"
DATA_EMBEDDED_COLUMN_NAME="text"
EMBEDDING_MODEL="roberta-base"
AUTOENCODER_LAYER_SIZES="768,256,768"
BAG_SIZE=20
TASK_TYPE="classification" # All tasks are classification

# --- WandB Configuration ---
WANDB_ENTITY="stonelake-university-of-amsterdam"
WANDB_PROJECT="MIL_MTL_small_MaxMLP"  # A dedicated project for MIL pre-tuning sweeps is recommended

# --- Execution Configuration ---
GPU_ID=0 # GPU for the sweep agent
# This seed is for data splitting consistency. It should match the seed you intend
# to use for the subsequent RL-MIL run for a fair comparison.
SWEEP_RUN_RANDOM_SEED=1


echo "---------------------------------------------------------------------"
echo "🚀 Starting Multi-Task MIL Pre-Tuning Sweep for: $BASELINE_TO_TUNE"
echo "   Labels: $TARGET_LABELS_MTL"
echo "   Dataset: $DATASET"
echo "   GPU ID for CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "   W&B Project: $WANDB_PROJECT"
echo "   Sweep Data Seed (for data split): $SWEEP_RUN_RANDOM_SEED"
echo "---------------------------------------------------------------------"

# Construct the python command for run_mil.py with sweep enabled
# Hyperparameters will be injected by the W&B agent based on the sweep's YAML configuration.
PYTHON_CMD_MIL_SWEEP="python3 run_mil.py \\
    --run_sweep \\
    --baseline \"$BASELINE_TO_TUNE\" \\
    --label $TARGET_LABELS_MTL \\
    --bag_size $BAG_SIZE \\
    --embedding_model \"$EMBEDDING_MODEL\" \\
    --dataset \"$DATASET\" \\
    --autoencoder_layer_sizes \"$AUTOENCODER_LAYER_SIZES\" \\
    --data_embedded_column_name \"$DATA_EMBEDDED_COLUMN_NAME\" \\
    --task_type \"$TASK_TYPE\" \\
    --random_seed $SWEEP_RUN_RANDOM_SEED \\
    --wandb_entity \"$WANDB_ENTITY\" \\
    --wandb_project \"$WANDB_PROJECT\" \\
    --gpu $GPU_ID"

# Run in a screen session with output logging
SESSION_NAME="mtl_mil_sweep_${DATASET}_${BASELINE_TO_TUNE}"
FULL_COMMAND_MIL_SWEEP="CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD_MIL_SWEEP"
LOG_FILE_PATH="sweep_run_log_${SESSION_NAME}_$(date +%s).txt"

echo "Executing W&B sweep agent launch in screen session '$SESSION_NAME':"
echo "$FULL_COMMAND_MIL_SWEEP"
echo "Full command output will be logged to: $LOG_FILE_PATH"
echo "---------------------------------------------------------------------"

screen -L -Logfile "$LOG_FILE_PATH" -dmS "$SESSION_NAME" bash -c "
  echo 'Starting Python script for sweep agent...'
  $FULL_COMMAND_MIL_SWEEP; \\
  echo ''; \\
  echo '--------------------------------------------------'; \\
  echo 'MTL MIL Pre-Tuning Sweep setup script finished.'; \\
  echo 'Check W&B for the SWEEP_ID if printed by run_mil.py.'; \\
  echo 'Then run: wandb agent YOUR_SWEEP_ID'; \\
  echo 'Screen session $SESSION_NAME will remain active.'; \\
  echo 'To detach: Ctrl+A then D'; \\
  echo '--------------------------------------------------'; \\
  exec bash"

echo "✅ MTL MIL Pre-Tuning Sweep setup launched in screen session: $SESSION_NAME"
echo "   To attach to the session, type: screen -r $SESSION_NAME"
echo "   Follow instructions printed by run_mil.py or on the W&B dashboard to start sweep agents."
echo "---------------------------------------------------------------------"