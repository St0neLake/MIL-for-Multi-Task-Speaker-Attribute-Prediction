#!/bin/bash

# A script to launch a single pre-tuning run for the base MIL model for MTL
# with a specific set of hyperparameters.

# Navigate to the root of the RL-MIL-main-snellius directory
cd .. # Assumes script is in a 'scripts' subdirectory
source venv/bin/activate

# --- Fixed Parameters for this Run ---
BASELINE_TO_TUNE="MaxMLP"
TARGET_LABELS_MTL="age gender party"
DATASET="political_data_with_age"
DATA_EMBEDDED_COLUMN_NAME="text"
EMBEDDING_MODEL="roberta-base"
AUTOENCODER_LAYER_SIZES="768,256,768"
BAG_SIZE=20
TASK_TYPE="classification"

# --- Hyperparameters for this Specific Run ---
BATCH_SIZE=64
DROPOUT_P=0.5
EPOCHS=800
HIDDEN_DIM=256
LEARNING_RATE=0.0004580371411577495
SCHEDULER_PATIENCE=5
EARLY_STOPPING_PATIENCE=50

# --- WandB Configuration ---
WANDB_ENTITY="stonelake-university-of-amsterdam"
WANDB_PROJECT="MIL_MTL_Final_run_MaxMLP" # Project for MIL pre-tuning runs

# --- Execution Configuration ---
GPU_ID=0
RANDOM_SEED=64 # The random seed for data splitting and model initialization

echo "---------------------------------------------------------------------"
echo "🚀 Starting Multi-Task MIL Single Run for: $BASELINE_TO_TUNE"
echo "   Labels: $TARGET_LABELS_MTL"
echo "   Dataset: $DATASET"
echo "   GPU ID for CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "   W&B Project: $WANDB_PROJECT"
echo "   Random Seed: $RANDOM_SEED"
echo "---------------------------------------------------------------------"

# Construct the python command for run_mil.py for a single run
PYTHON_CMD_MIL_SINGLE_RUN="python3 run_mil.py \
    --baseline \"$BASELINE_TO_TUNE\" \
    --label $TARGET_LABELS_MTL \
    --bag_size $BAG_SIZE \
    --embedding_model \"$EMBEDDING_MODEL\" \
    --dataset \"$DATASET\" \
    --autoencoder_layer_sizes \"$AUTOENCODER_LAYER_SIZES\" \
    --data_embedded_column_name \"$DATA_EMBEDDED_COLUMN_NAME\" \
    --task_type \"$TASK_TYPE\" \
    --random_seed $RANDOM_SEED \
    --wandb_entity \"$WANDB_ENTITY\" \
    --wandb_project \"$WANDB_PROJECT\" \
    --gpu $GPU_ID \
    --batch_size $BATCH_SIZE \
    --dropout_p $DROPOUT_P \
    --epochs $EPOCHS \
    --hidden_dim $HIDDEN_DIM \
    --learning_rate $LEARNING_RATE \
    --scheduler_patience $SCHEDULER_PATIENCE \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE"

# Run in a screen session with output logging
SESSION_NAME="mtl_mil_single_run_${DATASET}_${BASELINE_TO_TUNE}"
FULL_COMMAND_MIL_RUN="CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD_MIL_SINGLE_RUN"
LOG_FILE_PATH="run_log_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).txt"

echo "Executing Python script in screen session '$SESSION_NAME':"
echo "$FULL_COMMAND_MIL_RUN"
echo "Full command output will be logged to: $LOG_FILE_PATH"
echo "---------------------------------------------------------------------"

screen -L -Logfile "$LOG_FILE_PATH" -dmS "$SESSION_NAME" bash -c "
  echo 'Starting Python script for single run...'
  $FULL_COMMAND_MIL_RUN; \
  echo ''; \
  echo '--------------------------------------------------'; \
  echo 'MTL MIL single run script finished.'; \
  echo 'Screen session $SESSION_NAME will remain active.'; \
  echo 'To detach: Ctrl+A then D'; \
  echo '--------------------------------------------------'; \
  exec bash"

echo "✅ MTL MIL single run launched in screen session: $SESSION_NAME"
echo "   To attach to the session and view output, type: screen -r $SESSION_NAME"
echo "---------------------------------------------------------------------"