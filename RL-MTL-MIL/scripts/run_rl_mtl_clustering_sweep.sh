#!/bin/bash

# Navigate to the root of the RL-MIL-main-snellius directory
# Adjust this if your script is not in a 'scripts' subdirectory of RL-MIL-main-snellius
cd .. # Assumes script is in a 'scripts' subdirectory
source venv/bin/activate

# ---- RL-MTL with Task Clustering Hyperparameter Sweep Configuration ----
# This script initiates a W&B sweep for the RL-MTL framework WITH TASK CLUSTERING.

# --- Fixed Parameters for this Sweep ---
MTL_BASELINE_TYPE="MaxMLP"
MTL_TARGET_LABELS_STR="age gender party"

DATASET="political_data_with_age"
DATA_EMBEDDED_COLUMN_NAME="text"
EMBEDDING_MODEL="roberta-base"
AUTOENCODER_LAYER_SIZES="768,256,768" # For the base_network of ClusteredMeanMLP
BAG_SIZE=20
TASK_TYPE="classification"

# --- RL Specific Fixed Parameters ---
RL_MODEL="policy_only"
RL_TASK_MODEL="vanilla"
SAMPLE_ALGORITHM="without_replacement"
SEARCH_ALGORITHM="epsilon_greedy"
REG_ALG="sum"

# --- Task Clustering Specific Fixed Parameters ---
# Set to true to activate clustering logic in run_rlmil.py
CLUSTERING_ACTIVE="--clustering_active" # Pass as a flag
NUM_TASK_CLUSTERS=2                     # e.g., K=2 for 3 tasks
HIDDEN_DIM_CLUSTER_TRUNK=128            # Example dimension for shared cluster trunks
HIDDEN_DIM_FINAL_HEAD=64                # Example dimension for final heads after trunks
RECLUSTER_INTERVAL_E=10                 # Re-cluster every E epochs
KMEANS_RANDOM_STATE=42                  # For KMeans reproducibility

# Prefix: This helps configs.py select the correct sweep YAML.
# Adjust if your YAML selection logic in configs.py needs a specific prefix for clustering sweeps.
# e.g., if your YAML is hp_rl_mtl_clust.yaml and configs.py looks for "mtl_clust"
BASE_PREFIX_FOR_PY_SCRIPT="mtl_clust_sweep"

# --- WandB Configuration ---
WANDB_ENTITY="stonelake-university-of-amsterdam"   # Your W&B entity
WANDB_PROJECT="RL_MIL_MTL_Clustering_small_MaxMLP"              # Dedicated project for clustering sweeps

# --- Execution Configuration ---
GPU_ID=0
SWEEP_RUN_RANDOM_SEED=1 # Seed for data splitting consistency

# --- Fixed Pool Sizes (if not part of the sweep YAML) ---
TRAIN_POOL_SIZE=1
EVAL_POOL_SIZE=3
TEST_POOL_SIZE=10

echo "---------------------------------------------------------------------"
echo "🚀 Initiating RL-MTL with Task Clustering Hyperparameter Sweep"
echo "   Labels: $MTL_TARGET_LABELS_STR"
echo "   Dataset: $DATASET, Base MIL Model: $MTL_BASELINE_TYPE"
echo "   Clustering Active: Yes, K=$NUM_TASK_CLUSTERS"
echo "   Sweep Data Seed: $SWEEP_RUN_RANDOM_SEED"
echo "   W&B Project: $WANDB_PROJECT"
echo "   Target Sweep YAML should be selected by run_rlmil.py based on args."
echo "---------------------------------------------------------------------"

PYTHON_CMD_RL_SWEEP="CUDA_LAUNCH_BLOCKING=1 python3 run_rlmil.py \
    --run_sweep \
    --rl \
    --baseline \"$MTL_BASELINE_TYPE\" \
    --label $MTL_TARGET_LABELS_STR \
    --bag_size $BAG_SIZE \
    --embedding_model \"$EMBEDDING_MODEL\" \
    --dataset \"$DATASET\" \
    --autoencoder_layer_sizes \"$AUTOENCODER_LAYER_SIZES\" \
    --data_embedded_column_name \"$DATA_EMBEDDED_COLUMN_NAME\" \
    --task_type \"$TASK_TYPE\" \
    --random_seed $SWEEP_RUN_RANDOM_SEED \
    --wandb_entity \"$WANDB_ENTITY\" \
    --wandb_project \"$WANDB_PROJECT\" \
    --gpu $GPU_ID \
    --rl_model \"$RL_MODEL\" \
    --rl_task_model \"$RL_TASK_MODEL\" \
    --sample_algorithm \"$SAMPLE_ALGORITHM\" \
    --search_algorithm \"$SEARCH_ALGORITHM\" \
    --prefix \"$BASE_PREFIX_FOR_PY_SCRIPT\" \
    --train_pool_size $TRAIN_POOL_SIZE \
    --eval_pool_size $EVAL_POOL_SIZE \
    --test_pool_size $TEST_POOL_SIZE \
    $CLUSTERING_ACTIVE \
    --num_task_clusters $NUM_TASK_CLUSTERS \
    --hidden_dim_cluster_trunk $HIDDEN_DIM_CLUSTER_TRUNK \
    --hidden_dim_final_head $HIDDEN_DIM_FINAL_HEAD \
    --recluster_interval_E $RECLUSTER_INTERVAL_E \
    --kmeans_random_state $KMEANS_RANDOM_STATE \
    --balance_dataset"

if [ ! -z "$REG_ALG" ]; then
    PYTHON_CMD_RL_SWEEP="$PYTHON_CMD_RL_SWEEP --reg_alg $REG_ALG"
fi

SESSION_NAME="rl_mtl_clust_sweep_${DATASET}_K${NUM_TASK_CLUSTERS}"
FULL_COMMAND_RL_SWEEP="CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD_RL_SWEEP"
LOG_FILE_PATH="sweep_run_log_${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).txt"
echo "Full command output will be logged to: $LOG_FILE_PATH"

echo "Executing Python script to initialize W&B sweep and potentially start agent in screen session '$SESSION_NAME':"
echo "$FULL_COMMAND_RL_SWEEP"
echo "Monitor the output in the screen session or W&B dashboard."
echo "---------------------------------------------------------------------"

screen -L -Logfile "$LOG_FILE_PATH" -dmS "$SESSION_NAME" bash -c "
  echo 'Starting Python script for sweep agent for RL-MTL with Clustering...'
  $FULL_COMMAND_RL_SWEEP; \
  echo ''; \
  echo '--------------------------------------------------'; \
  echo 'RL-MTL Clustering Sweep setup script finished or agent started.'; \
  echo 'Screen session $SESSION_NAME will remain active if agent is running or if script ended with exec bash.'; \
  echo 'To detach: Ctrl+A then D'; \
  echo '--------------------------------------------------'; \
  exec bash"

echo "✅ RL-MTL Clustering Sweep setup launched in screen session: $SESSION_NAME"
echo "   To attach to the session, type: screen -r $SESSION_NAME"
echo "   If the script only initializes the sweep, run 'wandb agent YOUR_SWEEP_ID' separately."
echo "---------------------------------------------------------------------"