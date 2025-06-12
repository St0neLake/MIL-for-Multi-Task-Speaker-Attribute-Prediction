#!/bin/bash

# A script to launch a single, definitive RL-MTL run with task clustering,
# using the best hyperparameters found from a previous sweep on a subset.

# Navigate to the root of the RL-MIL-main-snellius directory
cd .. # Assumes script is in a 'scripts' subdirectory
source venv/bin/activate

# --- Final Run Configuration ---
RUN_PREFIX="final_clust_run_best_hps"
GPU_ID=0
RANDOM_SEED=64 # Use a consistent seed for your final run
WANDB_ENTITY="stonelake-university-of-amsterdam"    # Your W&B entity
WANDB_PROJECT="RL_MIL_MTL_big_cluster_final" # Dedicated project for RL-MTL sweeps

# --- Data & Model Architecture (Should match your experiment) ---
DATASET="political_data_with_age"
EMBEDDING_MODEL="roberta-base"
BASELINE_TYPE="MeanMLP"
AUTOENCODER_LAYER_SIZES="768,256,768"
BAG_SIZE=20

# --- Best Hyperparameters from Subset Run ---
EPOCHS=800
BATCH_SIZE=125
LEARNING_RATE=5.0e-05
ACTOR_LEARNING_RATE=2.03318568e-05
CRITIC_LEARNING_RATE=0.0001
HDIM=128
EPSILON=0.451614157
REG_COEF=0.00030797
WARMUP_EPOCHS=25
EARLY_STOPPING_PATIENCE=50 # Using a slightly higher patience for the final run is often a good idea
EVAL_POOL_SIZE=10          # Using a higher pool size for more stable final metrics

# --- Clustering Hyperparameters from Subset Run ---
NUM_TASK_CLUSTERS=2 # Assuming this was fixed during the sweep
HIDDEN_DIM_CLUSTER_TRUNK=128
HIDDEN_DIM_FINAL_HEAD=64
RECLUSTER_INTERVAL_E=5

echo "---------------------------------------------------------------------"
echo "🚀 Starting FINAL RL-MTL Run with Task Clustering"
echo "   Using best hyperparameters from subset sweep."
echo "   Run Prefix: $RUN_PREFIX"
echo "---------------------------------------------------------------------"

# Note: --run_sweep is NOT included for a single run
PYTHON_CMD="python3 run_rlmil.py \
    --rl \
    --clustering_active \
    --prefix \"$RUN_PREFIX\" \
    --gpu $GPU_ID \
    --random_seed $RANDOM_SEED \
    --wandb_entity \"$WANDB_ENTITY\" \
    --wandb_project \"$WANDB_PROJECT\" \
    --dataset \"$DATASET\" \
    --embedding_model \"$EMBEDDING_MODEL\" \
    --baseline \"$BASELINE_TYPE\" \
    --autoencoder_layer_sizes \"$AUTOENCODER_LAYER_SIZES\" \
    --bag_size $BAG_SIZE \
    --label age gender party \
    --task_type "classification" \
    --balance_dataset \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --actor_learning_rate $ACTOR_LEARNING_RATE \
    --critic_learning_rate $CRITIC_LEARNING_RATE \
    --hdim $HDIM \
    --epsilon $EPSILON \
    --reg_coef $REG_COEF \
    --warmup_epochs $WARMUP_EPOCHS \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --eval_pool_size $EVAL_POOL_SIZE \
    --num_task_clusters $NUM_TASK_CLUSTERS \
    --hidden_dim_cluster_trunk $HIDDEN_DIM_CLUSTER_TRUNK \
    --hidden_dim_final_head $HIDDEN_DIM_FINAL_HEAD \
    --recluster_interval_E $RECLUSTER_INTERVAL_E"

# Run directly in the terminal
echo "Executing command:"
echo "$PYTHON_CMD"
eval "$PYTHON_CMD"

echo "✅ Final run finished."