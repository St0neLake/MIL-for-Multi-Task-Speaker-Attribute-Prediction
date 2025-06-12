#!/bin/bash

# A script to launch a single, specific RL-MTL run without task clustering,
# using a predefined set of hyperparameters.

# Navigate to the root of the RL-MIL-main directory
cd .. # Assumes script is in a 'scripts' subdirectory
source venv/bin/activate

# --- Final Run Configuration ---
RUN_PREFIX="final_run_mtl_best_hps"
GPU_ID=0
RANDOM_SEED=64 # Use a consistent seed for reproducibility
WANDB_ENTITY="stonelake-university-of-amsterdam"
WANDB_PROJECT="RL_MIL_MTL_Final_Run"

# --- Data & Model Architecture ---
DATASET="political_data_with_age"
EMBEDDING_MODEL="roberta-base"
DATA_EMBEDDED_COLUMN_NAME="text"
BASELINE_TYPE="MeanMLP"
AUTOENCODER_LAYER_SIZES="768,256,768"
BAG_SIZE=20
TASK_TYPE="classification"

# --- Specific Hyperparameters To Use ---
EPOCHS=800
BATCH_SIZE=128
LEARNING_RATE=4.171455905508819e-05   # For the MIL task_model
ACTOR_LEARNING_RATE=1.6885128518671337e-05
CRITIC_LEARNING_RATE=5.348406944845868e-05
HDIM=32
# Epsilon and Reg Coef were not in your list, using reasonable defaults
EPSILON=0.451614157
REG_COEF=0.00030797
EARLY_STOPPING_PATIENCE=50
EVAL_POOL_SIZE=10               # A stable value for evaluation
TRAIN_POOL_SIZE=1

# --- RL Algorithm Parameters ---
RL_MODEL="policy_only"
RL_TASK_MODEL="vanilla"
SAMPLE_ALGORITHM="without_replacement"
SEARCH_ALGORITHM="epsilon_greedy"
REG_ALG="sum"

echo "---------------------------------------------------------------------"
echo "🚀 Starting Single RL-MTL Run (No Clustering)"
echo "   Tasks: age, gender, party"
echo "   Using a specific, predefined set of hyperparameters."
echo "   Run Prefix: $RUN_PREFIX"
echo "---------------------------------------------------------------------"

# Note: --run_sweep and --clustering_active are NOT included
PYTHON_CMD="python3 run_rlmil.py \
    --rl \
    --prefix \"$RUN_PREFIX\" \
    --gpu $GPU_ID \
    --random_seed $RANDOM_SEED \
    --wandb_entity \"$WANDB_ENTITY\" \
    --wandb_project \"$WANDB_PROJECT\" \
    --dataset \"$DATASET\" \
    --embedding_model \"$EMBEDDING_MODEL\" \
    --data_embedded_column_name \"$DATA_EMBEDDED_COLUMN_NAME\" \
    --baseline \"$BASELINE_TYPE\" \
    --autoencoder_layer_sizes \"$AUTOENCODER_LAYER_SIZES\" \
    --bag_size $BAG_SIZE \
    --label age gender party \
    --task_type \"$TASK_TYPE\" \
    --balance_dataset \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --actor_learning_rate $ACTOR_LEARNING_RATE \
    --critic_learning_rate $CRITIC_LEARNING_RATE \
    --hdim $HDIM \
    --epsilon $EPSILON \
    --reg_coef $REG_COEF \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --train_pool_size $TRAIN_POOL_SIZE \
    --eval_pool_size $EVAL_POOL_SIZE \
    --test_pool_size 10 \
    --rl_model \"$RL_MODEL\" \
    --rl_task_model \"$RL_TASK_MODEL\" \
    --sample_algorithm \"$SAMPLE_ALGORITHM\" \
    --search_algorithm \"$SEARCH_ALGORITHM\" \
    --reg_alg \"$REG_ALG\""

# Run directly in the terminal
echo "Executing command:"
echo "$PYTHON_CMD"
eval "$PYTHON_CMD"

echo "✅ Run finished."