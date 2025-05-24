#!/bin/bash

cd .. # Navigate to the root of the RL-MIL directory (e.g., alirezaziabari/rl-mil/)
source venv/bin/activate

# ---- MTL Configuration ----
mtl_baseline_type="MeanMLP" # Start with one baseline for testing
mtl_target_labels_str="age gender party" # Your three target labels for MTL

# GPU to use (ensure this GPU ID is available and works for you)
# The gpus array from your script was (0 1 2 3). Let's pick the first one.
mtl_gpu_id=0

# wandb config (Matches your provided script)
wandb_entity="stonelake-university-of-amsterdam"
wandb_project="RL_MIL_MTL_Test" # Suggestion: New project for MTL experiments

# Dataset and Embedding Details (Matches your provided script)
dataset="political_data_with_age"
data_embedded_column_name="text"
embedding_model="roberta-base"

# RL-MIL Specific Settings (Matches your provided script)
rl_task_model="vanilla"
sample_algorithm="without_replacement"
autoencoder_layer_sizes="768,256,768"
bag_size=20 # This is an array in your script, taking the first value
rl_model="policy_only"
search_algorithm="epsilon_greedy"
reg_alg="sum" # This will add --reg_alg "sum" to the command
prefix="mtl_test_age_gender_party" # Descriptive prefix for this MTL run

# Hyperparameters for the run (Using defaults from your script or common values)
# For a first test, you might want to use fewer epochs.
epochs=50       # Reduced for a quicker test; adjust as needed (original script implies sweeps define this)
batch_size=32   # From your script's python call
actor_learning_rate=1e-4 # Placeholder - sweeps would determine optimal
critic_learning_rate=1e-4 # Placeholder - sweeps would determine optimal
learning_rate=1e-5       # Placeholder - sweeps would determine optimal (for MIL part)
hdim=8                   # Placeholder - sweeps would determine optimal (for PolicyNetwork actor/critic)
train_pool_size=1        # From your script
eval_pool_size=10        # From your script
test_pool_size=10        # From your script
epsilon=0.1              # Placeholder for epsilon_greedy
reg_coef=0.01            # Placeholder for reg_alg="sum"
mil_task_head_hidden_dim=512

# This random seed will be used for this specific MTL run (data splitting, initializations)
# Your original script used random_seed 0 for sweeps, and 42 for a specific run. Let's use 42.
random_seed_for_run=44

# ---- Constants ----
task_type="classification" # All three tasks are classification

echo "---------------------------------------------------------------------"
echo "🚀 Starting MTL RL-MIL Run"
echo "Labels: $mtl_target_labels_str"
echo "Dataset: $dataset, Baseline: $mtl_baseline_type"
echo "GPU ID for CUDA_VISIBLE_DEVICES: $mtl_gpu_id"
echo "W&B Project: $wandb_project"
echo "---------------------------------------------------------------------"

# Construct the python command
# The --label argument will now receive "age gender party"
PYTHON_CMD="python3 run_rlmil.py \
    --rl \
    --baseline \"$mtl_baseline_type\" \
    --label $mtl_target_labels_str \
    --autoencoder_layer_sizes \"$autoencoder_layer_sizes\" \
    --data_embedded_column_name \"$data_embedded_column_name\" \
    --prefix \"$prefix\" \
    --dataset \"$dataset\" \
    --bag_size $bag_size \
    --batch_size $batch_size \
    --embedding_model \"$embedding_model\" \
    --train_pool_size $train_pool_size \
    --eval_pool_size $eval_pool_size \
    --test_pool_size $test_pool_size \
    --balance_dataset \
    --wandb_entity \"$wandb_entity\" \
    --wandb_project \"$wandb_project\" \
    --random_seed $random_seed_for_run \
    --task_type \"$task_type\" \
    --rl_model \"$rl_model\" \
    --search_algorithm \"$search_algorithm\" \
    --rl_task_model \"$rl_task_model\" \
    --sample_algorithm \"$sample_algorithm\" \
    --epochs $epochs \
    --actor_learning_rate $actor_learning_rate \
    --critic_learning_rate $critic_learning_rate \
    --learning_rate $learning_rate \
    --hdim $hdim \
    --hidden_dim $mil_task_head_hidden_dim \
    --epsilon $epsilon \
    --reg_coef $reg_coef \
    --gpu $mtl_gpu_id" # Pass the GPU index for run_rlmil.py's internal use if needed

# Add --reg_alg if the variable is set (it is in this case)
if [ ! -z "$reg_alg" ]; then
    PYTHON_CMD="$PYTHON_CMD --reg_alg $reg_alg"
fi

# To disable W&B for a purely local test, uncomment the next line
# PYTHON_CMD="$PYTHON_CMD --no_wandb"

# To run without sweeps (which is typical for a direct test like this)
# ensure --run_sweep is NOT in PYTHON_CMD unless you intend to trigger a sweep definition in run_rlmil.py
# The original script adds --run_sweep. For this test, we are *not* running a hyperparameter sweep,
# but a single run with fixed (chosen) hyperparameters.
# So, remove --run_sweep if it was implicitly added.
# The run_rlmil.py script structure seems to use --run_sweep to trigger wandb.agent().
# If you remove --run_sweep, it should fall into the single run logic in main().

# Let's ensure --run_sweep is NOT passed for this direct test,
# unless your run_rlmil.py is structured to use it differently for single runs.
# Assuming non-sweep execution for this test.

# Run in a screen session
SESSION_NAME="mtl_${prefix}_${mtl_baseline_type}"
FULL_COMMAND="CUDA_VISIBLE_DEVICES=$mtl_gpu_id $PYTHON_CMD"

echo "Executing in screen session '$SESSION_NAME':"
echo "$FULL_COMMAND"
echo "---------------------------------------------------------------------"

screen -dmS "$SESSION_NAME" bash -c "
  $FULL_COMMAND; \
  echo ''; \
  echo '--------------------------------------------------'; \
  echo 'MTL RL-MIL Run script finished in screen session.'; \
  echo 'Screen session $SESSION_NAME will remain active.'; \
  echo 'To detach: Ctrl+A then D'; \
  echo '--------------------------------------------------'; \
  exec bash" # Keeps the screen session alive

echo "✅ MTL RL-MIL Run launched in screen session: $SESSION_NAME"
echo "   To attach to the session, type: screen -r $SESSION_NAME"
echo "   To view GPU usage, type in another terminal: watch -n 1 nvidia-smi"
echo "---------------------------------------------------------------------"