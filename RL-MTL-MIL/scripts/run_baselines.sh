#!/bin/bash

cd ..
source venv/bin/activate

# For facebook dataset: ("care" "purity" "loyalty" "authority" "fairness")
# For political_data_with_age dataset: ("age" "gender" "party")
# For jigsaw datasets: ("hate")
target_labels=("hate")

gpus=(0 1 2 3 4 5 6 7)

# wandb config
wandb_entity="YOUR_WANDB_ENTITY"
wandb_project="YOUR_WANDB_PROJECT_NAME"

# Dataset is either: `political_data_with_age,` `facebook,` `jigsaw_5,` or `jigsaw_10`
dataset="jigsaw_5"

# For `facebook` and `political_data_with_age` datasets: "text"
# For `jigsaw` datasets: "comment_text"
data_embedded_column_name="comment_text"

# ---- Constants ----
baseline_type="SimpleMLP"

task_type="classification"

# autoencoder_layer_sizes should be a string of comma-separated integers. We used "768,256,768" in all experiments.
autoencoder_layer_sizes="768,256,768"

# The size of b_i in the paper. We used 20 in all experiments.
bag_sizes=(20)

embedding_models=("roberta-base")

# Get the total number of runs
total_runs=$((${#target_labels[@]} * ${#bag_sizes[@]} * ${#embedding_models[@]}))
current_run=1

# Loop over all combinations of baseline_type and target_label and embedding_model
for target_label_index in "${!target_labels[@]}"; do
  for bag_size_index in "${!bag_sizes[@]}"; do
    for embedding_model_index in "${!embedding_models[@]}"; do
        # Get current target label
        target_label=${target_labels[$target_label_index]}
        # Get current bag_size
        bag_size=${bag_sizes[$bag_size_index]}
        # Get current embedding_model
        embedding_model=${embedding_models[$embedding_model_index]}
        # Get current gpu
        gpu=${gpus[$target_label_index]}
        echo "$baseline_type, $dataset $target_label, bag_size_$bag_size, $embedding_model, gpu_$gpu ($current_run/$total_runs)"

        SESSION_NAME="${dataset}_${target_label}_${baseline_type}"

        # # only MIL models running 
        screen -dmS "$SESSION_NAME" bash -c "
        CUDA_VISIBLE_DEVICES=$gpu python3 run_baseline.py \
                                      --baseline "$baseline_type" \
                                      --label "$target_label" \
                                      --bag_size "$bag_size" \
                                      --embedding_model $embedding_model \
                                      --dataset $dataset \
                                      --run_sweep \
                                      --wandb_entity $wandb_entity \
                                      --wandb_project $wandb_project \
                                      --autoencoder_layer_sizes $autoencoder_layer_sizes \
                                      --data_embedded_column_name $data_embedded_column_name \
                                      --task_type $task_type \
                                      --random_seed 0 ;
        exit"

        ((current_run++))
    done
  done
done
