#!/bin/bash

cd ..
source venv/bin/activate

dataset="political_data_with_age" # or "political_data_with_age" or "facebook"
whole_bag_size=100 # or 100
num_pos_samples=5 # only used in jigsaw datasets, 5 for making `jigsaw_5` and 10 for making `jigsaw_10`

data_embedded_column_name="text" # or "text"

random_seeds=(42 43 44 45 46)
gpus=(0 1 2 3 4)

# ---- Constants ----
embedding_model="roberta-base"

for random_seed_index in "${!random_seeds[@]}"; do
    random_seed=${random_seeds[$random_seed_index]}
    gpu=${gpus[$random_seed_index]}
    SESSION_NAME="data_${dataset}_${random_seed}"
    echo $SESSION_NAME
    screen -dmS "$SESSION_NAME" bash -c "
    CUDA_VISIBLE_DEVICES=$gpu python3 prepare_data.py \
                            --dataset "$dataset" \
                            --embedding_model "$embedding_model" \
                            --data_embedded_column_name "$data_embedded_column_name" \
                            --random_seed "$random_seed" \
                            --whole_bag_size "$whole_bag_size" \
                            --num_pos_samples "$num_pos_samples" \
                            --gpu 0 ;
    exit"
done