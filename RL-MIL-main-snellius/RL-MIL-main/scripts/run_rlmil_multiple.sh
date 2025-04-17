#!/bin/bash

cd ..
source venv/bin/activate

datasets=("political_data_with_age" "facebook" "jigsaw_5" "jigsaw_10")

baselines=("MeanMLP" "MaxMLP" "AttentionMLP" "repset")

random_seeds=(0 1 2 3 4 5 6 7 8 9)
gpus=(0 1 2 3 4 5 6 7)

# ---- Constants ----
facebook_labels=("care" "purity" "loyalty" "authority" "fairness")
facebook_embedding_models=("roberta-base")
facebook_embedded_column_name="text"

jigsaw_labels=("hate")
jigsaw_embedding_models=("roberta-base")
jigsaw_embedded_column_name="comment_text"

political_labels=("age" "gender" "party")
political_embedding_models=("roberta-base")
political_embedded_column_name="text"

sweep_random_seed=0

# For each dataset, each baseline, each label, embedding_model, and seed
for dataset in "${datasets[@]}"; do
    if [ "$dataset" = "yourmorals_incas" ]; then
        labels=("${yourmorals_incas_labels[@]}")
        embedding_models=("${yourmorals_incas_embedding_models[@]}")
        embedded_column_name=$yourmorals_incas_embedded_column_name
    elif [ "$dataset" = "political_data_with_age" ]; then
        labels=("${political_labels[@]}")
        embedding_models=("${political_embedding_models[@]}")
        embedded_column_name=$political_embedded_column_name
    elif [ "$dataset" = "facebook" ]; then
        labels=("${facebook_labels[@]}")
        embedding_models=("${facebook_embedding_models[@]}")
        embedded_column_name=$facebook_embedded_column_name
    elif [ "$dataset" = "jigsaw_10" ]; then
        labels=("${jigsaw_labels[@]}")
        embedding_models=("${jigsaw_embedding_models[@]}")
        embedded_column_name=$jigsaw_embedded_column_name
    elif [ "$dataset" = "jigsaw_5" ]; then
        labels=("${jigsaw_labels[@]}")
        embedding_models=("${jigsaw_embedding_models[@]}")
        embedded_column_name=$jigsaw_embedded_column_name
    fi

    for baseline in "${baselines[@]}"; do
        for label in "${labels[@]}"; do
            for embedding_model in "${embedding_models[@]}"; do
               for random_seed_index in "${!random_seeds[@]}"; do
                    random_seed=${random_seeds[$random_seed_index]}
                    gpu=${gpus[$random_seed_index]}

                    SESSION_NAME="rl_${random_seed}_${dataset}_${baseline}_${label}"
                    echo $SESSION_NAME
                    screen -dmS "$SESSION_NAME" bash -c "
                    CUDA_VISIBLE_DEVICES=$gpu python3 multiple_run_rlmil.py \
                                                    --dataset "$dataset" \
                                                    --baseline "$baseline" \
                                                    --label "$label" \
                                                    --embedding_model "$embedding_model" \
                                                    --bag_size 20 \
                                                    --autoencoder_layer_sizes "768,256,768" \
                                                    --data_embedded_column_name "$embedded_column_name" \
                                                    --random_seed "$random_seed" \
                                                    --gpu 0 \
                                                    --sweep_random_seed "$sweep_random_seed"
                                                                                    "
                done
            done
        done
    done
done