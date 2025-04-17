import argparse
import os
import shlex
import subprocess

import torch

from logger import get_logger
from prepare_data import create_dataset
from utils import set_seed, load_json, get_model_save_directory


def parse_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=["repset", "MaxMLP", "MeanMLP", "AttentionMLP", "random", "majority", "SimpleMLP"],
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        choices=[
            "care",
            "purity",
            "sex",
            "religion",
            "customized_political_opinion",
            "party",
            "gender",
            "equality",
            "proportionality",
            "loyalty",
            "authority",
            "fairness",
            "face",
            "honor",
            "dignity"
            "age",
            "education",
            "political_orientation",
            "ladder",
            "ethnicity",
            "age",
            "hate"
        ],
    )
    parser.add_argument(
        "--bag_size", type=int, required=True, help="Bag size is required."
    )
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument(
        "--autoencoder_layer_sizes", type=str, required=False, default=None
    )
    parser.add_argument(
        "--data_embedded_column_name",
        type=str,
        default="timeline_cleaned_tweets",
        required=False,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--sweep_random_seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=False,
    )

    args = parser.parse_args()

    if args.autoencoder_layer_sizes is not None:
        # Convert string in format of int1,int2,int3 to list of ints
        args.autoencoder_layer_sizes = [
            int(x) for x in args.autoencoder_layer_sizes.split(",")
        ]

    # Add a default run_sweep argument which is False
    args.run_sweep = False

    return args


def run_command_in_screen(session_name, cmd):
    screen_command = ["screen", "-dmS", session_name, "bash", "-c", cmd, "input('Press Enter to continue...')"]
    result = subprocess.run(screen_command)
    return result


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    task_type = 'classification'
    dev=False
    # logger = get_logger(args)
    # logger.info(f"{args=}")
    runner_name = "run_mil.py"

    if args.baseline =='SimpleMLP':
        runner_name = "run_baseline.py"
        args.autoencoder_layer_sizes = None
        
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # logger.info(f"{device=}")

    sweep_run_dir = get_model_save_directory(dataset=args.dataset,
                                       data_embedded_column_name=args.data_embedded_column_name,
                                       embedding_model_name=args.embedding_model, 
                                       target_column_name=args.label,
                                       bag_size=args.bag_size, baseline=args.baseline,
                                       autoencoder_layers=args.autoencoder_layer_sizes, random_seed=args.sweep_random_seed, 
                                       dev=dev, task_type=task_type, prefix=None)
    
    mil_config = load_json(os.path.join(sweep_run_dir, "best_model_config.json"))
    
    # Get rid of "sweep_config" key in the jsons
    mil_config.pop("sweep_config", None)

    # Set run_sweep to False
    mil_config["run_sweep"] = False

    # Set the random seed for both of them
    mil_config["random_seed"] = args.random_seed

    # logger.info(f'{mil_config=}')
    # logger.info(f'{best_rl_model_config_json=}')

    # Run the best model on the dataset with the given seed save the results
    run_mil_models_python_script_command = (
        f"./venv/bin/python3 {runner_name} "
        f"--baseline {args.baseline} "
        f"--label {args.label} "
        f"--bag_size {args.bag_size} "
        f"--embedding_model {args.embedding_model} "
        f"--dataset {args.dataset} "
        f"--task_type {task_type} "
        f"--no_wandb "
        f"--batch_size {mil_config['batch_size']} "
        f"--epochs {mil_config['epochs']} "
        f"--learning_rate {mil_config['learning_rate']} "
        f"--scheduler_patience {mil_config['scheduler_patience']} "
        f"--early_stopping_patience {mil_config['early_stopping_patience']} "
        f"--data_embedded_column_name {args.data_embedded_column_name} "
        f"--random_seed {args.random_seed} "
        f"--gpu 0 "
        f"--multiple_runs "
    )

    if args.baseline !='SimpleMLP':
        run_mil_models_python_script_command += f"--autoencoder_layer_sizes {args.autoencoder_layer_sizes} "
    if mil_config['n_elements'] is not None:
        run_mil_models_python_script_command += f"--n_elements {mil_config['n_elements']} "
    if mil_config['n_hidden_sets'] is not None:
        run_mil_models_python_script_command += f"--n_hidden_sets {mil_config['n_hidden_sets']} "
    if mil_config['hidden_dim'] is not None:
        run_mil_models_python_script_command += f"--hidden_dim {mil_config['hidden_dim']} "
    if mil_config['dropout_p'] is not None:
        run_mil_models_python_script_command += f"--dropout_p {mil_config['dropout_p']} "

    # logger.info(f'{run_mil_models_python_script_command=}')
    # logger.info(f'{shlex.split(run_mil_models_python_script_command)=}')

    run_mil_models_session_name = f"{args.label}_{args.embedding_model}_{args.baseline}"
    # pass CUDA_VISIBLE_DEVICES to the subprocess as an environment variable
    run_mil_result = subprocess.run(shlex.split(run_mil_models_python_script_command),
                                    # env={"CUDA_VISIBLE_DEVICES": args.gpu}
                                    )
    # logger.info(f"{run_mil_result.returncode=}")

    if run_mil_result.returncode != 0:
        raise Exception(f"run_mil_result.returncode={run_mil_result.returncode}")

    # logger.info(f"Finished running run_mil.py for {args.label}_{args.embedding_model}_{args.baseline}")