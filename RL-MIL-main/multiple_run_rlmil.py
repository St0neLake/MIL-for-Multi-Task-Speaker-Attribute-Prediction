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
        choices=["repset", "MaxMLP", "MeanMLP", "AttentionMLP", "random", "majority"],
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
    dev = False
    prefix = "loss"
    # logger = get_logger(args)
    # logger.info(f"{args=}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # logger.info(f"{device=}")

    sweep_run_dir = get_model_save_directory(dataset=args.dataset,
                                       data_embedded_column_name=args.data_embedded_column_name,
                                       embedding_model_name=args.embedding_model, 
                                       target_column_name=args.label,
                                       bag_size=args.bag_size, baseline=args.baseline,
                                       autoencoder_layers=args.autoencoder_layer_sizes, random_seed=args.sweep_random_seed, 
                                       dev=dev, task_type=task_type, prefix=None)
    
    # mil_config = load_json(os.path.join(sweep_run_dir, "best_model_config.json"))
    directories = [name for name in os.listdir(sweep_run_dir) if os.path.isdir(os.path.join(sweep_run_dir, name))]
    directories = ['only_ensemble_loss'] + [name for name in directories if name.startswith('neg')]

    for directory in directories:
        # print(directory)
        config = load_json(os.path.join(sweep_run_dir, directory, "sweep_best_model_config.json"))
        # Get rid of "sweep_config" key in the jsons
        for k, dict in config['sweep_config']['parameters'].items():
            if dict['distribution'] == 'constant':
                config[k] = dict['value']
        
        config.pop("sweep_config", None)

        # Set run_sweep to False
        config["run_sweep"] = False

        # Set the random seed for both of them
        config["random_seed"] = args.random_seed
    
        # Do the same for RL models
        run_rl_models_python_script_command = (
            f"./venv/bin/python3 run_rlmil.py "
            f"--baseline {args.baseline} "
            f"--label {args.label} "
            f"--bag_size {args.bag_size} "
            f"--embedding_model {args.embedding_model} "
            f"--dataset {args.dataset} "
            f"--rl "
            f"--no_wandb "
            f"--batch_size {config['batch_size']} "
            f"--epochs {config['epochs']} "
            f"--learning_rate {config['learning_rate']} "
            f"--scheduler_patience {config['scheduler_patience']} "
            f"--early_stopping_patience {config['early_stopping_patience']} "
            f"--actor_learning_rate {config['actor_learning_rate']} "
            f"--critic_learning_rate {config['critic_learning_rate']} "
            f"--hdim {config['hdim']} "
            f"--train_pool_size {config['train_pool_size']} "
            f"--eval_pool_size {config['eval_pool_size']} "
            f"--test_pool_size {config['test_pool_size']} "
            f"--autoencoder_layer_sizes {','.join([str(x) for x in config['autoencoder_layer_sizes']])} "
            f"--data_embedded_column_name {args.data_embedded_column_name} "
            f"--random_seed {args.random_seed} "
            f"--gpu 0 "
            f"--prefix {prefix} "
            f"--task_type {task_type} "
            f"--rl_model {config['rl_model']} "
            f"--sample_algorithm {config['sample_algorithm']} "
            f"--rl_task_model {config['rl_task_model']} "
            f"--search_algorithm {config['search_algorithm']} "
            f"--reg_coef {config['reg_coef']} "
            f"--epsilon {config['epsilon']} "
            f"--multiple_runs "
        )

        # if warmup_epochs key was in the best_mil_model_config_json, then add it to the run_rl_models_python_script_command
        if 'warmup_epochs' in config.keys():
            run_rl_models_python_script_command += f"--warmup_epochs {config['warmup_epochs']} "

        if config['only_ensemble']:
            run_rl_models_python_script_command += f"--only_ensemble "
        if config['reg_alg']:
            run_rl_models_python_script_command += f"--reg_alg {config['reg_alg']} "
        
        if config['no_autoencoder_for_rl']:
            run_rl_models_python_script_command += f"--no_autoencoder_for_rl "
        
        run_rl_models_session_name = f"rl_{args.dataset}_{args.label}_{args.baseline}"
        # pass CUDA_VISIBLE_DEVICES to the subprocess as an environment variable
        run_rl_result = subprocess.run(shlex.split(run_rl_models_python_script_command),
                                    # env={"CUDA_VISIBLE_DEVICES": args.gpu}
                                    )

        if run_rl_result.returncode != 0:
            raise Exception(f"run_rl_result.returncode={run_rl_result.returncode}")

