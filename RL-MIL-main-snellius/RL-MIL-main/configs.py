import argparse
import os

from utils import load_yaml_file


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
        nargs='+',  # This allows the Python script to receive multiple labels
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
            "dignity",
            "age",
            "education",
            "political_orientation",
            "ladder",
            "ethnicity",
            "age",
            "old_age",
            "hate",
            "extraversion",
            "neuroticism",
            "agreeableness",
            "conscientiousness",
            "openness",
        ],
    )
    parser.add_argument(
        "--bag_size", type=int, required=True, help="Bag size is required."
    )
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    # Wandb arguments
    parser.add_argument(
        "--run_sweep",
        action="store_true",
    )
    parser.add_argument(
        "--rl",
        action="store_true",
    )
    parser.add_argument(
        "--multiple_runs",
        action="store_true",
    )
    parser.add_argument(
        "--no_autoencoder_for_rl",
        action="store_true",
    )
    parser.add_argument(
        "--instance_labels_column",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--rl_model",
        type=str,
        default="policy_and_value",
        choices=["policy_and_value", "policy_only"],
        required=False,
    )
    parser.add_argument(
        "--rl_task_model",
        type=str,
        default="vanilla",
        choices=["vanilla", "ensemble"],
        required=False,
    )
    parser.add_argument(
        "--only_ensemble",
        action="store_true",
    )
    parser.add_argument(
        "--balance_dataset",
        action="store_true",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required=False,
        default="mola_mil",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=False
    )

    # Wandb shared hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        required=False,
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        required=False,
    )

    # MLP parameters
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        required=False,
    )

    # RepSet parameters
    parser.add_argument(
        "--n_elements",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--n_hidden_sets",
        type=int,
        required=False,
    )

    # RL parameters
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=False,
    )
    parser.add_argument(
        "--hdim",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--train_pool_size",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--eval_pool_size",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--test_pool_size",
        type=int,
        required=False,
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        required=False,
    )

    # Optional arguments
    parser.add_argument(
        "--autoencoder_layer_sizes", type=str, required=False, default=None
    )
    parser.add_argument(
        "--data_embedded_column_name",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=False,
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--search_algorithm",
        type=str,
        default="probability",
        choices=["probability", "epsilon_greedy"],
        required=False,
    )
    parser.add_argument(
        "--reg_alg",
        type=str,
        choices=["sum"],
        required=False,
    )
    parser.add_argument(
        "--reg_coef",
        type=float,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--sample_algorithm",
        type=str,
        choices=["static", "with_replacement", "without_replacement"],
        required=False,
    )
    args = parser.parse_args()

    if args.run_sweep and args.no_wandb:
        raise ValueError("Cannot pass both --run_sweep and --no_wandb")

    if args.baseline == "random":
        # Random does not run on sweep!
        args.run_sweep = False

    if args.autoencoder_layer_sizes is not None:
        # Convert string in format of int1,int2,int3 to list of ints
        args.autoencoder_layer_sizes = [
            int(x) for x in args.autoencoder_layer_sizes.split(",")
        ]
    if args.sample_algorithm == 'static':
        args.test_pool_size = 1
        args.eval_pool_size = 1

    if args.rl:
        if 'warmup' in args.prefix and args.only_ensemble:
            raise ValueError("Not efficient to run warmup and only_ensemble together")
        if 'warmup' in args.prefix and args.rl_model == "policy_only":
            raise ValueError("Does not make sense to run warmup with policy only since the F1 is not going to change!")
        if args.only_ensemble and args.sample_algorithm != "static":
            raise ValueError("Only ensemble only works with static sampling!")
        # if args.search_algorithm == "epsilon_greedy" and args.rl_task_model == "ensemble":
        #     raise ValueError("Does not make sense to run epsilon_greedy on top of ensemble!")
        if args.search_algorithm == "epsilon_greedy":
            args.prefix = f"{args.prefix}_{args.search_algorithm}"
        if args.reg_alg is not None:
            args.prefix = f"{args.prefix}_reg_{args.reg_alg}"

    if args.run_sweep:
        args.no_wandb = False
        if args.rl:
            if "loss" in args.prefix:
                if args.only_ensemble:
                    config_file = f"hp_ensemble_loss.yaml"
                elif args.rl_model == "policy_only":
                    config_file = f"hp_rl_mtl_loss.yaml"
                else:
                    config_file = f"hp_rl_{args.prefix}.yaml"
            else:
                config_file = f"hp_rl.yaml"
            sweep_config_file_address = (
                os.path.join(
                    os.path.dirname(__file__),
                    "yaml_configs",
                    config_file
                ))
        else:
            sweep_config_file_address = (
                os.path.join(
                    os.path.dirname(__file__),
                    "yaml_configs",
                    f"hp_{args.baseline}.yaml",
                ))
        sweep_config = load_yaml_file(sweep_config_file_address)
        args.sweep_config = sweep_config

    if args.only_ensemble:
        args.prefix = f"only_ensemble_{args.prefix}"
    else:
        if args.rl_task_model == 'vanilla':
            args.prefix = f"{args.rl_model}_{args.prefix}"
        else:
            args.prefix = f"{args.rl_model}_with_{args.rl_task_model}_{args.prefix}"
        if args.no_autoencoder_for_rl:
            args.prefix = f"no_autoencoder_{args.prefix}"
        # args.prefix = "v10_" + args.prefix
        # args.prefix = f"{args.prefix}_sample_{args.sample_algorithm}"
        args.prefix = f"neg_{args.prefix}_sample_{args.sample_algorithm}"
    return args
