import argparse
import os
from utils import load_yaml_file, get_model_name # Ensure load_yaml_file is in utils.py

def get_sweep_config_file_path(args):
    """
    Determines the sweep configuration YAML file path based on command-line arguments.
    This version prioritizes a dedicated clustering YAML when clustering is active.
    """
    if hasattr(args, 'sweep_config_yaml') and args.sweep_config_yaml:
        return args.sweep_config_yaml

    config_file = ""
    # --- Logic to select the correct YAML file ---
    if args.rl:
        if args.clustering_active:
            # If clustering is active, use the dedicated clustering sweep config
            config_file = "hp_rl_mtl_clust.yaml"
            print(f"Clustering is active, selecting sweep config: {config_file}")
        elif args.rl_model == 'policy_only':
            # Original logic for non-clustering policy_only models
            config_file = f"hp_rl_policy_only_loss_{args.search_algorithm}"
            if args.reg_alg:
                config_file += f"_reg_{args.reg_alg}"
            config_file += ".yaml"
            print(f"Policy-only RL, selecting sweep config: {config_file}")
        else:
            # Fallback for other RL models if needed
            config_file = "hp_rl_loss.yaml"
            print(f"Default RL, selecting sweep config: {config_file}")
    else: # Fallback for non-RL MIL sweeps
        config_file = f"hp_MeanMLP_MTL.yaml"
        print(f"Non-RL (MIL), selecting sweep config: {config_file}")

    return config_file

def parse_args():
    parser = argparse.ArgumentParser()

    # --- General Arguments ---
    general_args = parser.add_argument_group('General Arguments')
    general_args.add_argument("--random_seed", type=int, default=1, help="Random seed")
    general_args.add_argument("--gpu", type=int, default=0, help="GPU to use")
    general_args.add_argument("--prefix", type=str, default="default_run", help="Prefix for model save directory and W&B run name")
    general_args.add_argument("--dev", action="store_true", help="Run in development mode (e.g., with smaller dataset subset)")
    general_args.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    general_args.add_argument("--wandb_entity", type=str, default="stonelake-university-of-amsterdam", help="W&B entity name")
    general_args.add_argument("--wandb_project", type=str, default="RL_MIL_MTL_Clustering", help="W&B project name")
    general_args.add_argument("--multiple_runs", action="store_true", help="Flag to modify save path for multiple runs/sweeps.")

    # --- Data Arguments ---
    data_args = parser.add_argument_group('Data Arguments')
    data_args.add_argument("--dataset", type=str, default="political_data_with_age", help="Dataset to use")
    data_args.add_argument("--label", type=str, required=True, nargs='+', help="Target label(s) for prediction")
    data_args.add_argument("--data_embedded_column_name", type=str, default="text", help="Column name with embedded data")
    data_args.add_argument("--embedding_model", type=str, default="roberta-base", help="Name of the embedding model used")
    data_args.add_argument("--instance_labels_column", type=str, default=None, help="Column with instance-level labels if available")
    data_args.add_argument("--balance_dataset", action="store_true", help="Balance dataset using weighted random sampler")

    # --- Model Arguments ---
    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument("--baseline", type=str, default="MeanMLP", help="Baseline MIL model type (e.g., MeanMLP, AttentionMLP, MaxMLP)")
    model_args.add_argument("--task_type", type=str, default="classification", help="Type of task (classification or regression)")
    model_args.add_argument("--autoencoder_layer_sizes", type=str, default="768,256,768", help="Comma-separated layer sizes for autoencoder (base_network)")
    model_args.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for standard (non-clustered) MTL heads")
    model_args.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability")

    # --- Training Arguments ---
    train_args = parser.add_argument_group('Training Arguments')
    train_args.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    train_args.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_args.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for MIL model (task_model)")
    train_args.add_argument("--early_stopping_patience", type=int, default=20, help="Patience for early stopping (e.g., 15 for sweeps)")
    train_args.add_argument("--warmup_epochs", type=int, default=10, help="Number of epochs for warmup phase")
    train_args.add_argument("--scheduler_patience", type=int, default=10, help="Patience for learning rate scheduler (e.g., ReduceLROnPlateau).")

    # --- Reinforcement Learning (RL) Arguments ---
    rl_args = parser.add_argument_group('RL Arguments')
    rl_args.add_argument("--rl", action="store_true", help="Activate RL framework")
    rl_args.add_argument("--bag_size", type=int, default=20, help="Number of instances to select from each bag")
    rl_args.add_argument("--actor_learning_rate", type=float, default=1e-4, help="Learning rate for the actor network")
    rl_args.add_argument("--critic_learning_rate", type=float, default=1e-4, help="Learning rate for the critic network")
    rl_args.add_argument("--hdim", type=int, default=128, help="Hidden dimension for actor/critic networks")
    rl_args.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon-greedy exploration") # CORRECTED: Default is now a float
    rl_args.add_argument("--reg_coef", type=float, default=0.01, help="Regularization coefficient for RL loss")
    rl_args.add_argument("--train_pool_size", type=int, default=1, help="Number of selection runs for training reward estimation")
    rl_args.add_argument("--eval_pool_size", type=int, default=10, help="Number of selection runs for evaluation reward estimation")
    rl_args.add_argument("--test_pool_size", type=int, default=10, help="Number of selection runs for testing reward estimation")
    rl_args.add_argument("--rl_model", type=str, default='policy_only', help="RL model type (e.g., policy_only, policy_and_value)")
    rl_args.add_argument("--rl_task_model", type=str, default='vanilla', help="RL task model type (e.g., vanilla, ensemble)")
    rl_args.add_argument("--sample_algorithm", type=str, default='without_replacement', help="Instance sampling algorithm")
    rl_args.add_argument("--search_algorithm", type=str, default='epsilon_greedy', help="Search algorithm for instance selection")
    rl_args.add_argument("--reg_alg", type=str, default='sum', help="Regularization algorithm component for prefix/config selection")
    rl_args.add_argument("--no_autoencoder_for_rl", action="store_true", help="Disable using autoencoder for RL state representation")
    rl_args.add_argument("--only_ensemble", action="store_true", help="Run only ensemble model (random selection)")

    # --- Task Clustering Arguments ---
    cluster_args = parser.add_argument_group('Task Clustering Arguments')
    cluster_args.add_argument("--clustering_active", action="store_true", help="Activate task clustering.")
    cluster_args.add_argument("--num_task_clusters", type=int, default=2, help="Number of task clusters (K) for K-Means.")
    cluster_args.add_argument("--hidden_dim_cluster_trunk", type=int, default=128, help="Hidden dimension for shared cluster MLP trunks.")
    cluster_args.add_argument("--hidden_dim_final_head", type=int, default=64, help="Hidden dimension for final task-specific layers.")
    cluster_args.add_argument("--recluster_interval_E", type=int, default=10, help="Epoch interval for re-clustering tasks.")
    cluster_args.add_argument("--kmeans_random_state", type=int, default=42, help="Random state for KMeans for reproducibility.")

    # --- Sweep Arguments ---
    sweep_args = parser.add_argument_group('Sweep Arguments')
    sweep_args.add_argument("--run_sweep", action="store_true", help="Run a W&B sweep")
    sweep_args.add_argument("--sweep_config_yaml", type=str, default=None, help="Explicitly specify the sweep YAML file to use")

    args = parser.parse_args()

    # --- Process and Derive Arguments ---
    if isinstance(args.autoencoder_layer_sizes, str):
        args.autoencoder_layer_sizes = [int(i) for i in args.autoencoder_layer_sizes.split(",")]
    else: # If it's None or already a list
        args.autoencoder_layer_sizes = args.autoencoder_layer_sizes

    args.model_name = get_model_name(baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes)

    # Dynamically create prefix for RL runs if not provided with a specific structure
    if args.rl and "neg" not in args.prefix and "rl" not in args.prefix:
        temp_prefix = args.rl_model
        if args.rl_task_model == 'vanilla':
            temp_prefix = f"{temp_prefix}_{args.prefix}"
        temp_prefix = f"{temp_prefix}_{args.search_algorithm}"
        if args.reg_alg:
            temp_prefix = f"{temp_prefix}_reg_{args.reg_alg}"
        temp_prefix = f"neg_{temp_prefix}_sample_{args.sample_algorithm}"
        args.prefix = temp_prefix

    # Load sweep config from YAML if running a sweep
    if args.run_sweep:
        sweep_config_filename = get_sweep_config_file_path(args)
        sweep_config_file_address = os.path.join(os.path.dirname(__file__), "yaml_configs", sweep_config_filename)
        if os.path.exists(sweep_config_file_address):
            args.sweep_config = load_yaml_file(sweep_config_file_address)
            if not isinstance(args.sweep_config, dict):
                raise TypeError(f"Loaded sweep config from {sweep_config_file_address} is not a dictionary. Check YAML format.")
        else:
            raise FileNotFoundError(f"Could not find required sweep config file: {sweep_config_file_address}")

    # Add a flag to indicate if this is an MTL run
    args.is_mtl = isinstance(args.label, list) and len(args.label) > 1

    return args