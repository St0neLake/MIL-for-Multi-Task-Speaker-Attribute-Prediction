import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
import argparse
import traceback

from configs import parse_args
from RLMIL_Datasets import RLMILDataset
from logger import get_logger
from models import PolicyNetwork, sample_action, select_from_action, create_mil_model_with_dict, ClusteredMeanMLP, ClusteredMaxMLP
from utils import (
    get_data_directory,
    get_model_name,
    get_model_save_directory,
    read_data_split,
    preprocess_dataframe,
    create_preprocessed_dataframes,
    get_task_representations_from_activations,
    calculate_task_similarity_matrix,
    assign_tasks_to_clusters,
    get_df_mean_median_std,
    get_balanced_weights,
    load_yaml_file,
    EarlyStopping, save_json, load_json, create_mil_model
)

# Global variables
logger = None
DEVICE = None
BEST_REWARD = float("-inf")
args = None
global_run_dir = None

def finish_episode_policy_only(
        policy_network, train_dataloader, eval_dataloader, optimizer, device,
        bag_size, train_pool_size, scheduler, warmup, only_ensemble,
        epsilon, reg_coef, sample_algorithm,
        current_task_cluster_assignment: dict
):
    policy_network.eval()
    eval_pool = policy_network.create_pool_data(eval_dataloader, bag_size, train_pool_size, random=only_ensemble)

    sel_losses, regularization_losses = [], []
    policy_network.reset_reward_action()

    for batch_x, batch_y_dict, _, _  in train_dataloader:
        policy_network.train()
        batch_x = batch_x.to(device)

        action_probs, _, _ = policy_network(batch_x)

        action, action_log_prob = sample_action(
            action_probs,
            bag_size,
            device=device,
            random=(epsilon > np.random.random()) or only_ensemble,
            algorithm=sample_algorithm
        )
        sel_x = select_from_action(action, batch_x)

        # Pass current_task_cluster_assignment to train_minibatch
        sel_loss = policy_network.train_minibatch(sel_x, batch_y_dict, current_task_cluster_assignment)
        sel_losses.append(sel_loss)

        policy_network.eval() # For reward computation
        if not only_ensemble:
            # Pass current_task_cluster_assignment to expected_reward_loss
            reward, _, _ = policy_network.expected_reward_loss(eval_pool, current_task_cluster_assignment)
            policy_network.saved_actions.append(action_log_prob)
            policy_network.rewards.append(reward)
            if action_probs.dim() > 1 and action_probs.shape[1] > 0:
                regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))
            else:
                regularization_losses.append(torch.tensor(0.0).to(device))

    if only_ensemble:
        return 0, 0, 0, np.mean(sel_losses) if sel_losses else 0, 0

    if not policy_network.rewards or not policy_network.saved_actions:
        logger.warning("No rewards or saved_actions recorded in finish_episode_policy_only. Skipping RL update.")
        return 0, 0, 0, np.mean(sel_losses) if sel_losses else 0, 0

    policy_network.normalize_rewards(eps=1e-5)

    policy_losses_tensors = []
    for log_prob, reward_val in zip(policy_network.saved_actions, policy_network.rewards):
        policy_losses_tensors.append(-reward_val * log_prob.to(device)) # log_prob should already be on device if action_probs was

    optimizer.zero_grad()

    if not policy_losses_tensors:
        logger.warning("policy_losses_tensors is empty in finish_episode_policy_only. Skipping RL update.")
        return 0,0,0, np.mean(sel_losses) if sel_losses else 0, 0

    policy_loss_agg = torch.cat(policy_losses_tensors).mean()

    regularization_loss_val = torch.tensor(0.0).to(device)
    if regularization_losses:
        valid_reg_losses = [l for l in regularization_losses if isinstance(l, torch.Tensor)]
        if valid_reg_losses:
            regularization_loss_val = torch.stack(valid_reg_losses).mean() / 100.0 # Original scaling

    total_loss = policy_loss_agg + reg_coef * regularization_loss_val
    total_loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    policy_network.reset_reward_action()

    return total_loss.item(), policy_loss_agg.item(), 0, \
           np.mean(sel_losses) if sel_losses else 0, \
           (reg_coef * regularization_loss_val.item()) if regularization_losses and valid_reg_losses else 0

def finish_episode(
        policy_network,
        train_dataloader,
        eval_dataloader,
        optimizer,
        device,
        bag_size,
        train_pool_size,
        scheduler,
        warmup,
        only_ensemble,
        epsilon,
        reg_coef,
        sample_algorithm
):
    # Get one selection of eval data for computing reward
    policy_network.eval()
    eval_pool = policy_network.create_pool_data(eval_dataloader, bag_size, train_pool_size, random=only_ensemble)
    sel_losses, regularization_losses_policy = [], []

    # To store components for actor and critic losses
    log_probs_list = []
    exp_rewards_list = []
    actual_rewards_list = []

    for batch_x, batch_y_dict, indices, _ in train_dataloader:
        policy_network.train() # Set actor and critic to train mode
        batch_x = batch_x.to(device)
        batch_y_dict_device = {k: v.to(device) for k, v in batch_y_dict.items()}
        action_probs, _, exp_reward_for_batch = policy_network(batch_x)

        action, action_log_prob = sample_action(
            action_probs,
            bag_size,
            device=device,
            random=(epsilon > np.random.random()) or only_ensemble,
            algorithm=sample_algorithm
        )

        if not warmup: # Typically, task_model (MIL part) is trained only after warmup
            sel_x = select_from_action(action, batch_x)
            # sel_y is batch_y_dict_device
            sel_loss = policy_network.train_minibatch(sel_x, batch_y_dict_device)
            sel_losses.append(sel_loss)
        else:
            sel_losses.append(0) # No MIL training during warmup

        policy_network.eval() # Eval mode for reward computation
        if not only_ensemble:
            # reward_from_eval is the scalar combined reward (e.g. avg F1 over tasks)
            reward_from_eval, _, _ = policy_network.expected_reward_loss(eval_pool)

            # Store for policy and value loss calculation later
            log_probs_list.append(action_log_prob)
            exp_rewards_list.append(exp_reward_for_batch) # Critic's prediction for this batch_x state
            actual_rewards_list.append(reward_from_eval) # Actual (scalar) reward obtained

            regularization_losses_policy.append(action_probs.sum(dim=-1).mean(dim=-1)) # Policy regularization

    if only_ensemble: # If only running ensemble (random selection), no policy/critic update
        return 0, 0, 0, np.mean(sel_losses) if sel_losses else 0, 0

    # Calculate losses for actor (policy) and critic (value function)
    policy_losses_batch = []
    value_losses_batch = []

    policy_network.train() # Back to train for updates

    for log_prob, exp_reward_tensor, actual_reward_scalar in zip(log_probs_list, exp_rewards_list, actual_rewards_list):
        advantage = actual_reward_scalar - exp_reward_tensor.detach()

        policy_losses_batch.append(-log_prob.to(device) * advantage)
        R_tensor = torch.tensor([actual_reward_scalar] * len(exp_reward_tensor), device=device).float()
        value_losses_batch.append(F.smooth_l1_loss(exp_reward_tensor, R_tensor, reduction="mean")) # Use mean reduction per batch value loss

    optimizer.zero_grad() # Zero gradients for actor and critic optimizers

    # Sum up losses
    policy_loss_total = torch.cat(policy_losses_batch).mean() # Mean over all selected actions in epoch
    value_loss_total = torch.stack(value_losses_batch).mean()    # Mean over all critic estimates in epoch

    regularization_loss_total = torch.stack(regularization_losses_policy).mean() / 100.0 # Original scaling

    # Combined loss for actor-critic update
    total_update_loss = policy_loss_total + value_loss_total + reg_coef * regularization_loss_total

    total_update_loss.backward() # Backpropagate
    optimizer.step()           # Update actor and critic parameters

    if scheduler is not None:
        scheduler.step()

    policy_network.reset_reward_action() # Clear stored actions and rewards

    return total_update_loss.item(), policy_loss_total.item(), value_loss_total.item(), \
        np.mean(sel_losses) if sel_losses else 0, \
        (reg_coef * regularization_loss_total.item()) if regularization_losses_policy else 0

def prepare_data(args):
    logger.info(f"Prepare datasets: DATA={args.dataset}, Column={args.data_embedded_column_name}, Labels={args.label}")
    data_dir = get_data_directory(args.dataset, args.data_embedded_column_name, args.random_seed)
    if not os.path.exists(data_dir):
        raise ValueError("Data directory does not exist.")

    train_dataframe = read_data_split(data_dir, args.embedding_model, "train")
    val_dataframe = read_data_split(data_dir, args.embedding_model, "val")
    test_dataframe = read_data_split(data_dir, args.embedding_model, "test")

    logger.info(f"Current args.label before create_preprocessed_dataframes: {args.label}")
    # In run_rlmil.py, inside prepare_data()
    logger.info(f"Train DataFrame columns: {train_dataframe.columns}")
    logger.info(f"Train DataFrame info for target labels ({args.label}):")
    train_dataframe[args.label].info(verbose=True, show_counts=True) # DEBUG

    train_dataframe_processed, val_dataframe_processed, test_dataframe_processed, label2id_map, id2label_map = create_preprocessed_dataframes(
        train_dataframe, val_dataframe, test_dataframe,
        target_labels=args.label,
        task_type=args.task_type,
        extra_columns=[args.instance_labels_column] if args.instance_labels_column else []
    )

    args.label2id_map = label2id_map # e.g. {'age': {'young':0,...}, 'gender': {'M':0,...}}
    args.id2label_map = id2label_map

    args.output_dims_dict = {}
    for task_name, l2i_map in args.label2id_map.items():
        args.output_dims_dict[task_name] = len(l2i_map)
    logger.info(f"Output dimensions for tasks: {args.output_dims_dict}")

    # RLMILDataset will now use the 'labels' column from processed dataframes,
    # which contains a dictionary of labels.
    train_dataset = RLMILDataset(
        df=train_dataframe_processed,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )
    val_dataset = RLMILDataset(
        df=val_dataframe_processed,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )
    test_dataset = RLMILDataset(
        df=test_dataframe_processed,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )

    return train_dataset, val_dataset, test_dataset

def create_rl_model(trial_args, mil_best_model_dir_for_task_model_checkpoint, initial_task_cluster_assignment: dict):
    task_model_config_path = os.path.join(mil_best_model_dir_for_task_model_checkpoint, "..", "best_model_config.json")
    task_model_state_dict_path = os.path.join(mil_best_model_dir_for_task_model_checkpoint, "..", "best_model.pt") # For base_network

    config_for_task_model_structure = {}
    if os.path.exists(task_model_config_path):
        config_for_task_model_structure = load_json(task_model_config_path)
        logger.info(f"Loaded base task_model structure config from: {task_model_config_path}")

    # Ensure common args are set for both model types
    config_for_task_model_structure['input_dim'] = trial_args.input_dim
    config_for_task_model_structure['autoencoder_layer_sizes'] = trial_args.autoencoder_layer_sizes
    config_for_task_model_structure['output_dims_dict'] = trial_args.output_dims_dict
    config_for_task_model_structure['dropout_p'] = trial_args.dropout_p
    config_for_task_model_structure['baseline'] = trial_args.baseline # e.g., MeanMLP for ClusteredMeanMLP

    if trial_args.clustering_active and trial_args.num_task_clusters > 0:
        logger.info(f"Creating Clustered Task Model (e.g., ClusteredMeanMLP) with {trial_args.num_task_clusters} clusters.")
        # Add clustering-specific args
        config_for_task_model_structure['hidden_dim_cluster_trunk'] = trial_args.hidden_dim_cluster_trunk
        config_for_task_model_structure['hidden_dim_final_head'] = trial_args.hidden_dim_final_head
        config_for_task_model_structure['num_clusters'] = trial_args.num_task_clusters

        # Assuming ClusteredMeanMLP is the target if baseline is MeanMLP etc.
        if trial_args.baseline == "MeanMLP": # Or your specific clustered model name
            task_model_instance = ClusteredMeanMLP(**config_for_task_model_structure)
        else:
            logger.warning(f"Clustering active, but baseline {trial_args.baseline} doesn't have a specific Clustered variant. Falling back to standard MTL model.")
            # Fallback to standard MTL model (ensure create_mil_model_with_dict handles this)
            config_for_task_model_structure['hidden_dim'] = trial_args.hidden_dim # Use general hidden_dim
            task_model_instance = create_mil_model_with_dict(config_for_task_model_structure)
    else:
        logger.info("Creating Standard MTL Task Model (no clustering or K=0).")
        config_for_task_model_structure['hidden_dim'] = trial_args.hidden_dim # Use general hidden_dim for heads
        task_model_instance = create_mil_model_with_dict(config_for_task_model_structure)

    # Load pre-trained base_network weights if they exist and are compatible
    if os.path.exists(task_model_state_dict_path) and hasattr(task_model_instance, 'base_network'):
        base_mil_model_state_dict = torch.load(task_model_state_dict_path, map_location=torch.device("cpu"))
        base_network_weights = {
            k.replace("base_network.", ""): v
            for k, v in base_mil_model_state_dict.items()
            if k.startswith("base_network.")
        }
        if base_network_weights:
            task_model_instance.base_network.load_state_dict(base_network_weights, strict=False)
            logger.info(f"Loaded pre-trained weights into base_network of {type(task_model_instance).__name__}.")
        else:
            logger.info(f"No 'base_network.*' weights found in checkpoint {task_model_state_dict_path} for {type(task_model_instance).__name__}.")
    else:
        logger.info(f"No pre-trained base_network checkpoint found or task_model has no base_network attribute. Initializing from scratch.")

    policy_network = PolicyNetwork(
        task_model=task_model_instance,
        state_dim=trial_args.state_dim,
        hdim=trial_args.hdim,
        learning_rate=trial_args.learning_rate,
        device=DEVICE,
        task_type=trial_args.task_type,
        min_clip=getattr(trial_args, 'min_clip', None),
        max_clip=getattr(trial_args, 'max_clip', None),
        sample_algorithm=trial_args.sample_algorithm,
        no_autoencoder_for_rl=trial_args.no_autoencoder_for_rl,
        epsilon=trial_args.epsilon
    )
    return policy_network

def create_rl_model_clustered(trial_args, mil_base_model_checkpoint_dir: str):
    logger.info(f"Creating RL model. Attempting to load base_network config/checkpoint from: {mil_base_model_checkpoint_dir}")

    task_model_config_path = os.path.join(mil_base_model_checkpoint_dir, "best_model_config.json")
    task_model_state_dict_path = os.path.join(mil_base_model_checkpoint_dir, "best_model.pt")

    base_config_for_task_model = {}
    if os.path.exists(task_model_config_path):
        base_config_for_task_model = load_json(task_model_config_path)
        logger.info(f"Loaded base config for task_model structure from: {task_model_config_path}")
    else:
        logger.warning(f"Base task_model config not found at {task_model_config_path}. Using trial_args for structure.")

    # Override/ensure necessary fields from trial_args for the task_model structure
    base_config_for_task_model['input_dim'] = trial_args.input_dim
    base_config_for_task_model['autoencoder_layer_sizes'] = base_config_for_task_model.get('autoencoder_layer_sizes', trial_args.autoencoder_layer_sizes)
    base_config_for_task_model['output_dims_dict'] = trial_args.output_dims_dict
    base_config_for_task_model['dropout_p'] = trial_args.dropout_p
    base_config_for_task_model['baseline'] = trial_args.baseline

    task_model_instance = None
    # Check if clustering should be activated
    if trial_args.clustering_active and trial_args.num_task_clusters > 0 and \
       trial_args.num_task_clusters < len(trial_args.label if isinstance(trial_args.label, list) else [trial_args.label]):

        logger.info(f"Creating Clustered Task Model (base: {trial_args.baseline}) with {trial_args.num_task_clusters} clusters.")

        clustered_config = {
            'input_dim': trial_args.input_dim,
            'autoencoder_layer_sizes': base_config_for_task_model['autoencoder_layer_sizes'],
            'hidden_dim_cluster_trunk': trial_args.hidden_dim_cluster_trunk,
            'hidden_dim_final_head': trial_args.hidden_dim_final_head,
            'output_dims_dict': trial_args.output_dims_dict,
            'num_clusters': trial_args.num_task_clusters,
            'dropout_p': trial_args.dropout_p
        }

        # Instantiate the correct clustered model based on the baseline type
        if trial_args.baseline == "MeanMLP":
            task_model_instance = ClusteredMeanMLP(**clustered_config)
        elif trial_args.baseline == "MaxMLP":
            task_model_instance = ClusteredMaxMLP(**clustered_config)
        # Add elif for ClusteredAttentionMLP here if you create it
        else:
            logger.error(f"Clustering active, but no Clustered variant for baseline {trial_args.baseline}. Defaulting to standard MTL.")
            base_config_for_task_model['hidden_dim'] = trial_args.hidden_dim
            task_model_instance = create_mil_model_with_dict(base_config_for_task_model)

    else:
        # Fallback to standard (non-clustered) multi-headed MIL model
        logger.info("Creating Standard Multi-Headed MIL Task Model (clustering not active or K is trivial).")
        base_config_for_task_model['hidden_dim'] = trial_args.hidden_dim
        task_model_instance = create_mil_model_with_dict(base_config_for_task_model)

    # Load pre-trained base_network weights from the MIL stage if the checkpoint exists
    if os.path.exists(task_model_state_dict_path) and hasattr(task_model_instance, 'base_network'):
        loaded_state_dict = torch.load(task_model_state_dict_path, map_location=torch.device("cpu"))
        # Extract only the base_network parameters from the saved MIL model
        base_network_state_dict = {
            k.replace("base_network.", "", 1): v
            for k, v in loaded_state_dict.items()
            if k.startswith("base_network.")
        }
        if base_network_state_dict:
            missing_keys, unexpected_keys = task_model_instance.base_network.load_state_dict(base_network_state_dict, strict=False)
            if missing_keys: logger.warning(f"Missing keys when loading base_network state_dict: {missing_keys}")
            if unexpected_keys: logger.warning(f"Unexpected keys when loading base_network state_dict: {unexpected_keys}")
            logger.info(f"Loaded pre-trained weights into base_network of {type(task_model_instance).__name__} from {task_model_state_dict_path}.")
        else:
            logger.warning(f"No 'base_network.*' weights found in checkpoint {task_model_state_dict_path} for {type(task_model_instance).__name__}.")
    else:
        logger.info(f"No pre-trained base_network checkpoint found at {task_model_state_dict_path} or model has no 'base_network' attribute. Initializing from scratch.")

    # Create the final PolicyNetwork with the configured task_model
    policy_network = PolicyNetwork(
        task_model=task_model_instance,
        state_dim=trial_args.state_dim,
        hdim=trial_args.hdim,
        learning_rate=trial_args.learning_rate, # For task_model's own optimizer
        device=DEVICE,
        task_type=trial_args.task_type,
        min_clip=getattr(trial_args, 'min_clip', None),
        max_clip=getattr(trial_args, 'max_clip', None),
        sample_algorithm=trial_args.sample_algorithm,
        no_autoencoder_for_rl=trial_args.no_autoencoder_for_rl,
        epsilon=trial_args.epsilon
    )
    return policy_network

def load_mil_model_from_config(mil_config_file, state_dict):
    mil_config = load_json(mil_config_file)
    task_model = create_mil_model_with_dict(mil_config)
    task_model.load_state_dict(state_dict)

    return task_model

def load_model_from_config(mil_config_file, rl_config_file, rl_model_file):
    # TODO: make create_mil_model compatible with dictionary input
    mil_config = load_json(mil_config_file)
    rl_config = load_json(rl_config_file)
    task_model = create_mil_model(mil_config)
    policy_network = PolicyNetwork(task_model=task_model,
                                   state_dim=rl_config['state_dim'],
                                   hdim=rl_config['hdim'],
                                   learning_rate=0,
                                   device="cuda:0" if torch.cuda.is_available() else "cpu")
    policy_network.load_state_dict(torch.load(rl_model_file))

    return policy_network

def predict(policy_network, dataloader, bag_size=20, pool_size=10):
    pool_data = policy_network.create_pool_data(dataloader, bag_size, pool_size)
    preds = policy_network.predict_pool(pool_data)

    return preds

def get_first_batch_info(
        policy_network,
        eval_dataloader,
        device,
        bag_size,
        sample_algorithm,
        args_config,
        task_to_cluster_id_map: dict
):
    """
    Logs diagnostic information from the first batch of the evaluation dataloader.
    """
    log_dict = {}
    try:
        # Get the first batch from the dataloader
        batch_x, batch_y_dict, indices, instance_labels_from_loader = next(iter(eval_dataloader))
    except StopIteration:
        logger.warning("get_first_batch_info: eval_dataloader is empty. Skipping.")
        return log_dict # Return empty dict if no data

    batch_x = batch_x.to(device)

    with torch.no_grad():
        policy_network.eval()
        action_probs, _, _ = policy_network(batch_x) # Get action probabilities from the actor
        action, _ = sample_action(action_probs, bag_size, device, random=False, algorithm=sample_algorithm)

    # Log histograms of actor probabilities and selected actions for the first few bags in the batch
    for i in range(min(action_probs.shape[0], 3)): # Log for up to 3 samples for brevity
        if action_probs[i].numel() > 0:
             log_dict.update({
                f"actor/probs_sample_{i}": wandb.Histogram(action_probs[i].cpu().detach().numpy()),
                f"actor/action_sample_{i}": wandb.Histogram(action[i].cpu().numpy().tolist())
            })

    # Optional: If you wanted to log the clustered task_model's output for this selected batch
    # sel_x = select_from_action(action, batch_x)
    # with torch.no_grad():
    #    task_model_outputs_dict = policy_network.task_model(sel_x, task_to_cluster_id_map)
    #    # Now you could log something from task_model_outputs_dict if needed

    # Your existing logic for logging instance labels
    if args_config.instance_labels_column is not None and instance_labels_from_loader is not None and len(instance_labels_from_loader) > 0:
        instance_labels_tensor = instance_labels_from_loader.to(device)
        try:
            if instance_labels_tensor.shape[1] == batch_x.shape[1]:
                selected_instance_values = instance_labels_tensor[torch.arange(action.shape[0]).unsqueeze(1), action]
                selected_instance_sum = selected_instance_values.sum(dim=1)

                for i in range(min(batch_x.shape[0], 3)):
                    log_dict.update({f"actor/selected_instance_sum_sample_{i}": selected_instance_sum[i].item()})
            else:
                logger.warning(f"Shape mismatch for instance_labels_tensor {instance_labels_tensor.shape} and batch_x {batch_x.shape} in get_first_batch_info.")
        except IndexError as e:
            logger.error(f"IndexError in get_first_batch_info instance logging: {e}. Action shape: {action.shape}, Instance labels shape: {instance_labels_tensor.shape}")
        except Exception as e:
            logger.error(f"Other error in get_first_batch_info instance logging: {e}")

    return log_dict

def train(
        policy_network, optimizer, scheduler, early_stopping,
        train_dataloader, eval_dataloader, test_dataloader,
        device, bag_size, epochs, no_wandb,
        train_pool_size, eval_pool_size, test_pool_size,
        rl_model_type, prefix, epsilon_rl, reg_coef_rl, sample_algorithm_rl,
        args_config,
        warmup_epochs=0, run_name=None, only_ensemble=False
):
    global BEST_REWARD
    global logger

    if rl_model_type == 'policy_only':
        episode_function = finish_episode_policy_only
    elif rl_model_type == 'policy_and_value':
        # Ensure finish_episode is also adapted to take current_task_cluster_assignment
        episode_function = finish_episode
    else:
        raise ValueError(f"Unknown or un-adapted rl_model_type for clustering: {rl_model_type}")

    task_names_for_clustering = args_config.label
    current_task_cluster_assignment = {}

    # --- Perform initial clustering ---
    if args_config.clustering_active and args_config.num_task_clusters > 0 and \
       args_config.num_task_clusters < len(task_names_for_clustering):
        logger.info("Performing initial task clustering before training starts...")
        if hasattr(policy_network, 'task_model') and policy_network.task_model is not None:
            policy_network.task_model.eval()

        # Create a default map assuming all tasks are in cluster 0 for the first representation generation.
        # This is the key fix for the TypeError.
        initial_default_map = {task: 0 for task in task_names_for_clustering}

        initial_task_representations = get_task_representations_from_activations(
            policy_network.task_model,
            eval_dataloader,
            task_names_for_clustering,
            device,
            initial_default_map # Pass the default map here
        )

        if initial_task_representations and \
           len(initial_task_representations) == len(task_names_for_clustering) and \
           all(isinstance(vec, torch.Tensor) for vec in initial_task_representations.values()):
            # Now, create the first REAL cluster assignment using these representations
            current_task_cluster_assignment = assign_tasks_to_clusters(
                initial_task_representations, task_names_for_clustering,
                args_config.num_task_clusters, args_config.kmeans_random_state
            )
            logger.info(f"Initial task cluster assignment for training: {current_task_cluster_assignment}")
        else:
            logger.warning("Could not perform initial clustering from model. Defaulting all tasks to cluster 0.")
            current_task_cluster_assignment = {task: 0 for task in task_names_for_clustering}
    else:
        current_task_cluster_assignment = {task_name: 0 for task_name in task_names_for_clustering}
        logger.info(f"Clustering not active or K is trivial. Initializing all tasks to cluster 0 for training: {current_task_cluster_assignment}")

    # Log initial assignments and get first batch info
    if not no_wandb and not only_ensemble:
        if args_config.clustering_active and current_task_cluster_assignment:
             wandb.log({"initial_cluster_assignments": {k:float(v) for k,v in current_task_cluster_assignment.items()}}, commit=False)

        log_dict_init = get_first_batch_info(
            policy_network, eval_dataloader, device, bag_size, sample_algorithm_rl,
            args_config, current_task_cluster_assignment
        )
        if log_dict_init : wandb.log(log_dict_init, commit=False)

    # --- Main Training Loop ---
    for epoch in range(epochs):
        log_dict_epoch = {"epoch": epoch}
        current_warmup_phase = epoch < warmup_epochs

        # --- Periodic Re-clustering ---
        if args_config.clustering_active and args_config.num_task_clusters > 0 and \
           args_config.num_task_clusters < len(task_names_for_clustering) and \
           epoch > 0 and epoch % args_config.recluster_interval_E == 0:

            logger.info(f"Epoch {epoch}: Re-calculating task representations and clustering...")
            if hasattr(policy_network, 'task_model') and policy_network.task_model is not None:
                policy_network.task_model.eval()

            # Pass the CURRENT cluster map to the function
            temp_task_representations = get_task_representations_from_activations(
                policy_network.task_model, eval_dataloader, args_config.label, DEVICE,
                current_task_cluster_assignment # Pass the map from the previous epoch
            )
            if temp_task_representations and len(temp_task_representations) == len(args_config.label) and \
               all(isinstance(vec, torch.Tensor) for vec in temp_task_representations.values()):

                new_cluster_assignments = assign_tasks_to_clusters(
                    temp_task_representations, args_config.label, args_config.num_task_clusters, args_config.kmeans_random_state
                )
                if new_cluster_assignments and new_cluster_assignments != current_task_cluster_assignment:
                    logger.info(f"Task clusters CHANGED at epoch {epoch}.")
                    logger.info(f"Previous: {current_task_cluster_assignment}, New: {new_cluster_assignments}")
                    current_task_cluster_assignment = new_cluster_assignments
                    if not no_wandb: log_dict_epoch["cluster_assignments_changed_epoch"] = float(epoch)
                elif new_cluster_assignments:
                     logger.info(f"Task clusters remain unchanged: {current_task_cluster_assignment}")
            else:
                logger.warning(f"Epoch {epoch}: Failed to get all task representations. Skipping re-clustering.")

        if args_config.clustering_active and not no_wandb and current_task_cluster_assignment:
            log_dict_epoch["cluster_assignments"] = {k:float(v) for k,v in current_task_cluster_assignment.items()}

        actor_critic_loss_val, policy_loss_val, value_loss_val, mil_loss_val, reg_loss_val = episode_function(
            policy_network, train_dataloader, eval_dataloader, optimizer, device, bag_size,
            train_pool_size, scheduler, current_warmup_phase, only_ensemble,
            epsilon_rl, reg_coef_rl, sample_algorithm_rl,
            current_task_cluster_assignment
        )

        policy_network.eval()
        eval_pool = policy_network.create_pool_data(eval_dataloader, bag_size, eval_pool_size, random=only_ensemble)
        eval_avg_combined_reward, eval_avg_combined_loss, eval_ensemble_combined_reward = policy_network.expected_reward_loss(
            eval_pool, current_task_cluster_assignment
        )

        detailed_eval_metrics = {}
        if eval_pool and len(eval_pool) > 0 and eval_pool[0]:
            detailed_eval_metrics, _, _, _ = policy_network.compute_metrics_and_details(
                eval_pool[0], current_task_cluster_assignment
            )
        else:
            for task_name_iter in args_config.label:
                detailed_eval_metrics[f"{task_name_iter}/f1"]=0.0; detailed_eval_metrics[f"{task_name_iter}/accuracy"]=0.0; detailed_eval_metrics[f"{task_name_iter}/auc"]=0.0
            detailed_eval_metrics['loss'] = float('nan')

        early_stopping(eval_avg_combined_reward, policy_network, epoch=epoch)

        if not no_wandb:
            train_eval_pool = policy_network.create_pool_data(train_dataloader, bag_size, eval_pool_size, random=only_ensemble)
            detailed_train_metrics = {}
            if train_eval_pool and len(train_eval_pool) > 0 and train_eval_pool[0]:
                detailed_train_metrics, _, _, _ = policy_network.compute_metrics_and_details(
                    train_eval_pool[0], current_task_cluster_assignment
                )
            else:
                for task_name_iter in args_config.label:
                    detailed_train_metrics[f"{task_name_iter}/f1"]=0.0; detailed_train_metrics[f"{task_name_iter}/accuracy"]=0.0; detailed_train_metrics[f"{task_name_iter}/auc"]=0.0
                detailed_train_metrics['loss'] = float('nan')

            train_avg_combined_reward, _, train_ensemble_combined_reward = policy_network.expected_reward_loss(
                train_eval_pool, current_task_cluster_assignment
            )
            log_dict_epoch.update({
                "train/total_actor_critic_loss": actor_critic_loss_val, "train/policy_loss": policy_loss_val,
                "train/value_loss": value_loss_val, "train/reg_loss": reg_loss_val,
                "train/avg_mil_batch_combined_loss": mil_loss_val,
                "eval/avg_mil_combined_loss": eval_avg_combined_loss,
                f"train/avg_COMBINED_REWARD": train_avg_combined_reward,
                f"train/ensemble_COMBINED_REWARD": train_ensemble_combined_reward,
                f"eval/avg_COMBINED_REWARD": eval_avg_combined_reward,
                f"eval/ensemble_COMBINED_REWARD": eval_ensemble_combined_reward,
            })
            for task_name_iter in args_config.label:
                log_dict_epoch[f"train/{task_name_iter}_f1"] = detailed_train_metrics.get(f"{task_name_iter}/f1", 0.0)
                log_dict_epoch[f"train/{task_name_iter}_acc"] = detailed_train_metrics.get(f"{task_name_iter}/accuracy", 0.0)
                log_dict_epoch[f"train/{task_name_iter}_auc"] = detailed_train_metrics.get(f"{task_name_iter}/auc", 0.0)
                log_dict_epoch[f"eval/{task_name_iter}_f1"] = detailed_eval_metrics.get(f"{task_name_iter}/f1", 0.0)
                log_dict_epoch[f"eval/{task_name_iter}_acc"] = detailed_eval_metrics.get(f"{task_name_iter}/accuracy", 0.0)
                log_dict_epoch[f"eval/{task_name_iter}_auc"] = detailed_eval_metrics.get(f"{task_name_iter}/auc", 0.0)

            if early_stopping.counter == 0:
                 log_dict_epoch.update({
                    "best/eval_avg_mil_combined_loss": eval_avg_combined_loss,
                    f"best/eval_avg_COMBINED_REWARD": eval_avg_combined_reward,
                    f"best/eval_ensemble_COMBINED_REWARD": eval_ensemble_combined_reward,
                 })
                 for task_name_iter in args_config.label:
                    log_dict_epoch[f"best/eval_{task_name_iter}_f1"] = detailed_eval_metrics.get(f"{task_name_iter}/f1", 0.0)
                    log_dict_epoch[f"best/eval_{task_name_iter}_acc"] = detailed_eval_metrics.get(f"{task_name_iter}/accuracy", 0.0)

            if not only_ensemble :
                 batch_log_dict_epoch_train = get_first_batch_info(
                     policy_network, eval_dataloader, device, bag_size, sample_algorithm_rl,
                     args_config, current_task_cluster_assignment
                 )
                 if batch_log_dict_epoch_train: log_dict_epoch.update(batch_log_dict_epoch_train)
            wandb.log(log_dict_epoch)

        if run_name:
            current_sweep_metric_val = eval_ensemble_combined_reward # Or eval_avg_combined_reward
            if current_sweep_metric_val is not None and (BEST_REWARD == float("-inf") or current_sweep_metric_val > BEST_REWARD):
                # ... (Logic for saving best sweep model as before) ...
                pass # Placeholder for brevity

        if early_stopping.early_stop and not current_warmup_phase:
            logger.info(f"Early stopping at epoch {epoch}. Best metric for this run: {early_stopping.best_metric_value:.6f} at epoch {early_stopping.best_epoch}")
            break
    # --- End Epoch Loop ---

    # --- Final Evaluation at End of Training ---
    logger.info(f"Loading best model for this run from early stopping epoch {early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') and early_stopping.best_epoch !=-1 else 'N/A'}")
    if early_stopping.model_address and os.path.exists(early_stopping.model_address):
        policy_network.load_state_dict(torch.load(early_stopping.model_address))
    else:
        logger.warning(f"Could not load best model for this run from {early_stopping.model_address}")

    policy_network.eval()
    final_test_pool = policy_network.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
    final_eval_cluster_map = current_task_cluster_assignment

    final_detailed_test_metrics, _, _, _ = policy_network.compute_metrics_and_details(
        final_test_pool[0] if final_test_pool and final_test_pool[0] else [], final_eval_cluster_map
    )
    final_test_avg_reward, final_test_combined_loss, final_test_ensemble_reward = policy_network.expected_reward_loss(
        final_test_pool, final_eval_cluster_map
    )
    final_log_summary = {
        "final_eval_on_test/combined_loss": final_test_combined_loss,
        f"final_eval_on_test/avg_COMBINED_REWARD": final_test_avg_reward,
        f"final_eval_on_test/ensemble_COMBINED_REWARD": final_test_ensemble_reward,
        "final_cluster_assignment_for_eval": {k:float(v) for k,v in final_eval_cluster_map.items()},
        **{f"final_eval_on_test/{k.replace('loss', 'test_loss')}":v for k,v in final_detailed_test_metrics.items()}
    }
    if not no_wandb: wandb.log(final_log_summary)
    logger.info(f"Final Test Metrics for this run (model from epoch {early_stopping.best_epoch}): {final_log_summary}")

    if not run_name:
        save_json(os.path.join(early_stopping.models_dir, "results.json"), {
            "model": "rl-" + args_config.baseline, "embedding_model": args_config.embedding_model,
            "bag_size": args_config.bag_size, "dataset": args_config.dataset, "labels": args_config.label,
            "seed": args_config.random_seed, **final_log_summary
        })

    return policy_network

def main_sweep():
    global BEST_REWARD, DEVICE, logger, args, global_run_dir
    BEST_REWARD = float("-inf")

    try:
        run = wandb.init()
        trial_args = argparse.Namespace(**vars(args))
        config_from_wandb = wandb.config
        for key, value in config_from_wandb.items():
            setattr(trial_args, key, value)

        if logger is None:
            temp_log_dir = run.dir if run else "temp_sweep_log"
            os.makedirs(temp_log_dir, exist_ok=True)
            logger = get_logger(temp_log_dir)
            logger.warning("Global logger was None in main_sweep, re-initialized to trial dir.")

        if DEVICE is None: DEVICE = torch.device(f"cuda:{trial_args.gpu}" if torch.cuda.is_available() else "cpu")

        trial_args.no_wandb = False
        train_dataset, eval_dataset, test_dataset = prepare_data(trial_args)
        trial_args.input_dim = train_dataset.__getitem__(0)[0].shape[1]
        trial_args.state_dim = trial_args.autoencoder_layer_sizes[-1] if trial_args.autoencoder_layer_sizes else trial_args.input_dim

        logger.info(f"Sweep Trial ({run.id if run else 'N/A'}) effective args: {trial_args}")

        current_batch_size = trial_args.batch_size
        if (trial_args.balance_dataset) and (trial_args.task_type == "classification"):
            first_task_labels = [y_dict[trial_args.label[0]] for y_dict in train_dataset.Y]
            sample_weights = get_balanced_weights(first_task_labels)
            w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y), replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, num_workers=4, sampler=w_sampler)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=4)
        eval_dataloader = DataLoader(eval_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)

        pre_trained_mil_dir_label_part_sweep = "_".join(sorted(trial_args.label)) if isinstance(trial_args.label, list) else trial_args.label
        mil_checkpoint_dir_for_sweep_trial = get_model_save_directory(
            dataset=trial_args.dataset, data_embedded_column_name=trial_args.data_embedded_column_name,
            embedding_model_name=trial_args.embedding_model, target_column_name=pre_trained_mil_dir_label_part_sweep,
            bag_size=trial_args.bag_size, baseline=trial_args.baseline, autoencoder_layers=trial_args.autoencoder_layer_sizes,
            random_seed=trial_args.random_seed, dev=trial_args.dev, task_type=trial_args.task_type,
            prefix=None, multiple_runs=False
        )

        policy_network = create_rl_model_clustered(trial_args, mil_checkpoint_dir_for_sweep_trial)
        policy_network = policy_network.to(DEVICE)

        optimizer = optim.AdamW(
            [{"params": policy_network.actor.parameters(), "lr": trial_args.actor_learning_rate},
             {"params": policy_network.critic.parameters(), "lr": trial_args.critic_learning_rate},
             {"params": policy_network.task_model.parameters(), "lr": trial_args.learning_rate}],
            lr=trial_args.actor_learning_rate
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        trial_checkpoint_dir = run.dir if run else global_run_dir
        early_stopping_this_trial = EarlyStopping(
            models_dir=trial_checkpoint_dir,
            save_model_name=f"trial_checkpoint.pt",
            trace_func=logger.info, patience=trial_args.early_stopping_patience, verbose=True, descending=True
        )

        train(
            policy_network, optimizer, scheduler, early_stopping_this_trial,
            train_dataloader, eval_dataloader, test_dataloader,
            DEVICE, trial_args.bag_size, trial_args.epochs,
            trial_args.no_wandb,
            trial_args.train_pool_size, trial_args.eval_pool_size, trial_args.test_pool_size,
            trial_args.rl_model, trial_args.prefix, trial_args.epsilon, trial_args.reg_coef,
            trial_args.sample_algorithm,
            trial_args,
            trial_args.warmup_epochs, run_name=run.name if run else "sweep_trial",
            only_ensemble=trial_args.only_ensemble
        )
        if run: run.finish()

    except Exception as e:
        if logger:
            logger.error(f"Exception in main_sweep for run {run.id if 'run' in locals() and run else 'UNKNOWN'}: {type(e).__name__} - {e}")
            logger.error(traceback.format_exc())
        else:
            print(f"CRITICAL EXCEPTION in main_sweep (logger unavailable): {type(e).__name__} - {e}")
            traceback.print_exc()
        if 'run' in locals() and run: run.finish(exit_code=1)

def main():
    global DEVICE, logger, args, global_run_dir

    logger.info(f"Executing main() with args: {args}")

    train_dataset, eval_dataset, test_dataset = prepare_data(args) # Modifies args
    args.input_dim = train_dataset.__getitem__(0)[0].shape[1]
    args.state_dim = args.autoencoder_layer_sizes[-1] if args.autoencoder_layer_sizes else args.input_dim
    logger.info(f"Args after data prep for main run: {args}")

    current_batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 32
    if (args.balance_dataset) and (args.task_type == "classification"):
        first_task_labels = [y_dict[args.label[0]] for y_dict in train_dataset.Y]
        sample_weights = get_balanced_weights(first_task_labels)
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)

    pre_trained_mil_dir_label_part_main = "_".join(sorted(args.label)) if isinstance(args.label, list) else args.label
    mil_checkpoint_dir_for_main = get_model_save_directory(
        dataset=args.dataset, data_embedded_column_name=args.data_embedded_column_name,
        embedding_model_name=args.embedding_model, target_column_name=pre_trained_mil_dir_label_part_main,
        bag_size=args.bag_size, baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes,
        random_seed=args.random_seed, dev=args.dev, task_type=args.task_type,
        prefix=None, multiple_runs=False
    )

    policy_network = create_rl_model_clustered(args, mil_checkpoint_dir_for_main)
    policy_network = policy_network.to(DEVICE)

    optimizer = optim.AdamW(
        [{"params": policy_network.actor.parameters(), "lr": args.actor_learning_rate or 1e-4},
         {"params": policy_network.critic.parameters(), "lr": args.critic_learning_rate or 1e-4},
         {"params": policy_network.task_model.parameters(), "lr": args.learning_rate or 1e-5}],
        lr=args.actor_learning_rate or 1e-4,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    early_stopping_main_run = EarlyStopping(
        models_dir=global_run_dir, save_model_name=f"checkpoint.pt", trace_func=logger.info,
        patience=args.early_stopping_patience, verbose=True, descending=True
    )

    wandb_run_instance = None
    if not args.no_wandb:
        label_str_for_wandb = "_".join(sorted(args.label)) if isinstance(args.label, list) else args.label
        wandb_run_name = f"RL_MTL_{args.model_name}_{label_str_for_wandb}_{args.bag_size}_{args.prefix}"
        if args.clustering_active: wandb_run_name += f"_ClustK{args.num_task_clusters}"
        prefix_for_tag = args.prefix[:50] + ("..." if len(args.prefix) > 50 else "")
        wandb_run_instance = wandb.init(
            config=vars(args),
            tags=[
                f"DATASET_{args.dataset}", f"BAG_SIZE_{args.bag_size}", f"BASELINE_{args.baseline}",
                f"LABELS_{label_str_for_wandb}", f"EMBEDDING_{args.embedding_model}",
                f"SEED_{args.random_seed}", f"PREFIX_{prefix_for_tag}",
                f"CLUSTERING_{args.clustering_active}_K{args.num_task_clusters}"
            ],
            entity=args.wandb_entity, project=args.wandb_project, name=wandb_run_name,
        )

    # Call train; initial cluster assignment is handled inside train()
    train(
        policy_network, optimizer, scheduler, early_stopping_main_run,
        train_dataloader, eval_dataloader, test_dataloader,
        DEVICE, args.bag_size, args.epochs or 100,
        args.no_wandb,
        args.train_pool_size or 1, args.eval_pool_size or 10,
        args.test_pool_size or 10,
        args.rl_model, args.prefix, args.epsilon or 0.1, args.reg_coef or 0.01,
        args.sample_algorithm or "without_replacement",
        args,
        args.warmup_epochs,
        run_name=None,
        only_ensemble=args.only_ensemble
    )
    torch.save(policy_network.state_dict(), os.path.join(global_run_dir, f"final_model.pt"))

    if not args.no_wandb and wandb_run_instance:
        wandb_run_instance.finish()

if __name__ == "__main__":
    args = parse_args()

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    target_column_name_for_dir_main = "_".join(sorted(args.label)) if isinstance(args.label, list) else args.label
    global_run_dir = get_model_save_directory(
        dataset=args.dataset, data_embedded_column_name=args.data_embedded_column_name,
        embedding_model_name=args.embedding_model, target_column_name=target_column_name_for_dir_main,
        bag_size=args.bag_size, baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes,
        random_seed=args.random_seed, dev=args.dev, task_type=args.task_type, prefix=args.prefix,
        multiple_runs=args.multiple_runs
    )
    logger = get_logger(global_run_dir)
    logger.info(f"Global DEVICE set to: {DEVICE}")
    logger.info(f"Main run directory (for single runs or sweep's outputs): {global_run_dir}")

    args.model_name = get_model_name(baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes)

    if args.run_sweep:
        if not args.wandb_project:
            raise ValueError("wandb_project must be set for sweeps.")

        label_str_for_sweep_name = "_".join(sorted(args.label)) if isinstance(args.label, list) else args.label
        sweep_name_base = f"{args.prefix}_{args.dataset}_{label_str_for_sweep_name}_rl_{args.baseline}".replace("_", "-")
        if args.clustering_active:
            sweep_name_base += f"_clustK{args.num_task_clusters}"

        if not hasattr(args, 'sweep_config') or not args.sweep_config:
            logger.warning("Sweep config not found on args. Attempting to load default RL sweep config.")
            default_sweep_yaml = "hp_rl_mtl_loss.yaml" # Or determine based on args.prefix etc.
            sweep_config_file_address = os.path.join(os.path.dirname(__file__), "yaml_configs", default_sweep_yaml)
            if os.path.exists(sweep_config_file_address):
                args.sweep_config = load_yaml_file(sweep_config_file_address)
            else:
                raise FileNotFoundError(f"Default sweep YAML {default_sweep_yaml} not found.")

        args.sweep_config["name"] = sweep_name_base

        logger.info(f"Starting W&B sweep: {args.sweep_config.get('name')}")
        logger.info(f"Sweep config base from YAML: {args.sweep_config}")
        sweep_id = wandb.sweep(args.sweep_config, entity=args.wandb_entity, project=args.wandb_project)
        wandb.agent(sweep_id, function=main_sweep, count=args.sweep_config.get('run_cap', 50))
    else:
        main()
