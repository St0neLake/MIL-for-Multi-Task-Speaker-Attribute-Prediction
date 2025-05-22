import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from configs import parse_args
from RLMIL_Datasets import RLMILDataset
from logger import get_logger
from models import PolicyNetwork, sample_action, select_from_action, create_mil_model_with_dict
from utils import (
    get_data_directory,
    get_model_name,
    get_model_save_directory,
    read_data_split,
    preprocess_dataframe,
    create_preprocessed_dataframes,
    get_df_mean_median_std,
    get_balanced_weights,
    EarlyStopping, save_json, load_json, create_mil_model
)

logger = None
DEVICE = None
BEST_REWARD = float("-inf")

def finish_episode_policy_only(
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
    sel_losses, regularization_losses = [], []
    for batch_x, batch_y_dict, _, _  in train_dataloader:
        policy_network.train()
        batch_x = batch_x.to(device)
        batch_y_dict_device = {k: v.to(device) for k, v in batch_y_dict.items()}

        action_probs, _, _ = policy_network(batch_x)

        # logger.info(f"action_probs.shape={action_probs.shape}")
        action, action_log_prob = sample_action(action_probs,
                                                bag_size,
                                                device=device,
                                                random=(epsilon > np.random.random()) or only_ensemble,
                                                algorithm=sample_algorithm)
        sel_x = select_from_action(action, batch_x)

        sel_loss = policy_network.train_minibatch(sel_x, batch_y_dict_device)
        sel_losses.append(sel_loss)

        policy_network.eval()
        # reward = policy_network.compute_reward(eval_data)
        if not only_ensemble:
            reward, _, _ = policy_network.expected_reward_loss(eval_pool)
            policy_network.saved_actions.append(action_log_prob)
            policy_network.rewards.append(reward)
            regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))


    if only_ensemble:
        return 0, 0, 0, np.mean(sel_losses) if sel_losses else 0, 0

    policy_network.normalize_rewards(eps=1e-5)

    policy_losses = []
    policy_network.train()
    for log_prob, reward in zip(policy_network.saved_actions, policy_network.rewards):
        policy_losses.append(-reward * log_prob.cuda())

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_losses).mean()
    regularization_loss = torch.stack(regularization_losses).mean() / 100
    total_loss = policy_loss + reg_coef * regularization_loss
    # perform backprop
    total_loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    # reset rewards and action buffer
    policy_network.reset_reward_action()

    return total_loss.item(), policy_loss.item(), 0, \
        np.mean(sel_losses) if sel_losses else 0, \
        (reg_coef * regularization_loss.item()) if regularization_losses else 0

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

def create_rl_model(args, mil_best_model_dir):
    task_model_config_path = os.path.join(mil_best_model_dir, "..", "best_model_config.json")

    if os.path.exists(task_model_config_path):
        mil_config_dict = load_json(task_model_config_path)
    else:
        logger.warning(f"MIL config {task_model_config_path} not found. Using args for MIL model creation.")
        mil_config_dict = vars(args).copy() # Use current run's args

    # Ensure output_dims_dict is present and correct for MTL
    mil_config_dict["output_dims_dict"] = args.output_dims_dict
    mil_config_dict["number_of_classes"] = None # Deprecate, use output_dims_dict

    for key in ["baseline", "input_dim", "hidden_dim", "dropout_p", "autoencoder_layer_sizes", "n_hidden_sets", "n_elements"]:
        if key not in mil_config_dict and hasattr(args, key):
            mil_config_dict[key] = getattr(args, key)
        elif key in mil_config_dict and hasattr(args, key) and getattr(args,key) is not None: # Allow overriding loaded config with args
             mil_config_dict[key] = getattr(args, key)


    task_model = create_mil_model_with_dict(mil_config_dict)

    task_model_state_dict_path = os.path.join(mil_best_model_dir, "..", "best_model.pt")
    if args.rl_task_model == "ensemble":
        logger.warning("Ensemble loading for MTL task_model needs careful state_dict alignment.")
        # Implement ensemble loading if necessary, ensuring compatibility with multi-head task_model
        # This might involve loading a base and then specific heads if the ensemble was single-task.
        # For simplicity in initial MTL, you might train task_model from scratch or use a single pre-trained state.
        # Placeholder for ensemble loading:
        # ensemble_state_dict_path = os.path.join(mil_best_model_dir, "..", "only_ensemble_loss_sweep_best_rl_model.pt") # Example
        # if os.path.exists(ensemble_state_dict_path):
        #     ensemble_state_dict = torch.load(ensemble_state_dict_path, map_location=torch.device("cpu"))
        #     # Careful mapping to multi-head task_model needed here
        #     # task_model.load_state_dict(mapped_ensemble_state_dict, strict=False)
        pass # Requires careful implementation for MTL
    elif os.path.exists(task_model_state_dict_path):
        logger.info(f"Loading task_model state_dict from {task_model_state_dict_path}")
        task_model_state_dict = torch.load(task_model_state_dict_path, map_location=torch.device("cpu"))
        task_model.load_state_dict(task_model_state_dict, strict=False)
    else:
        logger.info(f"No pre-trained task_model state_dict found at {task_model_state_dict_path}. Training task_model from scratch.")

    policy_network = PolicyNetwork(
        task_model=task_model,
        state_dim=args.state_dim,
        hdim=args.hdim,
        learning_rate=args.learning_rate,
        device=DEVICE,
        task_type=args.task_type,
        min_clip=args.min_clip,
        max_clip=args.max_clip,
        sample_algorithm=args.sample_algorithm,
        no_autoencoder_for_rl=args.no_autoencoder_for_rl,
        epsilon=args.epsilon
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

def get_first_batch_info(policy_network, eval_dataloader, device, bag_size, sample_algorithm, args_config):
    log_dict = {}
    batch_x, batch_y_dict, indices, instance_labels_from_loader = next(iter(eval_dataloader))
    batch_x = batch_x.to(device)
    action_probs, _, _ = policy_network(batch_x) # From actor
    action, _ = sample_action(action_probs, bag_size, device, random=False, algorithm=sample_algorithm)

    for i in range(min(action_probs.shape[0], 3)): # Log for first 3 samples in batch for brevity
        log_dict.update({
            f"actor/probs_sample_{i}": wandb.Histogram(action_probs[i].cpu().detach().numpy()),
            f"actor/action_sample_{i}": wandb.Histogram(action[i].cpu().numpy().tolist())
        })

    if args_config.instance_labels_column is not None and len(instance_labels_from_loader) > 0:
        instance_labels_tensor = instance_labels_from_loader.to(device)
        try:
            if instance_labels_tensor.shape[1] == batch_x.shape[1]: # Check if num_instances match
                selected_instance_values = instance_labels_tensor[torch.arange(action.shape[0]).unsqueeze(1), action]
                # Example: sum of selected instance labels (if they are scores or binary indicators)
                selected_instance_sum = selected_instance_values.sum(dim=1)

                # Log this sum for the first few samples
                for i in range(min(batch_x.shape[0], 3)):
                    log_dict.update({f"actor/selected_instance_sum_sample_{i}": selected_instance_sum[i].item()})
            else:
                logger.warning(f"Shape mismatch for instance_labels_tensor {instance_labels_tensor.shape} and batch_x {batch_x.shape} for instance-level logging.")

        except IndexError as e:
            logger.error(f"IndexError in get_first_batch_info instance logging: {e}. Action shape: {action.shape}, Instance labels shape: {instance_labels_tensor.shape}")
        except Exception as e:
            logger.error(f"Other error in get_first_batch_info instance logging: {e}")


    return log_dict

def train(
        policy_network,
        optimizer,
        scheduler,
        early_stopping,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        device,
        bag_size,
        epochs,
        no_wandb,
        train_pool_size,
        eval_pool_size,
        test_pool_size,
        rl_model,
        prefix,
        epsilon,
        reg_coef,
        sample_algorithm,
        args,
        warmup_epochs=0,
        run_name=None,
        # task_type='classification',
        only_ensemble=False,
):
    global BEST_REWARD

    if rl_model == 'policy_and_value':
        logger.warning("finish_episode for 'policy_and_value' needs MTL adaptation if used.")
        episode_function = finish_episode # This would need MTL changes similar to finish_episode_policy_only
    elif rl_model == 'policy_only':
        episode_function = finish_episode_policy_only

    # metric = 'f1' if task_type == 'classification' else 'r2'

    # wandb.watch(policy_network.actor, log="all", log_freq=100, log_graph=True)
    if not no_wandb and not only_ensemble:
        log_dict = get_first_batch_info(policy_network, eval_dataloader, device, bag_size, sample_algorithm, args)
        wandb.log(log_dict)

    # logger.info(f"Training model started ....")
    for epoch in range(epochs):
        log_dict = {}
        current_warmup_phase = epoch < warmup_epochs

        total_actor_critic_loss, policy_loss, value_loss, avg_batch_mil_loss, reg_loss = episode_function(
            policy_network=policy_network,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader, # Used for reward calculation within episode_function
            optimizer=optimizer,      # For actor/critic
            scheduler=scheduler,      # For actor/critic
            device=device,
            bag_size=bag_size,
            train_pool_size=train_pool_size, # For creating eval_pool inside episode_function
            warmup=current_warmup_phase, # Not used by adapted finish_episode_policy_only directly
            only_ensemble=only_ensemble,
            epsilon=epsilon,
            reg_coef=reg_coef,
            sample_algorithm=sample_algorithm
        )

        policy_network.eval()

        # Evaluate on validation set using the current policy
        eval_pool = policy_network.create_pool_data(eval_dataloader, bag_size, eval_pool_size, random=only_ensemble)
        eval_avg_combined_reward, eval_avg_combined_loss, eval_ensemble_combined_reward = policy_network.expected_reward_loss(eval_pool)

        # For detailed per-task metrics on eval set for logging
        detailed_eval_metrics = {} # Initialize
        if eval_pool and len(eval_pool) > 0:
            representative_eval_data_list = eval_pool[0] # This is a list of 4-tuples
            detailed_eval_metrics, _, _, _ = policy_network.compute_metrics_and_details(representative_eval_data_list)
        else:
            print("[WARNING train] eval_pool is empty, cannot compute detailed_eval_metrics.")
            # Populate with default zero/NaN metrics if needed for logging
            for task_name in args.label:
                detailed_eval_metrics[f"{task_name}/f1"] = 0.0
                detailed_eval_metrics[f"{task_name}/accuracy"] = 0.0
                if args.task_type == 'classification':
                    detailed_eval_metrics[f"{task_name}/auc"] = 0.0
            detailed_eval_metrics['loss'] = float('nan')

        early_stopping(eval_avg_combined_reward, policy_network) # Early stopping based on combined reward

        if not no_wandb:
            train_eval_pool = policy_network.create_pool_data(train_dataloader, bag_size, eval_pool_size, random=only_ensemble) # Using eval_pool_size for consistency
            detailed_train_metrics = {}
            if train_eval_pool and len(train_eval_pool) > 0:
                representative_train_data_list = train_eval_pool[0]
                detailed_train_metrics, _, _, _ = policy_network.compute_metrics_and_details(representative_train_data_list)
            else:
                print("[WARNING train] train_eval_pool is empty, cannot compute detailed_train_metrics.")
                for task_name in args.label:
                    detailed_train_metrics[f"{task_name}/f1"] = 0.0
                    detailed_train_metrics[f"{task_name}/accuracy"] = 0.0
                detailed_train_metrics['loss'] = float('nan')

            train_avg_combined_reward, _, train_ensemble_combined_reward = policy_network.expected_reward_loss(train_eval_pool)

            log_dict.update({
                "epoch": epoch,
                "train/total_actor_critic_loss": total_actor_critic_loss,
                "train/policy_loss": policy_loss,
                "train/value_loss": value_loss, # Will be 0 for policy_only
                "train/reg_loss": reg_loss,
                "train/avg_mil_batch_combined_loss": avg_batch_mil_loss, # Avg of MIL losses during training step

                "eval/avg_mil_combined_loss": eval_avg_combined_loss, # Avg MIL loss on full eval_pool
                f"train/avg_COMBINED_REWARD": train_avg_combined_reward,
                f"train/ensemble_COMBINED_REWARD": train_ensemble_combined_reward,
                f"eval/avg_COMBINED_REWARD": eval_avg_combined_reward,
                f"eval/ensemble_COMBINED_REWARD": eval_ensemble_combined_reward,
            })

            # Log per-task F1 and Accuracy (assuming classification)
            for task_name in args.label:
                log_dict[f"train/{task_name}_f1"] = detailed_train_metrics.get(f"{task_name}/f1", 0.0)
                log_dict[f"train/{task_name}_acc"] = detailed_train_metrics.get(f"{task_name}/accuracy", 0.0)
                log_dict[f"eval/{task_name}_f1"] = detailed_eval_metrics.get(f"{task_name}/f1", 0.0)
                log_dict[f"eval/{task_name}_acc"] = detailed_eval_metrics.get(f"{task_name}/accuracy", 0.0)
                if args.task_type == 'classification': # Log AUC for classification tasks
                     log_dict[f"train/{task_name}_auc"] = detailed_train_metrics.get(f"{task_name}/auc", 0.0)
                     log_dict[f"eval/{task_name}_auc"] = detailed_eval_metrics.get(f"{task_name}/auc", 0.0)


            if not only_ensemble:
                batch_log_dict_epoch = get_first_batch_info(policy_network, eval_dataloader, device, bag_size, sample_algorithm, args)
                log_dict.update(batch_log_dict_epoch)

            if early_stopping.counter == 0: # If this is the best model so far by early stopping
                log_dict.update({
                    "best/eval_avg_mil_combined_loss": eval_avg_combined_loss,
                    f"best/eval_avg_COMBINED_REWARD": eval_avg_combined_reward,
                    f"best/eval_ensemble_COMBINED_REWARD": eval_ensemble_combined_reward,
                })
                for task_name in args.label: # Log best per-task metrics
                    log_dict[f"best/eval_{task_name}_f1"] = detailed_eval_metrics.get(f"{task_name}/f1", 0.0)
                    log_dict[f"best/eval_{task_name}_acc"] = detailed_eval_metrics.get(f"{task_name}/accuracy", 0.0)
            wandb.log(log_dict)

        if run_name:  # sweep
            current_sweep_metric = eval_ensemble_combined_reward
            if current_sweep_metric > BEST_REWARD:
                logger.info(
                    f"Sweep run {run_name}: New best model at epoch {epoch}. Combined Ensemble Reward "
                    f"increased ({BEST_REWARD:.6f} --> {current_sweep_metric:.6f})."
                )
                BEST_REWARD = current_sweep_metric
                torch.save(
                    policy_network.state_dict(),
                    os.path.join(early_stopping.models_dir, "sweep_best_model.pt"),
                )

                # Save best model config for this sweep run
                best_sweep_config_params = {
                    "critic_learning_rate": args.critic_learning_rate, # from wandb.config in main_sweep
                    "actor_learning_rate": args.actor_learning_rate, # from wandb.config
                    "learning_rate": args.learning_rate, # MIL model LR, from wandb.config
                    "epochs": args.epochs, # from wandb.config
                    "hdim": args.hdim, # from wandb.config
                    # Add other hyperparams from wandb.config that defined this run
                    "epsilon": args.epsilon,
                    "reg_coef": args.reg_coef,
                    "warmup_epochs": args.warmup_epochs
                }
                full_best_config = vars(args).copy()
                full_best_config.update(best_sweep_config_params) # Ensure sweep params override
                save_json(
                    path=os.path.join(early_stopping.models_dir, "sweep_best_model_config.json"),
                    data=full_best_config
                )

                # Evaluate this new best sweep model on test set and save results.json
                policy_network.eval()
                test_pool = policy_network.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
                detailed_test_metrics_sweep, _, _, _ = policy_network.compute_metrics_and_details(test_pool)
                test_avg_combined_reward_sweep, test_combined_loss_sweep, test_ensemble_combined_reward_sweep = policy_network.expected_reward_loss(test_pool)

                results_json = {
                    "model": "rl-" + args.baseline,
                    "embedding_model": args.embedding_model,
                    "bag_size": args.bag_size,
                    "dataset": args.dataset,
                    "labels": args.label, # List of labels
                    "seed": args.random_seed, # Seed for this specific run within the sweep
                    "sweep_run_name": run_name,
                    "test/combined_loss": test_combined_loss_sweep,
                    f"test/avg_COMBINED_REWARD": test_avg_combined_reward_sweep,
                    f"test/ensemble_COMBINED_REWARD": test_ensemble_combined_reward_sweep,
                    # Add per-task test metrics
                }
                for task_name_res in args.label:
                    results_json[f"test/{task_name_res}_f1"] = detailed_test_metrics_sweep.get(f"{task_name_res}/f1", 0.0)
                    results_json[f"test/{task_name_res}_acc"] = detailed_test_metrics_sweep.get(f"{task_name_res}/accuracy", 0.0)
                    if args.task_type == 'classification':
                         results_json[f"test/{task_name_res}_auc"] = detailed_test_metrics_sweep.get(f"{task_name_res}/auc", 0.0)

                # Also save the eval metrics that led to this being the best
                results_json[f"eval_at_best/avg_COMBINED_REWARD"] = eval_avg_combined_reward
                results_json[f"eval_at_best/ensemble_COMBINED_REWARD"] = eval_ensemble_combined_reward
                for task_name_res in args.label:
                     results_json[f"eval_at_best/{task_name_res}_f1"] = detailed_eval_metrics.get(f"{task_name_res}/f1", 0.0)

                save_json(os.path.join(early_stopping.models_dir, "results.json"), results_json)

        if current_warmup_phase and epoch == warmup_epochs -1 : # End of warmup
            logger.info("Warmup phase finished. Resetting early stopping counter.")
            early_stopping.counter = 0
            early_stopping.best_score = None # Reset best score to start fresh after warmup
            early_stopping.val_loss_min = np.Inf # Or appropriate initial value

        if early_stopping.early_stop and not current_warmup_phase :
            logger.info(f"Early stopping at epoch {epoch} out of {epochs}")
            break

    # Load the best model based on early_stopping for final evaluation
    logger.info(f"Loading best model from epoch {early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else 'N/A'}")
    policy_network.load_state_dict(torch.load(early_stopping.model_address))
    policy_network.eval()

    final_test_pool = policy_network.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
    final_detailed_test_metrics, _, _, _ = policy_network.compute_metrics_and_details(final_test_pool)
    final_test_avg_reward, final_test_combined_loss, final_test_ensemble_reward = policy_network.expected_reward_loss(final_test_pool)

    final_log_summary = {
        "final_eval_on_test/combined_loss": final_test_combined_loss,
        f"final_eval_on_test/avg_COMBINED_REWARD": final_test_avg_reward,
        f"final_eval_on_test/ensemble_COMBINED_REWARD": final_test_ensemble_reward
    }
    for task_name in args.label:
        final_log_summary[f"final_eval_on_test/{task_name}_f1"] = final_detailed_test_metrics.get(f"{task_name}/f1", 0.0)
        final_log_summary[f"final_eval_on_test/{task_name}_acc"] = final_detailed_test_metrics.get(f"{task_name}/accuracy", 0.0)
        if args.task_type == 'classification':
            final_log_summary[f"final_eval_on_test/{task_name}_auc"] = final_detailed_test_metrics.get(f"{task_name}/auc", 0.0)

    if not no_wandb:
        wandb.log(final_log_summary)
    logger.info(f"Final Test Metrics: {final_log_summary}")

    # If not a sweep run, save final results here too
    if not run_name: # This implies it's a single run, not part of a sweep agent call
        results_json_single_run = {
            "model": "rl-" + args.baseline, "embedding_model": args.embedding_model,
            "bag_size": args.bag_size, "dataset": args.dataset, "labels": args.label,
            "seed": args.random_seed,
            **final_log_summary # Include all final test metrics
        }
        save_json(os.path.join(early_stopping.models_dir, "results.json"), results_json_single_run)

    return policy_network


def main_sweep():
    global BEST_REWARD, DEVICE, logger
    BEST_REWARD = float("-inf")

    run = wandb.init( # Sweep agent will init this run
        # tags are set by the sweep config or can be added here
    )
    config_from_wandb = wandb.config

    # Override args with sweep config. args is the initial parse_args()
    args.critic_learning_rate = config_from_wandb.critic_learning_rate
    args.actor_learning_rate = config_from_wandb.actor_learning_rate
    args.learning_rate = config_from_wandb.learning_rate # For MIL model
    args.epochs = config_from_wandb.epochs
    args.hdim = config_from_wandb.hdim # For critic
    if hasattr(config_from_wandb, 'early_stopping_patience'):
        args.early_stopping_patience = config_from_wandb.early_stopping_patience
    args.warmup_epochs = config_from_wandb.get("warmup_epochs", args.warmup_epochs) # Default from initial args if not in sweep
    args.epsilon = config_from_wandb.get("epsilon", args.epsilon)
    args.reg_coef = config_from_wandb.get("reg_coef", args.reg_coef)
    # Other args like batch_size might also be part of the sweep
    if hasattr(config_from_wandb, 'batch_size'):
        args.batch_size = config_from_wandb.batch_size

    args.no_wandb = False # Wandb is active for sweeps

    train_dataset, eval_dataset, test_dataset = prepare_data(args)
    current_batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 32 # Default

    if (args.balance_dataset) & (args.task_type == "classification"):
        sample_weights = get_balanced_weights(train_dataset.Y.tolist())
        logger.warning("Dataset balancing for MTL needs careful consideration on which task to balance or how to combine.")
        # Example: balance based on first task's labels
        first_task_labels = [y_dict[args.label[0]] for y_dict in train_dataset.Y]
        sample_weights = get_balanced_weights(first_task_labels)
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)

    args.input_dim = train_dataset.__getitem__(0)[0].shape[1]
    if args.autoencoder_layer_sizes is None:
        args.state_dim = args.input_dim
    else:
        args.state_dim = args.autoencoder_layer_sizes[-1]

    # MODEL CREATION
    # run_dir for this sweep run. Use wandb.run.dir for sweep-specific artifacts.
    # The main model_save_directory is for the overall experiment, sweep_best_model.pt will go there.
    target_column_name_for_dir = "_".join(args.label) if isinstance(args.label, list) else args.label
    run_dir_for_sweep = get_model_save_directory(
        dataset=args.dataset, data_embedded_column_name=args.data_embedded_column_name,
        embedding_model_name=args.embedding_model, target_column_name=target_column_name_for_dir,
        bag_size=args.bag_size, baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes,
        random_seed=args.random_seed, dev=args.dev, task_type=args.task_type, prefix=args.prefix,
        multiple_runs=args.multiple_runs
    )

    policy_network = create_rl_model(args, run_dir_for_sweep) # mil_best_model_dir is for loading base MIL model
    policy_network = policy_network.to(DEVICE)

    optimizer = optim.AdamW(
        [{"params": policy_network.actor.parameters(), "lr": args.actor_learning_rate,},
         {"params": policy_network.critic.parameters(), "lr": args.critic_learning_rate,}],
        lr=args.actor_learning_rate, # Fallback, but individual LRs should dominate
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) # Example scheduler

    # Early stopping for this specific sweep run's checkpoint (not the global best)
    # The global "sweep_best_model.pt" is saved within train() based on BEST_REWARD.
    sweep_run_checkpoint_dir = wandb.run.dir
    early_stopping_sweep_run = EarlyStopping(
        models_dir=sweep_run_checkpoint_dir, save_model_name=f"sweep_run_checkpoint.pt",
        trace_func=logger.info, patience=args.early_stopping_patience, verbose=True, descending=True
    )

    train(
        policy_network=policy_network, optimizer=optimizer, scheduler=scheduler,
        early_stopping=early_stopping_sweep_run,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, test_dataloader=test_dataloader,
        device=DEVICE, bag_size=args.bag_size, epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, no_wandb=args.no_wandb,
        train_pool_size=args.train_pool_size, eval_pool_size=args.eval_pool_size, test_pool_size=args.test_pool_size,
        run_name=run.name,
        only_ensemble=args.only_ensemble, rl_model=args.rl_model,
        prefix=args.prefix, epsilon=args.epsilon, reg_coef=args.reg_coef,
        sample_algorithm=args.sample_algorithm, args=args # Pass full args
    )
    run.finish()


def main():
    global DEVICE, logger, args

    # Model name and directory (Primary directory for this configuration, not sweep-run specific)
    target_column_name_for_dir = "_".join(args.label) if isinstance(args.label, list) else args.label
    run_dir = get_model_save_directory(
        dataset=args.dataset, data_embedded_column_name=args.data_embedded_column_name,
        embedding_model_name=args.embedding_model, target_column_name=target_column_name_for_dir,
        bag_size=args.bag_size, baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes,
        random_seed=args.random_seed, dev=args.dev, task_type=args.task_type, prefix=args.prefix,
        multiple_runs=args.multiple_runs
    )
    # Re-initialize logger for this main run if it wasn't set globally yet
    if logger is None: logger = get_logger(run_dir)
    logger.info(f"Initial args: {args}")

    # Data preparation
    train_dataset, eval_dataset, test_dataset = prepare_data(args)

    if args.task_type == 'regression': # Will be dict for MTL if mixed
        args.min_clip, args.max_clip = None, None # Placeholder for classification
    else:
        args.min_clip, args.max_clip = None, None

    # Dataloaders
    current_batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 128 # Default
    if (args.balance_dataset) & (args.task_type == "classification"):
        logger.info(f"Using weighted random sampler to balance the dataset (MTL balancing needs review)")
        # Simplified balancing based on the first task for now
        first_task_labels = [y_dict[args.label[0]] for y_dict in train_dataset.Y] # Y is pd.Series of dicts
        sample_weights = get_balanced_weights(first_task_labels)
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=current_batch_size, shuffle=False, num_workers=4)

    # Model dimensions
    args.input_dim = train_dataset.__getitem__(0)[0].shape[1] # Instance feature dimension
    if args.autoencoder_layer_sizes is None:
        args.state_dim = args.input_dim
    else:
        args.state_dim = args.autoencoder_layer_sizes[-1]
    # args.output_dims_dict is set in prepare_data

    logger.info(f"Output dimensions for tasks: {args.output_dims_dict}")
    logger.info(f"Train dataset size: {train_dataset.__len__()}")
    logger.info(f"Sample instance shape: {train_dataset.__getitem__(0)[0].shape}")
    logger.info(f"Sample label dict: {train_dataset.__getitem__(0)[1]}")


    if not args.no_wandb:
        label_str_for_wandb = "_".join(args.label) if isinstance(args.label, list) else args.label
        wandb_run_name = f"RL_MTL_{args.model_name}_{label_str_for_wandb}_{args.bag_size}_{args.prefix}"

        # Shorten the prefix for the tag
        prefix_for_tag = args.prefix
        if len(prefix_for_tag) > 50: # 50 is arbitrary, leaves room for "PREFIX_"
            prefix_for_tag = prefix_for_tag[:25] + "..." + prefix_for_tag[-22:] # Example truncation

        run = wandb.init(
            config=vars(args), # Log all args
            tags=[
                f"DATASET_{args.dataset}", f"BAG_SIZE_{args.bag_size}", f"BASELINE_{args.baseline}",
                f"LABELS_{label_str_for_wandb}", f"EMBEDDING_{args.embedding_model}",
                f"SEED_{args.random_seed}", f"PREFIX_{prefix_for_tag}"
            ],
            entity=args.wandb_entity, project=args.wandb_project, name=wandb_run_name,
        )

    # Model, Optimizer, Scheduler, EarlyStopping
    policy_network = create_rl_model(args, run_dir) # run_dir is for loading base MIL state
    policy_network = policy_network.to(DEVICE)

    # Optimizer for actor and critic (PolicyNetwork's own parameters)
    # The PolicyNetwork.task_optim handles the MIL model's parameters.
    actor_critic_optimizer = optim.AdamW(
        [{"params": policy_network.actor.parameters(), "lr": args.actor_learning_rate or 1e-4}, # Provide defaults if None
         {"params": policy_network.critic.parameters(), "lr": args.critic_learning_rate or 1e-4}],
        lr=args.actor_learning_rate or 1e-4, # Default overall LR for optimizer if needed
    )

    actor_critic_scheduler = optim.lr_scheduler.ExponentialLR(actor_critic_optimizer, gamma=0.9)

    early_stopping_main_run = EarlyStopping(
        models_dir=run_dir, save_model_name=f"checkpoint.pt", trace_func=logger.info,
        patience=args.early_stopping_patience, verbose=True, descending=True # Reward, so descending=True
    )

    train( # Call the modified train function
        policy_network=policy_network, optimizer=actor_critic_optimizer, scheduler=actor_critic_scheduler,
        early_stopping=early_stopping_main_run,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, test_dataloader=test_dataloader,
        device=DEVICE, bag_size=args.bag_size, epochs=args.epochs or 100, # Default epochs
        warmup_epochs=args.warmup_epochs, no_wandb=args.no_wandb,
        train_pool_size=args.train_pool_size or 1, eval_pool_size=args.eval_pool_size or 10, # Defaults
        test_pool_size=args.test_pool_size or 10, # Default
        run_name=None, # Not a sweep agent call
        only_ensemble=args.only_ensemble, rl_model=args.rl_model,
        prefix=args.prefix, epsilon=args.epsilon or 0.1, reg_coef=args.reg_coef or 0.01, # Defaults
        sample_algorithm=args.sample_algorithm or "without_replacement", args=args # Pass full args
    )
    torch.save(policy_network.state_dict(), os.path.join(run_dir, f"final_model.pt"))

    if not args.no_wandb:
        run.finish()


if __name__ == "__main__":
    args = parse_args()

    # Setup main logger and device globally for access in functions
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Determine run directory for logging based on potentially multiple labels
    target_column_name_for_dir_main = "_".join(args.label) if isinstance(args.label, list) else args.label
    main_run_dir = get_model_save_directory(
        dataset=args.dataset, data_embedded_column_name=args.data_embedded_column_name,
        embedding_model_name=args.embedding_model, target_column_name=target_column_name_for_dir_main,
        bag_size=args.bag_size, baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes,
        random_seed=args.random_seed, dev=args.dev, task_type=args.task_type, prefix=args.prefix,
        multiple_runs=args.multiple_runs
    )
    logger = get_logger(main_run_dir) # Initialize logger with the correct path

    logger.info(f"Global DEVICE set to: {DEVICE}")
    logger.info(f"Run directory: {main_run_dir}")

    # Model name (used for W&B run name prefixing, etc.)
    # Ensure get_model_name can handle args.autoencoder_layer_sizes if it's a list or None
    model_name_str = get_model_name(baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes)
    args.model_name = model_name_str

    if args.run_sweep:
        if not args.wandb_project:
            raise ValueError("wandb_project must be set for sweeps.")
        label_str_for_sweep = "_".join(args.label) if isinstance(args.label, list) else args.label
        args.sweep_config["name"] = f"{args.prefix}_{args.dataset}_{label_str_for_sweep}_rl_{args.baseline}".replace("_", "-")

        logger.info(f"Starting W&B sweep with ID: {args.sweep_config.get('name', 'N/A')}")
        sweep_id = wandb.sweep(args.sweep_config, entity=args.wandb_entity, project=args.wandb_project)
        wandb.agent(sweep_id, function=main_sweep, count=args.sweep_config.get('run_cap', 50)) # Pass main_sweep
    else:
        # args.run_name = f"{args.prefix}_{args.dataset}_{label_str_for_sweep}_rl_{args.baseline}_no_sweep" # Set for single run if needed
        main()