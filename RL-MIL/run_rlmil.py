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
    get_df_mean_median_std,
    get_balanced_weights,
    EarlyStopping, save_json, load_json, create_mil_model
)

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
    for batch_x, batch_y, _, _  in train_dataloader:
        policy_network.train()
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        action_probs, _, _ = policy_network(batch_x)
        # logger.info(f"action_probs.shape={action_probs.shape}")
        action, action_log_prob = sample_action(action_probs, 
                                                bag_size, 
                                                device=device, 
                                                random=(epsilon > np.random.random()) or only_ensemble,
                                                algorithm=sample_algorithm)
        sel_x = select_from_action(action, batch_x)
        sel_y = batch_y
        sel_loss = policy_network.train_minibatch(sel_x, sel_y)
        sel_losses.append(sel_loss)
        policy_network.eval()
        # reward = policy_network.compute_reward(eval_data)
        if not only_ensemble:
            reward, _, _ = policy_network.expected_reward_loss(eval_pool)
            policy_network.saved_actions.append(action_log_prob)
            policy_network.rewards.append(reward)
            regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))

    
    if only_ensemble:
        return 0, 0, 0, np.mean(sel_losses), 0

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
        np.mean(sel_losses), reg_coef * regularization_loss.item()

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
    eval_pool = policy_network.create_pool_data(eval_dataloader, bag_size, train_pool_size)
    sel_losses, regularization_losses = [], []
    for batch_x, batch_y, indices in train_dataloader:
        policy_network.train()
        batch_x, batch_y, _ = batch_x.to(device), batch_y.to(device), indices.to(device)
        action_probs, _, exp_reward = policy_network(batch_x)
        action, action_log_prob = sample_action(action_probs, 
                                                bag_size, 
                                                device=device, 
                                                random=(epsilon > np.random.random()) or only_ensemble,
                                                algorithm=sample_algorithm)
        if not warmup:
            sel_x = select_from_action(action, batch_x)
            sel_y = batch_y
            sel_loss = policy_network.train_minibatch(sel_x, sel_y)
            sel_losses.append(sel_loss)
        else:
            sel_losses.append(0)
        policy_network.eval()
        # reward = policy_network.compute_reward(eval_data)
        if not only_ensemble:
            reward, _, _ = policy_network.expected_reward_loss(eval_pool)
            policy_network.saved_actions.append((action_log_prob, exp_reward))
            policy_network.rewards.append(reward)
            regularization_losses.append(action_probs.sum(dim=-1).mean(dim=-1))

    
    if only_ensemble:
        return 0, 0, 0, np.mean(sel_losses), 0
    
    policy_losses = []
    value_losses = []
    policy_network.train()
    # logger.debug(policy_network.rewards)
    # policy_network.normalize_rewards(eps)
    # logger.debug(policy_network.rewards)
    for (log_prob, exp_reward), reward in zip(policy_network.saved_actions, policy_network.rewards):
        advantage = torch.abs(reward - exp_reward)

        # calculate actor (policy) loss
        # logger.debug(advantage)
        # logger.debug(log_prob.cuda())
        # logger.debug(-advantage * log_prob.cuda())
        # logger.debug("_"*40)
        # logger.debug(advantage.shape, log_prob.shape)
        policy_losses.append(-advantage * log_prob.cuda())
        # logger.debug(policy_losses[-1].shape)
        # calculate critic (exp_reward) loss using L1 smooth loss
        R_tensor = torch.tensor([reward] * len(exp_reward))
        R_tensor = R_tensor.to(device)
        # logger.debug(F.smooth_l1_loss(exp_reward, R_tensor))
        # logger.debug(exp_reward.shape, R_tensor.shape)
        value_losses.append(F.smooth_l1_loss(exp_reward, R_tensor, reduction="none"))
        # logger.debug(F.smooth_l1_loss(exp_reward, R_tensor))
        # logger.debug("_"*40)
    # from IPython import embed; embed()
    # reset gradients
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_losses).sum()
    value_loss = torch.cat(value_losses).sum()
    regularization_loss = torch.stack(regularization_losses).mean() / 100
    total_loss = policy_loss + value_loss + reg_coef * regularization_loss
    # sum up all the values of policy_losses and value_losses
    # loss = torch.cat(policy_losses, dim=0).sum() + torch.stack(value_losses).sum()

    # perform backprop
    total_loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    # reset rewards and action buffer
    policy_network.reset_reward_action()

    return total_loss.item(), policy_loss.item(), value_loss.item(), \
        np.mean(sel_losses), regularization_loss.item()

def prepare_data(args):
    logger.info(f"Prepare datasets: DATA={args.dataset}, Column={args.data_embedded_column_name}")
    data_dir = get_data_directory(args.dataset, args.data_embedded_column_name, args.random_seed)
    if not os.path.exists(data_dir):
        raise ValueError("Data directory does not exist.")
    train_dataframe = read_data_split(data_dir, args.embedding_model, "train")
    val_dataframe = read_data_split(data_dir, args.embedding_model, "val")
    test_dataframe = read_data_split(data_dir, args.embedding_model, "test")

    train_dataframe_mean, train_dataframe_median, train_dataframe_std = get_df_mean_median_std(
        train_dataframe, args.label
    )
    if args.instance_labels_column is not None:
        extra_columns = [args.instance_labels_column]
    else:
        extra_columns = []
    train_dataframe, label2id, id2label = preprocess_dataframe(df=train_dataframe, dataframe_set="train", label=args.label,
                                           train_dataframe_mean=train_dataframe_mean,
                                           train_dataframe_median=train_dataframe_median,
                                           train_dataframe_std=train_dataframe_std, task_type=args.task_type,
                                           extra_columns=extra_columns)
    val_dataframe, _, _ = preprocess_dataframe(df=val_dataframe, dataframe_set="val", label=args.label,
                                         train_dataframe_mean=train_dataframe_mean,
                                         train_dataframe_median=train_dataframe_median,
                                         train_dataframe_std=train_dataframe_std, task_type=args.task_type,
                                         extra_columns=extra_columns)
    test_dataframe, _, _ = preprocess_dataframe(df=test_dataframe, dataframe_set="test", label=args.label,
                                          train_dataframe_mean=train_dataframe_mean,
                                          train_dataframe_median=train_dataframe_median,
                                          train_dataframe_std=train_dataframe_std, task_type=args.task_type,
                                          extra_columns=extra_columns)

    # If label2id and id2label were valid dictionaries, add them to args
    if label2id is not None and id2label is not None:
        args.label2id = label2id
        args.id2label = id2label

    train_dataset = RLMILDataset(
        df=train_dataframe,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )
    val_dataset = RLMILDataset(
        df=val_dataframe,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )
    test_dataset = RLMILDataset(
        df=test_dataframe,
        bag_masks=None,
        subset=False,
        task_type=args.task_type,
        instance_labels_column=args.instance_labels_column,
    )

    number_of_classes = len(train_dataframe["labels"].unique())

    return train_dataset, val_dataset, test_dataset, number_of_classes

def create_rl_model(args, mil_best_model_dir):
    if args.rl_task_model == "ensemble":
        for ensemble_dir in os.listdir(os.path.join(mil_best_model_dir, "..")):
            if "only_"+args.rl_task_model in ensemble_dir:
                mil_best_model_dir = os.path.join(mil_best_model_dir, "..", ensemble_dir)
                break
        logger.info(f"Loading ensemble model from {mil_best_model_dir}")
        ensemble_state_dict = torch.load(os.path.join(mil_best_model_dir, "sweep_best_model.pt"), map_location=torch.device("cpu"))
        state_dict = {}
        for k in ensemble_state_dict.keys():
            if k.startswith("task_model."):
                state_dict[k.split("task_model.")[1]] = ensemble_state_dict[k]
    else:
        state_dict = torch.load(os.path.join(mil_best_model_dir, "..", "best_model.pt"))
    task_model = load_mil_model_from_config(os.path.join(mil_best_model_dir, "..", "best_model_config.json"),
                                            state_dict)
    policy_network = PolicyNetwork(task_model=task_model,
                                   state_dim=args.state_dim,
                                   hdim=args.hdim,
                                   learning_rate=args.learning_rate,
                                   device=DEVICE,
                                   task_type=args.task_type,
                                   min_clip=args.min_clip,
                                   max_clip=args.max_clip,
                                   sample_algorithm=args.sample_algorithm,
                                   no_autoencoder=args.no_autoencoder_for_rl)
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

def get_first_batch_info(policy_network, eval_dataloader, device, bag_size, sample_algorithm):
    log_dict = {}
    batch_x, batch_y, indices, instance_labels = next(iter(eval_dataloader))
    batch_x = batch_x.to(device)
    action_probs, _, _ = policy_network(batch_x)
    action, _ = sample_action(action_probs, bag_size, device, random=False, algorithm=sample_algorithm)
    if len(instance_labels) != 0:
        instance_labels = instance_labels.to(device)
        selected_intance_labels = instance_labels[torch.arange(action.shape[0]).unsqueeze(1), action]
        selected_intance_count = selected_intance_labels.sum(dim=1)
    for i in range(action_probs.shape[0]):
        log_dict.update({f"actor/probs_{i}": action_probs[i].cpu().detach().numpy(),
                    f"actor/action_{i}": wandb.Histogram(action[i].cpu().numpy().tolist())})
        if len(instance_labels) != 0:
            if batch_y[i] == 1:
                log_dict.update({f"actor/selected_instance_count_{i}": selected_intance_count[i]})
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
        warmup_epochs=0,
        run_name=None,
        task_type='classification',
        only_ensemble=False, 
):
    if rl_model == 'policy_and_value':
        episode_function = finish_episode
    elif rl_model == 'policy_only':
        episode_function = finish_episode_policy_only
    
    metric = 'f1' if task_type == 'classification' else 'r2'
    
    # wandb.watch(policy_network.actor, log="all", log_freq=100, log_graph=True)
    if not no_wandb and not only_ensemble:
        log_dict = get_first_batch_info(policy_network, eval_dataloader, device, bag_size, sample_algorithm)
        wandb.log(log_dict)
    
    # logger.info(f"Training model started ....")
    for epoch in range(epochs):
        log_dict = {}
        warmup = epoch < warmup_epochs
        total_loss, policy_loss, value_loss, mil_loss, reg_loss = episode_function(
            policy_network=policy_network,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            bag_size=bag_size,
            train_pool_size=train_pool_size,
            warmup=warmup,
            only_ensemble=only_ensemble,
            epsilon=epsilon,
            reg_coef=reg_coef,
            sample_algorithm=sample_algorithm
        )
        # logger.info(f"Finished epoch {epoch}")
        # if not no_wandb and not only_ensemble:
        #     for indx, layer in enumerate(policy_network.actor.actor):
        #         if isinstance(layer, torch.nn.Linear):
        #             wandb.log({f"parameters/actor_{indx}_weight": wandb.Histogram(layer.weight.cpu().detach().numpy().tolist()),
        #                     f"parameters/actor_{indx}_bias": wandb.Histogram(layer.bias.cpu().detach().numpy().tolist())})
        #             wandb.log({f"gradients/actor_{indx}_weight": wandb.Histogram(layer.weight.grad.cpu().detach().numpy().tolist()),
        #                     f"gradients/actor_{indx}_bias": wandb.Histogram(layer.bias.grad.cpu().detach().numpy().tolist())})
        policy_network.eval()
        # eval_data = policy_network.select_from_dataloader(eval_dataloader, bag_size)
        eval_pool = policy_network.create_pool_data(eval_dataloader, bag_size, eval_pool_size, random=only_ensemble)
        reward, eval_loss, ensemble_reward = policy_network.expected_reward_loss(eval_pool)
        
        early_stopping(reward, policy_network)

        if not no_wandb:
            train_pool = policy_network.create_pool_data(train_dataloader, bag_size, eval_pool_size, random=only_ensemble)
            train_reward, _, train_ensemble_reward = policy_network.expected_reward_loss(train_pool)
            log_dict.update({"train/total_loss": total_loss,
                        "train/policy_loss": policy_loss,
                        "train/value_loss": value_loss,
                        "train/reg_loss": reg_loss,
                        "train/mil_loss": mil_loss,
                        "eval/avg_mil_loss": eval_loss,
                        f"train/avg_{metric}": train_reward,
                        f"train/ensemble_{metric}": train_ensemble_reward,
                        f"eval/avg_{metric}": reward,
                        f"eval/ensemble_{metric}": ensemble_reward})

            # log action probabilities
            if not only_ensemble:
                batch_log_dict = get_first_batch_info(policy_network, eval_dataloader, device, bag_size, sample_algorithm)
                log_dict.update(batch_log_dict)
         
            # log best model based on early stopping
            if early_stopping.counter == 0:
                log_dict.update({"best/eval_avg_mil_loss": eval_loss,
                            f"best/eval_avg_{metric}": reward,
                            f"best/eval_ensemble_{metric}": ensemble_reward})
            wandb.log(log_dict)

        if run_name:  # sweep
            global BEST_REWARD
            # print(f"ensemble rewards: {ensemble_reward:.6f}, Best rewaed: {BEST_REWARD:.6f}, Reward: {reward:.6f}")
            if ensemble_reward > BEST_REWARD:
                logger.info(
                    f"Found the best model in all of sweep runs in sweep run {run_name} at epoch {epoch}. ensemble F1 "
                    f"increased ({BEST_REWARD:.6f} --> {ensemble_reward:.6f})."
                )
                best_sweep_config = {
                    "critic_learning_rate": args.critic_learning_rate,
                    "actor_learning_rate": args.actor_learning_rate,
                    "learning_rate": args.learning_rate,
                    "epoch": args.epochs,
                    "hdim": args.hdim,
                }
                logger.info(
                    f"Saving the model in run {run_name}, with parameters config={best_sweep_config}"
                )
                BEST_REWARD = ensemble_reward
                torch.save(
                    policy_network.state_dict(),
                    os.path.join(
                        early_stopping.models_dir,
                        "sweep_best_model.pt",
                    ),
                )
                # TODO: move this part to utils
                best_model_config = {}
                args_dict = vars(args)
                config_dict = dict(best_sweep_config)
                for key in set(args_dict.keys()).union(config_dict.keys()):
                    if key in args_dict and key in config_dict:
                        best_model_config[key] = config_dict[key]
                    elif key in args_dict:
                        best_model_config[key] = args_dict[key]
                    else:
                        best_model_config[key] = config_dict[key]
                save_json(
                    path=os.path.join(early_stopping.models_dir, "sweep_best_model_config.json"),
                    data=best_model_config
                )
                policy_network.eval()
                # from IPython import embed; embed();
                test_pool = policy_network.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
                test_avg_reward, test_loss, test_ensemble_reward = policy_network.expected_reward_loss(test_pool)

                train_pool = policy_network.create_pool_data(train_dataloader, bag_size, eval_pool_size, random=only_ensemble)
                train_reward, _, train_ensemble_reward = policy_network.expected_reward_loss(train_pool)
            
                dictionary = {
                    "model": "rl-" + args.baseline,
                    "embedding_model": args.embedding_model,
                    "bag_size": args.bag_size,
                    "dataset": args.dataset,
                    "label": args.label,
                    "seed": args.random_seed,
                    "test/loss": test_loss,
                    f"test/{metric}": None,
                    f"test/avg-{metric}": test_avg_reward,
                    f"test/ensemble-{metric}": test_ensemble_reward,
                    f"train/avg-{metric}": train_reward,
                    f"train/ensemble-{metric}": train_ensemble_reward,
                    f"eval/avg-{metric}": reward,
                    f"eval/ensemble-{metric}": ensemble_reward
                }
                if task_type == 'classification':
                    dictionary.update({"test/accuracy": None,
                                       "test/precision": None,
                                       "test/recall": None,
                                       })

                save_json(os.path.join(early_stopping.models_dir, "results.json"), dictionary)
        # when warmup is done, reset early stopping
        if warmup_epochs + 1 == epoch:
            early_stopping.counter = 0
            early_stopping.early_stop = False
        # early stopping after warmup
        if early_stopping.early_stop and not warmup:
            logger.info(f"Early stopping at epoch {epoch} out of {epochs}")
            break

    # load the best model
    policy_network.load_state_dict(torch.load(early_stopping.model_address))
    policy_network.eval()
    
    test_pool = policy_network.create_pool_data(test_dataloader, bag_size, test_pool_size, random=only_ensemble)
    test_avg_reward, test_loss, test_ensemble_reward = policy_network.expected_reward_loss(test_pool)
    dictionary = {"test/avg_mil_loss": test_loss,
                  f"test/avg_{metric}": test_avg_reward,
                  f"test/ensemble_{metric}": test_ensemble_reward}
    if not no_wandb:
        wandb.log(dictionary)
    logger.info(dictionary)
    
    return policy_network


def main_sweep():
    run = wandb.init(
        tags=[
            f"DATASET_{args.dataset}",
            f"BAG_SIZE_{args.bag_size}",
            f"BASELINE_{args.baseline}",
            f"LABEL_{args.label}",
            f"EMBEDDING_MODEL_{args.embedding_model}",
        ],
    )
    config = wandb.config

    args.critic_learning_rate = config.critic_learning_rate
    args.actor_learning_rate = config.actor_learning_rate
    args.learning_rate = config.learning_rate
    args.epochs = config.epochs
    args.hdim = config.hdim
    args.early_stopping_patience = config.early_stopping_patience
    args.warmup_epochs = config.get("warmup_epochs", 0)
    args.epsilon = config.get("epsilon", 0)
    args.no_wandb = False
    # Model Optimizer Scheduler EarlyStopping
    policy_network = create_rl_model(args, run_dir)
    # from IPython import embed; embed(); exit()
    policy_network = policy_network.to(DEVICE)

    optimizer = optim.AdamW(
        [{"params": policy_network.actor.parameters(),
          "lr": args.actor_learning_rate,},
         {"params": policy_network.critic.parameters(),
          "lr": args.critic_learning_rate,}],
        lr=args.learning_rate,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2])
    early_stopping = EarlyStopping(models_dir=run_dir, save_model_name=f"sweep_checkpoint.pt",
                                   trace_func=logger.info,
                                   patience=args.early_stopping_patience, verbose=True, descending=False)

    policy_network = train(
        policy_network=policy_network,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        device=DEVICE,
        bag_size=args.bag_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        no_wandb=args.no_wandb,
        train_pool_size=args.train_pool_size,
        eval_pool_size=args.eval_pool_size,
        test_pool_size=args.test_pool_size,
        run_name=run.name,
        task_type=args.task_type,
        only_ensemble=args.only_ensemble,
        rl_model=args.rl_model,
        prefix=args.prefix,
        epsilon=args.epsilon,
        reg_coef=args.reg_coef,
        sample_algorithm=args.sample_algorithm
    )

    run.finish()


def main():
    if not args.no_wandb:
        run = wandb.init(
            tags=[
                f"DATASET_{args.dataset}",
                f"BAG_SIZE_{args.bag_size}",
                f"BASELINE_{args.baseline}",
                f"LABEL_{args.label}",
                f"ACTOR_LR_{args.actor_learning_rate}",
                f"CRITIC_LR_{args.critic_learning_rate}",
                f"MIL_LR_{args.learning_rate}",
                f"EMBEDDING_MODEL_{args.embedding_model}",
            ],
            config=args,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=f"RL_{args.model_name}_{args.label}_{args.bag_size}_2sided_ExponentialLR",
        )
        
    # Model Optimizer Scheduler EarlyStopping
    policy_network = create_rl_model(args, run_dir)
    policy_network = policy_network.to(DEVICE)

    optimizer = optim.AdamW([{"params": policy_network.actor.parameters(),
                              "lr": args.actor_learning_rate,},
                             {"params": policy_network.critic.parameters(),
                              "lr": args.critic_learning_rate,
                              },],
                            lr=args.learning_rate,)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader))
    # scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2])
    early_stopping = EarlyStopping(models_dir=run_dir,
                                   save_model_name=f"checkpoint.pt",
                                   trace_func=logger.info, patience=args.early_stopping_patience, verbose=True,
                                   descending=False)

    policy_network = train(
        policy_network=policy_network,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        test_dataloader=test_dataloader,
        device=DEVICE,
        bag_size=args.bag_size,
        epochs=args.epochs,
        no_wandb=args.no_wandb,
        train_pool_size=args.train_pool_size,
        eval_pool_size=args.eval_pool_size,
        test_pool_size=args.test_pool_size,
        task_type=args.task_type,
        only_ensemble=args.only_ensemble,
        rl_model=args.rl_model,
        epsilon=args.epsilon,
        run_name="no_sweep", # uncomment it to force to write the json result
        warmup_epochs=args.warmup_epochs,
        prefix=args.prefix,
        reg_coef=args.reg_coef,
        sample_algorithm=args.sample_algorithm
    )
    torch.save(policy_network.state_dict(),
                os.path.join(early_stopping.models_dir, f"model.pt",))

    if not args.no_wandb:
        run.finish()


if __name__ == "__main__":
    BEST_REWARD = float("-inf")
    args = parse_args()
    # Model name and directory
    run_dir = get_model_save_directory(dataset=args.dataset,
                                       data_embedded_column_name=args.data_embedded_column_name,
                                       embedding_model_name=args.embedding_model,
                                       target_column_name=args.label, 
                                       bag_size=args.bag_size,
                                       baseline=args.baseline,
                                       autoencoder_layers=args.autoencoder_layer_sizes,
                                       random_seed=args.random_seed,
                                       dev=args.dev, 
                                       task_type=args.task_type, 
                                       prefix=args.prefix,
                                       multiple_runs=args.multiple_runs)
    logger = get_logger(run_dir)
    logger.info(f"{args=}")

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE={DEVICE}")

    model_name = get_model_name(baseline=args.baseline, autoencoder_layers=args.autoencoder_layer_sizes)
    args.model_name = model_name

    # read data
    train_dataset, eval_dataset, test_dataset, number_of_classes = prepare_data(args)
    
    if args.task_type == 'regression':
        args.min_clip, args.max_clip = float(train_dataset.Y.min()), float(train_dataset.Y.max())
    else:
        args.min_clip, args.max_clip = None, None
        
    if (args.balance_dataset) & (args.task_type == "classification"):
        logger.info(f"Using weighted random sampler to balance the dataset")
        sample_weights = get_balanced_weights(train_dataset.Y.tolist())
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y.tolist()), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.number_of_classes = number_of_classes
    args.input_dim = train_dataset.__getitem__(0)[0].shape[1]
    if args.autoencoder_layer_sizes is None:
        args.state_dim = args.input_dim
    else:
        args.state_dim = args.autoencoder_layer_sizes[-1]

    logger.info(f"{number_of_classes=}")
    # log train_dataset shape
    logger.info(f"{train_dataset.__len__()=}")
    logger.info(f"{train_dataset.__getitem__(0)[0].shape=}")
    logger.info(f"{train_dataset.__getitem__(0)[1].shape=}")
    logger.info(f"{train_dataset.__getitem__(0)[1]=}")

    if args.run_sweep:
        args.sweep_config["name"] = f"{args.prefix}_{args.dataset}_{args.label}_rl_{args.baseline}".replace("_", "-")
        # sweep_config = args.pop("sweep_config")
        # sweep_config.update(vars(args))
        sweep_id = wandb.sweep(args.sweep_config, entity=args.wandb_entity, project=args.wandb_project)
        wandb.agent(sweep_id, main_sweep)
    else:
        args.run_name = f"{args.prefix}_{args.dataset}_{args.label}_rl_{args.baseline}_no_sweep"
        main()
