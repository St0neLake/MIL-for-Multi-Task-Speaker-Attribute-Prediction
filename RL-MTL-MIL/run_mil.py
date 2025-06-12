import os
import torch
import wandb # type: ignore
import numpy as np # CHANGED MTL: Added for potential metric aggregation
from torch.utils.data import DataLoader, WeightedRandomSampler
from argparse import Namespace

from RLMIL_Datasets import RLMILDataset
from configs import parse_args
from logger import get_logger
from models import MaxMLP, MeanMLP, AttentionMLP, ApproxRepSet, random_model, majority_model, create_mil_model # CHANGED MTL: Added create_mil_model
from utils import (
    AverageMeter,
    EarlyStopping,
    get_classification_metrics,
    get_crossentropy, get_mse,  get_r2_score,
    get_data_directory,
    get_model_save_directory,
    get_model_name,
    create_bag_masks,
    read_data_split,
    create_preprocessed_dataframes, # This should be your updated version
    get_balanced_weights,
    set_seed, save_json
)

# Global logger and device, initialized in main
logger = None
DEVICE = None
BEST_VAL_METRIC = float("inf") # For loss minimization, or -inf for maximization

# CHANGED MTL: This function will now handle multi-task classification
def classification_mtl(): # Renamed for clarity, original was classification()
    config = args # args from parse_args(), potentially overridden by wandb.config in sweep

    # CHANGED MTL: run_name might need to incorporate multiple labels if desired
    label_str_for_run_name = "_".join(sorted(args.label)) if isinstance(args.label, list) else args.label
    run_name = f"bs={config.batch_size}_e={config.epochs}_lr={config.learning_rate}_{label_str_for_run_name}"

    wandb_tags = [
        f"BAG_SIZE_{args.bag_size}",
        f"BASELINE_{args.baseline}",
        f"LABELS_{label_str_for_run_name}", # CHANGED MTL
        f"EMBEDDING_MODEL_{args.embedding_model}",
        f"DATASET_{args.dataset}",
        f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
        f"RANDOM_SEED_{args.random_seed}",
        "MTL_MIL_PreTune" # CHANGED MTL: Tag for this specific pre-tuning run
    ]

    if args.run_sweep:
        run = wandb.init(tags=wandb_tags) # config will be updated by wandb.config
        config = wandb.config # wandb.config contains the hyperparameters for this specific sweep trial
        # Update run name with sweep trial hyperparams for clarity in W&B
        run_name = f"bs={config.batch_size}_e={config.epochs}_lr={config.learning_rate}_{label_str_for_run_name}_sweep_trial"
        if hasattr(run, 'name') and run.name is not None: # W&B might auto-name it
            run.name = run_name
        # Update args with sweep config to ensure consistency if args is used later
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    elif not args.no_wandb:
        run = wandb.init(
            config=vars(args), # Log all args for single runs
            tags=wandb_tags,
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name
        )

    logger.info(f"Effective config for this run: {config}") # Log the config being used (from args or sweep)
    batch_size = config.batch_size # Use config from here on for hyperparams
    epochs = config.epochs
    learning_rate = config.learning_rate
    # scheduler_patience = config.scheduler_patience # from args or sweep config
    # early_stopping_patience = config.early_stopping_patience # from args or sweep config

    # Ensure scheduler_patience and early_stopping_patience are in config (from args default or sweep)
    scheduler_patience = getattr(config, 'scheduler_patience', args.scheduler_patience)
    early_stopping_patience = getattr(config, 'early_stopping_patience', args.early_stopping_patience)


    # CHANGED MTL: DataLoader now yields y_dict
    if (args.balance_dataset) & (args.task_type == "classification"):
        logger.info(f"Using weighted random sampler to balance the dataset for MTL (based on first task: {args.label[0]})")
        # For MTL, balancing is complex. This balances based on the first task's labels.
        # train_dataset.Y is a Series of dicts. We need labels for one task.
        first_task_labels = [y_dict[args.label[0]] for y_dict in train_dataset.Y]
        sample_weights = get_balanced_weights(first_task_labels)
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # CHANGED MTL: Model creation using args which should now have output_dims_dict
    # args.output_dims_dict is populated in the __main__ block
    # args.input_dim is also populated in __main__
    # config (from sweep) might override hidden_dim, dropout_p, etc.
    # We pass 'args' to create_mil_model, which now includes sweep-set values for some HPs
    # and pre-calculated input_dim and output_dims_dict.
    current_args_for_model = Namespace(**vars(args)) # Create a namespace from current args state
    # Update with sweep config values if any (e.g. hidden_dim, dropout_p)
    for hp_key in ['hidden_dim', 'dropout_p', 'n_elements', 'n_hidden_sets']:
        if hasattr(config, hp_key):
            setattr(current_args_for_model, hp_key, getattr(config, hp_key))

    model = create_mil_model(current_args_for_model) # create_mil_model uses args.output_dims_dict

    if model is None:
        logger.error(f"Model could not be created for baseline: {args.baseline}")
        return

    model = model.to(DEVICE)
    logger.info(f"MTL Model: {model}")

    # CHANGED MTL: Loss function - one for each task, then combined
    # Assuming all tasks are classification and use CrossEntropyLoss for this minimal change
    criterion_dict = {task_name: torch.nn.CrossEntropyLoss() for task_name in args.output_dims_dict.keys()}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience)

    # CHANGED MTL: Early stopping metric. For sweeps, this should match 'name' in YAML.
    # e.g., if YAML optimizes 'eval/avg_F1' (goal: maximize), then descending=True.
    # If YAML optimizes 'eval/combined_loss' (goal: minimize), then descending=False.
    # For this pre-tuning run, let's assume we minimize a combined loss for simplicity here,
    # or maximize an average F1. The YAML should guide this.
    # Let's use combined loss for early stopping here and make it a minimization problem.
    # The YAML for MeanMLP.yaml has metric: eval/loss, goal: minimize.
    early_stopping_metric_is_loss = True # Defaulting to loss for this MIL pre-tuning
    early_stopping_descending = False
    if args.run_sweep and hasattr(args.sweep_config, 'metric') and args.sweep_config['metric']['goal'] == 'maximize':
        early_stopping_descending = True
        early_stopping_metric_is_loss = False


    early_stopping = EarlyStopping(
        models_dir=run_dir,
        save_model_name="checkpoint.pt",
        trace_func=logger.info,
        patience=early_stopping_patience,
        verbose=True,
        descending=early_stopping_descending # True if maximizing (e.g. F1), False if minimizing (e.g. loss)
    )

    global BEST_VAL_METRIC # Use this to track the best metric for saving best_model.pt across sweep runs
    BEST_VAL_METRIC = -np.inf if early_stopping_descending else np.inf


    for epoch in range(epochs):
        model.train()
        train_combined_loss_meter = AverageMeter()
        # CHANGED MTL: Per-task training loss meters (optional for detailed logging)
        train_task_loss_meters = {task: AverageMeter() for task in args.output_dims_dict.keys()}

        for batch in train_dataloader:
            x, y_dict = batch[0], batch[1] # y_dict is {'age': tensor, 'gender': tensor, ...}
            x = x.to(DEVICE)

            optimizer.zero_grad()
            output_dict = model(x) # model returns {'age': logits, 'gender': logits, ...}

            current_batch_total_loss = 0
            for task_name, task_output in output_dict.items():
                task_labels = y_dict[task_name].to(DEVICE)
                loss = criterion_dict[task_name](task_output, task_labels.long()) # Assuming classification labels are long
                current_batch_total_loss += loss # Simple sum of losses
                train_task_loss_meters[task_name].update(loss.item(), x.size(0)) # Log individual task loss

            current_batch_total_loss.backward()
            optimizer.step()
            train_combined_loss_meter.update(current_batch_total_loss.item(), x.size(0))

        model.eval()

        # CHANGED MTL: Calculate combined validation loss and per-task metrics
        val_combined_loss_meter = AverageMeter()
        val_task_metrics = {task: {"f1": [], "acc": [], "auc": []} for task in args.output_dims_dict.keys()}
        all_val_task_f1s = []

        with torch.no_grad():
            for batch_val in val_dataloader:
                x_val, y_val_dict = batch_val[0], batch_val[1]
                x_val = x_val.to(DEVICE)

                output_val_dict = model(x_val)

                current_val_batch_total_loss = 0
                for task_name, task_output_val in output_val_dict.items():
                    task_labels_val = y_val_dict[task_name].to(DEVICE)
                    loss_val = criterion_dict[task_name](task_output_val, task_labels_val.long())
                    current_val_batch_total_loss += loss_val
                val_combined_loss_meter.update(current_val_batch_total_loss.item(), x_val.size(0))

        # Get detailed metrics for each task on the validation set
        wandb_log_dict = {"epoch": epoch, "train/combined_loss": train_combined_loss_meter.avg}
        for task_name in args.output_dims_dict.keys():
            # Log individual training task losses
            wandb_log_dict[f"train/{task_name}_loss"] = train_task_loss_meters[task_name].avg

            # Calculate and log validation metrics per task
            # Assuming get_classification_metrics from utils.py is adapted to take target_task_name
            # and model returns dict. If not, this part needs careful adaptation.
            # For minimal change, let's assume utils.get_classification_metrics can be called per task.
            # This implies passing the full val_dataloader again for each task metric, which is inefficient
            # but adheres to "minimal change" of this script.
            # A more efficient way would be to collect all_preds/all_labels in the loop above.

            # For now, let's log a combined metric that the sweep can use.
            # If sweep metric is 'eval/avg_F1', calculate it.
            # If sweep metric is 'eval/loss', use val_combined_loss_meter.avg.
            # We need to compute per-task metrics to get an avg_F1 or log them.
            # Re-iterating val_dataloader for metrics is inefficient but simpler for now.
            metrics_task_val, _, _, _ = get_classification_metrics(model, val_dataloader, DEVICE, target_task_name=task_name, detailed=True)
            wandb_log_dict[f"eval/{task_name}_accuracy"] = metrics_task_val["acc"]
            wandb_log_dict[f"eval/{task_name}_f1"] = metrics_task_val["f1"]
            wandb_log_dict[f"eval/{task_name}_auc"] = metrics_task_val["auc"]
            all_val_task_f1s.append(metrics_task_val["f1"])

        val_avg_f1 = np.mean(all_val_task_f1s) if all_val_task_f1s else 0
        wandb_log_dict["eval/combined_loss"] = val_combined_loss_meter.avg
        wandb_log_dict["eval/avg_F1"] = val_avg_f1 # Log this for sweep if YAML uses it

        scheduler.step(val_combined_loss_meter.avg) # Step scheduler based on combined validation loss

        if not args.no_wandb or args.run_sweep:
            wandb.log(wandb_log_dict)

        # CHANGED MTL: Determine metric for early stopping based on sweep config or default
        metric_to_monitor_for_early_stop = val_combined_loss_meter.avg
        if not early_stopping_metric_is_loss: # if maximizing (e.g. avg_F1)
            metric_to_monitor_for_early_stop = val_avg_f1

        early_stopping(metric_to_monitor_for_early_stop, model, epoch=epoch) # Pass epoch

        if early_stopping.counter == 0: # This means current epoch is the best so far
            logger.info(f"Epoch: {epoch+1}/{epochs} Train Combined Loss: {train_combined_loss_meter.avg:.6f}")
            logger.info(f"Epoch: {epoch+1}/{epochs} Eval Combined Loss: {val_combined_loss_meter.avg:.6f}, Eval Avg F1: {val_avg_f1:.6f}")
            for task_name in args.output_dims_dict.keys():
                 logger.info(f"  Eval {task_name} - F1: {wandb_log_dict[f'eval/{task_name}_f1']:.4f}, Acc: {wandb_log_dict[f'eval/{task_name}_accuracy']:.4f}, AUC: {wandb_log_dict[f'eval/{task_name}_auc']:.4f}")


        # Global best model saving (for the best run in a sweep, or best epoch in a single run)
        current_best_val_metric_for_saving = False
        if early_stopping_descending:
            if metric_to_monitor_for_early_stop > BEST_VAL_METRIC:
                BEST_VAL_METRIC = metric_to_monitor_for_early_stop
                current_best_val_metric_for_saving = True
        else:
            if metric_to_monitor_for_early_stop < BEST_VAL_METRIC:
                BEST_VAL_METRIC = metric_to_monitor_for_early_stop
                current_best_val_metric_for_saving = True

        if current_best_val_metric_for_saving: # If this epoch/run is the best overall so far
            logger.info(
                f"Found new overall best model (Metric: {BEST_VAL_METRIC:.6f}) at epoch {epoch} for run {run_name if not args.run_sweep else wandb.run.name}."
            )
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

            # Save config that led to this best_model.pt
            # This config should be args + sweep-defined HPs for this trial
            best_model_run_config = vars(args).copy() # Start with initial args
            if args.run_sweep: # Override with HPs from this specific sweep trial
                for key, value in config.items(): # config is wandb.config
                     if hasattr(best_model_run_config, key):
                        best_model_run_config[key] = value

            # Ensure output_dims_dict and input_dim are correctly in the saved config
            best_model_run_config['output_dims_dict'] = args.output_dims_dict
            best_model_run_config['input_dim'] = args.input_dim
            best_model_run_config['autoencoder_layer_sizes'] = args.autoencoder_layer_sizes # ensure this is saved
            if 'sweep_config' in best_model_run_config: # Don't save the whole sweep config object
                del best_model_run_config['sweep_config']


            save_json(path=os.path.join(run_dir, "best_model_config.json"), data=best_model_run_config)

            # CHANGED MTL: Test set evaluation per task
            test_combined_loss_meter = AverageMeter()
            test_task_metrics_summary = {}
            all_test_task_f1s = []

            with torch.no_grad():
                for batch_test in test_dataloader:
                    x_test, y_test_dict = batch_test[0], batch_test[1]
                    x_test = x_test.to(DEVICE)
                    output_test_dict = model(x_test)
                    current_test_batch_total_loss = 0
                    for task_name_test, task_output_test in output_test_dict.items():
                        task_labels_test = y_test_dict[task_name_test].to(DEVICE)
                        loss_test = criterion_dict[task_name_test](task_output_test, task_labels_test.long())
                        current_test_batch_total_loss += loss_test
                    test_combined_loss_meter.update(current_test_batch_total_loss.item(), x_test.size(0))

            results_json = {
                "model": args.baseline, "embedding_model": args.embedding_model,
                "bag_size": args.bag_size, "dataset": args.dataset,
                "labels": args.label, # List of labels
                "seed": args.random_seed,
                "test/combined_loss": test_combined_loss_meter.avg,
                # These are placeholders for single-task compatibility in results.json, MTL needs per-task
                "test/accuracy": None, "test/precision": None, "test/recall": None, "test/f1": None,
                "test/avg-f1": None, "test/ensemble-f1": None, # RL-MIL specific, None here
            }
            for task_name_test in args.output_dims_dict.keys():
                metrics_task_test, _, _, _ = get_classification_metrics(model, test_dataloader, DEVICE, target_task_name=task_name_test, detailed=True)
                results_json[f"test/{task_name_test}_accuracy"] = metrics_task_test["acc"]
                results_json[f"test/{task_name_test}_precision"] = metrics_task_test["precision"]
                results_json[f"test/{task_name_test}_recall"] = metrics_task_test["recall"]
                results_json[f"test/{task_name_test}_f1"] = metrics_task_test["f1"]
                results_json[f"test/{task_name_test}_auc"] = metrics_task_test["auc"]
                all_test_task_f1s.append(metrics_task_test["f1"])

            results_json["test/avg_F1_macro_across_tasks"] = np.mean(all_test_task_f1s) if all_test_task_f1s else 0
            save_json(os.path.join(run_dir, "results.json"), results_json)

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch+1} out of {epochs}")
            break

    logger.info(f"Loading the best model from early stopping checkpoint (epoch {early_stopping.best_epoch}) for final test evaluation.")
    if early_stopping.model_address and os.path.exists(early_stopping.model_address):
        model.load_state_dict(torch.load(early_stopping.model_address))
    else:
        logger.warning("No early stopping checkpoint found or model_address not set. Using model from last epoch for final test.")
    model.eval()

    # Final test evaluation (logging to WandB)
    final_test_log_dict = {}
    final_test_combined_loss_meter = AverageMeter()
    final_all_test_task_f1s = []

    with torch.no_grad(): # Recalculate test loss for loaded best model
        for batch_test in test_dataloader:
            x_test, y_test_dict = batch_test[0], batch_test[1]
            x_test = x_test.to(DEVICE)
            output_test_dict = model(x_test)
            current_test_batch_total_loss = 0
            for task_name_test, task_output_test in output_test_dict.items():
                task_labels_test = y_test_dict[task_name_test].to(DEVICE)
                loss_test = criterion_dict[task_name_test](task_output_test, task_labels_test.long())
                current_test_batch_total_loss += loss_test
            final_test_combined_loss_meter.update(current_test_batch_total_loss.item(), x_test.size(0))
    final_test_log_dict["test/combined_loss"] = final_test_combined_loss_meter.avg

    logger.info(f"Final Test Metrics (loaded best model from epoch {early_stopping.best_epoch}):")
    logger.info(f"  Test Combined Loss: {final_test_combined_loss_meter.avg:.6f}")

    for task_name in args.output_dims_dict.keys():
        metrics_task_final_test, _, _, _ = get_classification_metrics(model, test_dataloader, DEVICE, target_task_name=task_name, detailed=True)
        final_test_log_dict[f"test/{task_name}_accuracy"] = metrics_task_final_test["acc"]
        final_test_log_dict[f"test/{task_name}_f1"] = metrics_task_final_test["f1"]
        final_test_log_dict[f"test/{task_name}_auc"] = metrics_task_final_test["auc"]
        final_all_test_task_f1s.append(metrics_task_final_test["f1"])
        logger.info(f"  Test {task_name} - F1: {metrics_task_final_test['f1']:.4f}, Acc: {metrics_task_final_test['acc']:.4f}, AUC: {metrics_task_final_test['auc']:.4f}")

    final_test_avg_f1 = np.mean(final_all_test_task_f1s) if final_all_test_task_f1s else 0
    final_test_log_dict["test/avg_F1_macro_across_tasks"] = final_test_avg_f1
    logger.info(f"  Test Avg F1 across tasks: {final_test_avg_f1:.6f}")


    if not args.no_wandb or args.run_sweep:
        wandb.log(final_test_log_dict)
        if args.run_sweep: # For sweep, also log the primary metric chosen for optimization as a summary
             wandb.summary[args.sweep_config['metric']['name']] = metric_to_monitor_for_early_stop if early_stopping.best_epoch !=-1 else (val_avg_f1 if not early_stopping_metric_is_loss else val_combined_loss_meter.avg)

        run.finish()


# CHANGED MTL: Regression would also need similar MTL adaptation if used.
# For "minimal change", assuming classification_mtl is the focus.
def regression():
    # This function would need significant adaptation for MTL similar to classification_mtl:
    # - model returning dict of regression outputs
    # - per-task MSE loss, combined loss
    # - per-task R2 score for evaluation, combined metric for early stopping/sweep
    # - args.output_dims_dict where values are 1 for each regression task
    logger.error("MTL Regression not fully implemented in this minimal change version of run_mil.py")
    pass


if __name__ == "__main__":
    args = parse_args()

    # CHANGED MTL: Handle multiple labels for directory naming
    label_str_for_path = "_".join(sorted(args.label)) if isinstance(args.label, list) else args.label

    run_dir = get_model_save_directory(
        dataset=args.dataset,
        data_embedded_column_name=args.data_embedded_column_name,
        embedding_model_name=args.embedding_model,
        target_column_name=label_str_for_path, # CHANGED MTL
        bag_size=args.bag_size,
        baseline=args.baseline,
        autoencoder_layers=args.autoencoder_layer_sizes,
        random_seed=args.random_seed,
        dev=args.dev,
        task_type=args.task_type,
        prefix=None, # CHANGED MTL: Set prefix to None for MIL pre-tuning to save config in base folder
        multiple_runs=args.multiple_runs
    )
    logger = get_logger(run_dir)
    logger.info(f"Run arguments: {args}")

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE={DEVICE}")
    logger.info(f"Run sweep: {args.run_sweep}")

    set_seed(args.random_seed)

    BAG_EMBEDDED_COLUMN_NAME = "bag_embeddings" # This should be consistent
    DATA_DIR = get_data_directory(args.dataset, args.data_embedded_column_name, args.random_seed)

    # BEST_VAL_METRIC is initialized globally now
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} does not exist.")
        raise ValueError("Data directory does not exist.")

    MODEL_NAME = get_model_name(args.baseline, args.autoencoder_layer_sizes)

    train_dataframe_full = read_data_split(DATA_DIR, args.embedding_model, "train")
    val_dataframe_full = read_data_split(DATA_DIR, args.embedding_model, "val")
    test_dataframe_full = read_data_split(DATA_DIR, args.embedding_model, "test")

    # CHANGED MTL: create_preprocessed_dataframes now uses list of labels
    # It will also return label2id_map and id2label_map as dicts of dicts
    train_dataframe, val_dataframe, test_dataframe, label2id_map, id2label_map = create_preprocessed_dataframes(
        train_dataframe_full,
        val_dataframe_full,
        test_dataframe_full,
        args.label, # Pass the list of labels
        args.task_type
    )

    # CHANGED MTL: Populate args.output_dims_dict based on label2id_map from training data
    args.output_dims_dict = {}
    if isinstance(args.label, list):
        for task_label_name in args.label:
            if task_label_name in label2id_map:
                args.output_dims_dict[task_label_name] = len(label2id_map[task_label_name])
            else: # Should not happen if create_preprocessed_dataframes is correct
                 raise ValueError(f"Label2id mapping not found for task: {task_label_name}")
    else: # Single task case (though this script is being adapted for MTL)
        args.output_dims_dict = {args.label: len(label2id_map.get(args.label, {}))}

    logger.info(f"Output dimensions for tasks: {args.output_dims_dict}")


    # Assert that number of NaNs are zero in the 'labels' column (which is now a dict)
    # This check needs to be adapted if 'labels' contains dicts that might be empty
    # For now, assuming dropna in preprocess_dataframe handles rows with missing original labels
    # assert train_dataframe['labels'].apply(lambda x: all(pd.notna(val) for val in x.values())).all()
    # assert val_dataframe['labels'].apply(lambda x: all(pd.notna(val) for val in x.values())).all()
    # assert test_dataframe['labels'].apply(lambda x: all(pd.notna(val) for val in x.values())).all()
    logger.info(f"Train DataFrame shape after preprocessing: {train_dataframe.shape}")
    logger.info(f"Val DataFrame shape after preprocessing: {val_dataframe.shape}")
    logger.info(f"Test DataFrame shape after preprocessing: {test_dataframe.shape}")


    # This RLMILDataset needs to be able to handle the 'labels' column being a dictionary of labels
    # The __getitem__ in your RLMILDataset (snellius version) was already adapted for this.
    train_bag_masks = create_bag_masks(train_dataframe, args.bag_size, BAG_EMBEDDED_COLUMN_NAME)
    val_bag_masks = create_bag_masks(val_dataframe, args.bag_size, BAG_EMBEDDED_COLUMN_NAME)
    test_bag_masks = create_bag_masks(test_dataframe, args.bag_size, BAG_EMBEDDED_COLUMN_NAME)

    train_dataset = RLMILDataset(
        df=train_dataframe,
        bag_masks=train_bag_masks,
        subset=True, # For MIL, subset=True means instances are selected from bag based on bag_masks
        task_type=args.task_type,
    )
    val_dataset = RLMILDataset(
        df=val_dataframe,
        bag_masks=val_bag_masks,
        subset=True,
        task_type=args.task_type,
    )
    test_dataset = RLMILDataset(
        df=test_dataframe,
        bag_masks=test_bag_masks,
        subset=True,
        task_type=args.task_type,
    )

    logger.info(f"Train dataset length: {train_dataset.__len__()}")
    if train_dataset.__len__() > 0:
        sample_x, sample_y_dict, _, _ = train_dataset.__getitem__(0)
        logger.info(f"Sample instance batch shape from train_dataset: {sample_x.shape}")
        logger.info(f"Sample labels dict from train_dataset: {sample_y_dict}")
        args.input_dim = sample_x.shape[1] # Instance feature dimension
    else:
        logger.warning("Training dataset is empty after preprocessing. Cannot determine input_dim or proceed with training.")
        # Fallback or error if input_dim cannot be determined from data (e.g. from config if data is empty)
        # For now, this will likely cause issues later if training is attempted.
        # You might need to set args.input_dim from a config if data can be empty.
        # args.input_dim = some_default_or_config_value (e.g. 768 for roberta-base embeddings)
        # This depends on how robust you want this part to be for empty datasets.
        # If train_dataset is empty, subsequent dataloader and training will fail.
        # This indicates a severe issue with data preprocessing or filtering if it happens.
        if args.run_sweep and wandb.run: wandb.run.finish(exit_code=1) # Exit sweep if no data
        exit("Training dataset empty, cannot proceed.")


    if args.task_type == 'classification': # This script is now focused on classification_mtl
        if args.baseline == "random" or args.baseline == 'majority':
            logger.warning(f"Baseline {args.baseline} not adapted for MTL in this script version. Exiting.")
            # random_model(train_dataframe, test_dataframe, args, logger) # Needs MTL adaptation
            # majority_model(train_dataframe, test_dataframe, args, logger) # Needs MTL adaptation
            exit()
    elif args.task_type == 'regression':
        # args.min_clip, args.max_clip would need to be dicts for MTL regression
        logger.warning("MTL Regression part is not fully implemented in this version of run_mil.py")

    torch.cuda.empty_cache()
    if args.run_sweep:
        # Ensure sweep_config from args is used
        if not hasattr(args, 'sweep_config') or not args.sweep_config:
            raise ValueError("Sweep config not found in args. Load it in configs.py based on baseline for MIL sweeps.")

        # CHANGED MTL: The sweep name might need to reflect multiple labels
        args.sweep_config["name"] = f"{args.dataset}_{label_str_for_path}_{args.baseline}_MIL_PreTune" # CHANGED
        logger.info(f"Sweep configuration to be used: {args.sweep_config}")

        sweep_id = wandb.sweep(args.sweep_config, entity=args.wandb_entity, project=args.wandb_project)

        # The function passed to wandb.agent needs to be the main training/evaluation function
        # For this script, it's classification_mtl (or regression_mtl if implemented)
        if args.task_type == 'regression':
            # wandb.agent(sweep_id, regression_mtl, count=args.sweep_config.get('run_cap')) # if regression_mtl is defined
            logger.error("MTL Regression sweep not implemented yet.")
        elif args.task_type == 'classification':
            wandb.agent(sweep_id, classification_mtl, count=args.sweep_config.get('run_cap', 50)) # default run_cap
    else: # Single run (not a sweep agent)
        if args.task_type == 'regression':
            regression() # This would be regression_mtl()
        elif args.task_type == 'classification':
            classification_mtl()