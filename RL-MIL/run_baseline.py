import os

import torch
import wandb

from torch.utils.data import DataLoader, WeightedRandomSampler

from RLMIL_Datasets import SimpleDataset
from configs import parse_args
from logger import get_logger
from models import SimpleMLP
from utils import (
    AverageMeter,
    EarlyStopping,
    get_classification_metrics,
    get_crossentropy, get_mse,  get_r2_score, 
    get_data_directory,
    get_model_save_directory,
    get_model_name,
    read_data_split,
    create_preprocessed_dataframes,
    get_balanced_weights,
    set_seed, save_json
)


def classification():
    config = args
    run_name = f"bs={config.batch_size}_e={config.epochs}_lr={config.learning_rate}"
    wandb_tags = [
        f"BAG_SIZE_{args.bag_size}",
        f"BASELINE_{args.baseline}",
        f"LABEL_{args.label}",
        f"EMBEDDING_MODEL_{args.embedding_model}",
        f"DATASET_{args.dataset}",
        f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
        f"RANDOM_SEED_{args.random_seed}"
    ]
    if args.run_sweep:
        run = wandb.init(
            tags=wandb_tags
        )
        config = wandb.config
        run_name = f"bs={config.batch_size}_e={config.epochs}_lr={config.learning_rate}"
        run.name = run_name
    elif not args.no_wandb:
        run = wandb.init(
            tags=wandb_tags,
            entity=args.wandb_entity,
            project=args.wandb_project,
        )
        run.name = run_name

    logger.info(f"{config=}")
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    scheduler_patience = config.scheduler_patience
    early_stopping_patience = config.early_stopping_patience

    if (args.balance_dataset) & (args.task_type == "classification"):
        logger.info(f"Using weighted random sampler to balance the dataset")
        sample_weights = get_balanced_weights(train_dataset.Y.tolist())
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y.tolist()), replacement=True)
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

    model = SimpleMLP(
        input_dim=args.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=args.number_of_classes,
        dropout_p=config.dropout_p,
    )

    model = model.to(DEVICE)
    logger.info(f"{model=}")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=scheduler_patience
    )

    early_stopping = EarlyStopping(models_dir=run_dir, save_model_name="checkpoint.pt", trace_func=logger.info,
                                   patience=early_stopping_patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_average_meter = AverageMeter()

        for batch in train_dataloader:
            x, y = batch[0], batch[1]
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_average_meter.update(loss.item(), x.size(0))

        model.eval()
        val_loss = get_crossentropy(model, val_dataloader, DEVICE)

        scheduler.step(val_loss)
        train_classification_metrics = get_classification_metrics(
            model, train_dataloader, DEVICE
        )
        val_classification_metrics = get_classification_metrics(
            model, val_dataloader, DEVICE
        )

        if not args.no_wandb or args.run_sweep:
            wandb.log(
                {
                    "train/loss": train_average_meter.avg,
                    "train/accuracy": train_classification_metrics["acc"],
                    "train/precision": train_classification_metrics["precision"],
                    "train/recall": train_classification_metrics["recall"],
                    "train/f1": train_classification_metrics["f1"],
                    "eval/loss": val_loss,
                    "eval/accuracy": val_classification_metrics["acc"],
                    "eval/precision": val_classification_metrics["precision"],
                    "eval/recall": val_classification_metrics["recall"],
                    "eval/f1": val_classification_metrics["f1"],
                }
            )
        early_stopping(val_loss, model)
        if early_stopping.counter == 0:
            logger.info(f"Epoch: {epoch+1}/{epochs} train loss: {train_average_meter.avg:.6f}, accuracy: {train_classification_metrics['acc']:.6f}, precision: {train_classification_metrics['precision']:.6f}, recall: {train_classification_metrics['recall']:.6f}, f1: {train_classification_metrics['f1']:.6f}")
            logger.info(f"Epoch: {epoch+1}/{epochs} eval loss: {val_loss:.6f}, accuracy: {val_classification_metrics['acc']:.6f}, precision: {val_classification_metrics['precision']:.6f}, recall: {val_classification_metrics['recall']:.6f}, f1: {val_classification_metrics['f1']:.6f} ")

        global BEST_VAL_LOSS
        if val_loss < BEST_VAL_LOSS:
            logger.info(
                f"Found the best model in all of sweep runs in sweep run {run_name} at epoch {epoch}. Validation loss "
                f"decreased ({BEST_VAL_LOSS:.6f} --> {val_loss:.6f})."
            )
            logger.info(
                f"Saving the model in run {run_name}, with parameters {config=}"
            )
            BEST_VAL_LOSS = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(
                    run_dir,
                    "best_model.pt",
                ),
            )
            best_model_config = {}
            if args.run_sweep:
                args_dict = vars(args)
                config_dict = dict(config)
                for key in set(args_dict.keys()).union(config_dict.keys()):
                    if key in args_dict and key in config_dict:
                        best_model_config[key] = config_dict[key]
                    elif key in args_dict:
                        best_model_config[key] = args_dict[key]
                    else:
                        best_model_config[key] = config_dict[key]
            else:
                best_model_config = vars(args)

            save_json(
                path=os.path.join(run_dir, "best_model_config.json"),
                data=best_model_config,
            )
            
            test_loss = get_crossentropy(model, test_dataloader, DEVICE)
            test_classification_metrics = get_classification_metrics(model, test_dataloader, DEVICE)
            
            dictionary = {
                "model": args.baseline,
                "embedding_model": args.embedding_model,
                "bag_size": args.bag_size,
                "dataset": args.dataset,
                "label": args.label,
                "seed": args.random_seed,
                "test/loss": test_loss,
                "test/accuracy": test_classification_metrics["acc"],
                "test/precision": test_classification_metrics["precision"],
                "test/recall": test_classification_metrics["recall"],
                "test/f1": test_classification_metrics["f1"],
                "test/avg-f1": None,
                "test/ensemble-f1": None,
            }
            save_json(os.path.join(run_dir, "results.json"), dictionary)

        if early_stopping.early_stop:
            logger.info(
                f"Early stopping at epoch {epoch} out of {epochs}"
            )
            break

    logger.info(f"Loading the best model from early stopping checkpoint")
    model.load_state_dict(torch.load(early_stopping.model_address))
    model.eval()

    test_loss = get_crossentropy(model, test_dataloader, DEVICE)
    test_classification_metrics = get_classification_metrics(
        model, test_dataloader, DEVICE
    )
    if not args.no_wandb or args.run_sweep:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/accuracy": test_classification_metrics["acc"],
                "test/precision": test_classification_metrics["precision"],
                "test/recall": test_classification_metrics["recall"],
                "test/f1": test_classification_metrics["f1"],
            }
        )

        run.finish()
    logger.info(f"Test loss: {test_loss:.6f}, accuracy: {test_classification_metrics['acc']:.6f}, precision: {test_classification_metrics['precision']:.6f}, recall: {test_classification_metrics['recall']:.6f}, f1: {test_classification_metrics['f1']:.6f} ")

def regression():
    config = args
    run_name = f"bs={config.batch_size}_e={config.epochs}_lr={config.learning_rate}"
    wandb_tags = [
        f"BAG_SIZE_{args.bag_size}",
        f"BASELINE_{args.baseline}",
        f"LABEL_{args.label}",
        f"EMBEDDING_MODEL_{args.embedding_model}",
        f"DATASET_{args.dataset}",
        f"DATA_EMBEDDED_COLUMN_NAME_{args.data_embedded_column_name}",
        f"RANDOM_SEED_{args.random_seed}"
    ]
    if args.run_sweep:
        run = wandb.init(
            tags=wandb_tags
        )
        config = wandb.config
        run_name = f"bs={config.batch_size}_e={config.epochs}_lr={config.learning_rate}"
        run.name = run_name
    elif not args.no_wandb:
        run = wandb.init(
            tags=wandb_tags,
            entity=args.wandb_entity,
            project=args.wandb_project,
        )
        run.name = run_name

    logger.info(f"{config=}")
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    scheduler_patience = config.scheduler_patience
    early_stopping_patience = config.early_stopping_patience

    if args.balance_dataset:
        logger.info(f"Using weighted random sampler to balance the dataset")
        sample_weights = get_balanced_weights(train_dataset.Y.tolist())
        w_sampler = WeightedRandomSampler(sample_weights, len(train_dataset.Y.tolist()), replacement=True)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=w_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SimpleMLP(
        input_dim=args.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=args.number_of_classes,
        dropout_p=config.dropout_p,
    )

    model = model.to(DEVICE)
    logger.info(f"{model=}")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience)

    early_stopping = EarlyStopping(models_dir=run_dir, save_model_name="checkpoint.pt", trace_func=logger.info,
                                   patience=early_stopping_patience, verbose=True)

    for epoch in range(epochs):
        model.train()
        train_average_meter = AverageMeter()

        for x, y, _ in train_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat.squeeze(), y.squeeze())
            loss.backward()
            optimizer.step()

            train_average_meter.update(loss.item(), x.size(0))

        model.eval()
        val_loss = get_mse(model, val_dataloader, DEVICE)

        scheduler.step(val_loss)
        train_r2 = get_r2_score(model=model, dataloader=train_dataloader, device=DEVICE, 
                                min_clip=args.min_clip, max_clip=args.max_clip)
        val_r2 = get_r2_score(model=model, dataloader=val_dataloader, device=DEVICE, 
                              min_clip=args.min_clip, max_clip=args.max_clip)

        if not args.no_wandb or args.run_sweep:
            wandb.log(
                {
                    "train/loss": train_average_meter.avg,
                    "train/r2: ": train_r2,
                    "eval/loss": val_loss,
                    "eval/r2": val_r2
                }
            )
        early_stopping(val_loss, model)
        if early_stopping.counter == 0:
            logger.info(f"Epoch: {epoch+1}/{epochs} train loss: {train_average_meter.avg:.6f}, r2: {train_r2:.6f}")
            logger.info(f"Epoch: {epoch+1}/{epochs} eval loss: {val_loss:.6f}, r2: {val_r2:.6f}")

        global BEST_VAL_LOSS
        if val_loss < BEST_VAL_LOSS:
            logger.info(
                f"Found the best model in all of sweep runs in sweep run {run_name} at epoch {epoch}. Validation loss "
                f"decreased ({BEST_VAL_LOSS:.6f} --> {val_loss:.6f})."
            )
            logger.info(
                f"Saving the model in run {run_name}, with parameters {config=}"
            )
            BEST_VAL_LOSS = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(
                    run_dir,
                    "best_model.pt",
                ),
            )
            best_model_config = {}
            if args.run_sweep:
                args_dict = vars(args)
                config_dict = dict(config)
                for key in set(args_dict.keys()).union(config_dict.keys()):
                    if key in args_dict and key in config_dict:
                        best_model_config[key] = config_dict[key]
                    elif key in args_dict:
                        best_model_config[key] = args_dict[key]
                    else:
                        best_model_config[key] = config_dict[key]
            else:
                best_model_config = vars(args)

            save_json(
                path=os.path.join(run_dir, "best_model_config.json"),
                data=best_model_config,
            )
            
            test_loss = get_mse(model, test_dataloader, DEVICE)
            test_r2 = get_r2_score(model=model, dataloader=test_dataloader, device=DEVICE, min_clip=args.min_clip, max_clip=args.max_clip)
            
            dictionary = {
                "model": args.baseline,
                "embedding_model": args.embedding_model,
                "bag_size": args.bag_size,
                "dataset": args.dataset,
                "label": args.label,
                "seed": args.random_seed,
                "test/loss": test_loss,
                "test/r2": test_r2,
                "test/avg-r2": None,
                "test/ensemble-r2": None,
            }
            save_json(os.path.join(run_dir, "results.json"), dictionary)

        if early_stopping.early_stop:
            logger.info(
                f"Early stopping at epoch {epoch} out of {epochs}"
            )
            break

    logger.info(f"Loading the best model from early stopping checkpoint")
    model.load_state_dict(torch.load(early_stopping.model_address))
    model.eval()

    test_loss = get_mse(model, test_dataloader, DEVICE)
    test_r2 = get_r2_score(model=model, dataloader=test_dataloader, device=DEVICE, min_clip=args.min_clip, max_clip=args.max_clip)
    if not args.no_wandb or args.run_sweep:
        wandb.log(
            {
                "test/loss": test_loss,
                "test/r2": test_r2
            }
        )

        run.finish()
    logger.info(f"Test loss: {test_loss:.6f}, r2: {test_r2:.6f}")



if __name__ == "__main__":
    args = parse_args()
    args.autoencoder_layer_sizes = None
    if args.baseline != 'SimpleMLP':
            raise ValueError("Only running SimpleMLP")
    run_dir = get_model_save_directory(dataset=args.dataset,
                                       data_embedded_column_name=args.data_embedded_column_name,
                                       embedding_model_name=args.embedding_model, target_column_name=args.label,
                                       bag_size=args.bag_size, baseline=args.baseline,
                                       autoencoder_layers=args.autoencoder_layer_sizes, random_seed=args.random_seed, 
                                       dev=args.dev, task_type=args.task_type, prefix=None, multiple_runs=args.multiple_runs)
    logger = get_logger(run_dir)
    logger.info(f"{args=}")

    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"{DEVICE=}")
    logger.info(f"{args.run_sweep=}")

    set_seed(args.random_seed)

    BAG_EMBEDDED_COLUMN_NAME = "bag_embeddings"

    DATA_DIR = get_data_directory(args.dataset, args.data_embedded_column_name, args.random_seed)

    BEST_VAL_LOSS = float("inf")
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        logger.error(f"Data directory {DATA_DIR} does not exist.")
        raise ValueError("Data directory does not exist.")

    MODEL_NAME = get_model_name(args.baseline, args.autoencoder_layer_sizes)

    train_dataframe = read_data_split(DATA_DIR, args.embedding_model, "train")
    val_dataframe = read_data_split(DATA_DIR, args.embedding_model, "val")
    test_dataframe = read_data_split(DATA_DIR, args.embedding_model, "test")

    logger.info(
        f"{args.label} label distribution in train set before label encoding:"
    )
    logger.info(f"{train_dataframe[args.label].value_counts()}")
    logger.info(
        f"{args.label} label distribution in val set before label encoding:"
    )
    logger.info(f"{val_dataframe[args.label].value_counts()}")
    logger.info(
        f"{args.label} label distribution in test set before label encoding:"
    )
    logger.info(f"{test_dataframe[args.label].value_counts()}")

    train_dataframe, val_dataframe, test_dataframe, label2id, id2label = create_preprocessed_dataframes(
        train_dataframe,
        val_dataframe,
        test_dataframe,
        args.label,
        args.task_type
    )
    # Number of classes are the number of unique labels in train_dataframe
    args.number_of_classes = len(train_dataframe["labels"].unique())
    # If label2id and id2label were valid dictionaries, add them to args
    if label2id is not None and id2label is not None:
        args.label2id = label2id
        args.id2label = id2label

    # Assert that number of NaNs are zero
    assert train_dataframe.isnull().sum().sum() == 0
    assert val_dataframe.isnull().sum().sum() == 0
    assert test_dataframe.isnull().sum().sum() == 0

    logger.info(f"{train_dataframe.shape=}")
    logger.info(f"{val_dataframe.shape=}")
    logger.info(f"{test_dataframe.shape=}")

    logger.info(f"{args.label} label distribution in train set after label encoding:")
    logger.info(f"{train_dataframe['labels'].value_counts()}")
    logger.info(f"{args.label} label distribution in val set after label encoding:")
    logger.info(f"{val_dataframe['labels'].value_counts()}")
    logger.info(f"{args.label} label distribution in test set after label encoding:")
    logger.info(f"{test_dataframe['labels'].value_counts()}")

    train_dataset = SimpleDataset(
        df=train_dataframe,
        task_type=args.task_type,
    )
    val_dataset = SimpleDataset(
        df=val_dataframe,
        task_type=args.task_type,
    )
    test_dataset = SimpleDataset(
        df=test_dataframe,
        task_type=args.task_type,
    )

    # log train_dataset shape
    logger.info(f"{train_dataset.__len__()=}")
    logger.info(f"{train_dataset.__getitem__(0)[0].shape=}")
    logger.info(f"{train_dataset.__getitem__(0)[1].shape=}")
    logger.info(f"{train_dataset.__getitem__(0)[1]=}")

    args.input_dim = train_dataset.__getitem__(0)[0].shape[-1]

    if args.task_type == 'regression':
        args.min_clip = float(train_dataset.Y.min())
        args.max_clip = float(train_dataset.Y.max())
        
    torch.cuda.empty_cache()
    if args.run_sweep:
        # assert args.sweep_config is a dict
        args.sweep_config["name"] = f"{args.dataset}_{args.label}_{args.baseline}"
        logger.info(f"{args.sweep_config=}")
        sweep_id = wandb.sweep(args.sweep_config, entity=args.wandb_entity, project=args.wandb_project)
        if args.task_type == 'regression':
            wandb.agent(sweep_id, regression,)
        elif args.task_type == 'classification':
            wandb.agent(sweep_id, classification,)
    else:
        if args.task_type == 'regression':
            regression()
        elif args.task_type == 'classification':
            classification()
