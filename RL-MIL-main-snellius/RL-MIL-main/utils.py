import json
import os
import pickle
import random
from argparse import Namespace
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn import functional as F
from yaml.loader import SafeLoader
from typing import List, Dict, Any, Optional

from models import create_mil_model

SKLEARN_LABEL_ENCODER = LabelEncoder()

REGRESSION_LABELS = [
    "care",
    "purity",
    "equality",
    "proportionality",
    "loyalty",
    "authority",
    "fairness",
    "honor",
    "dignity",
    "face"
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_mse(model, test_generator, device):
    model.eval()
    loss = torch.nn.MSELoss()
    all_y = []
    all_y_hat = []
    for x, y, _ in test_generator:
        x = x.to(device)
        output = model(x)
        all_y_hat.append(output)
        all_y.append(y)

    yt = torch.cat(all_y_hat).squeeze().to("cpu")
    yt_hat = torch.cat(all_y).squeeze().to("cpu")
    # print(yt_hat.shape, yt.shape)
    mse = loss(yt_hat, yt).item()
    return mse


def get_crossentropy(model, dataloader, device):
    loss = torch.nn.CrossEntropyLoss()
    all_y = []
    all_y_hat = []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        output = model(x)
        all_y_hat.append(output)
        all_y.append(y)
        # all_y.append(y[:, 0])

    yt_hat = torch.cat(all_y_hat).squeeze().to("cpu")
    yt = torch.cat(all_y).squeeze()

    crossentropy_loss = loss(yt_hat, yt).item()
    return crossentropy_loss


def get_r2_score(model, dataloader, device, min_clip, max_clip):
    model.eval()
    all_y = []
    all_y_hat = []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        output = model(x)
        all_y_hat.append(output.cpu().detach().numpy())
        all_y.append(y.cpu().detach().numpy())

    yt = np.concatenate(all_y).squeeze()
    yt_hat = np.concatenate(all_y_hat).squeeze()
    score = r2_score_with_clip(yt, yt_hat, min_clip, max_clip)
    return score

def r2_score_with_clip(yt, yt_hat, min_clip, max_clip):
    yt_hat[yt_hat < min_clip] = min_clip
    yt_hat[yt_hat > max_clip] = max_clip
    score = r2_score(yt, yt_hat)
    return score


def get_classification_metrics(model, dataloader, device, average='macro', detailed=False):
    model.eval()
    all_y = []
    all_y_hat = []
    all_y_hat_prob = []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        x = x.to(device)
        output = model(x)
        output = F.softmax(output, dim=1)
        y_prob = torch.softmax(output, dim=1)
        y_hat = torch.argmax(output, dim=1).cpu()
        all_y_hat.extend(y_hat.tolist())
        y = y.to(torch.int64)
        all_y.extend(y.tolist())
        all_y_hat_prob.extend(y_prob.cpu().detach().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y, all_y_hat, average=average, zero_division=0
    )
    f1_micro = f1_score(all_y, all_y_hat, average="micro")
    all_y, all_y_hat_prob= np.array(all_y), np.array(all_y_hat_prob)
    # print(all_y)
    # print(all_y_hat_prob)
    # print(all_y.shape, all_y_hat_prob.shape)
    if all_y_hat_prob.shape[1] == 2:
        auc = roc_auc_score(all_y, all_y_hat_prob[:, 1], average='macro')
    else:
        auc = roc_auc_score(all_y, all_y_hat_prob, average='macro', multi_class='ovr')
    metrics = {
        "acc": accuracy_score(all_y, all_y_hat),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_micro": f1_micro,
        "auc": auc,
    }
    if not detailed:
        return metrics
    return metrics, all_y, all_y_hat, all_y_hat_prob


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_pickle(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, "rb") as f:
        dataset = pickle.load(f)
    return dataset


def load_yaml_file(file):
    with open(file, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_data_directory(dataset: str, data_embedded_column_name: str, random_seed: int) -> str:
    return os.path.join(
        os.path.dirname(__file__), "data", f"seed_{random_seed}", dataset, data_embedded_column_name
    )


def get_model_save_directory(dataset: str, data_embedded_column_name: str, embedding_model_name: str,
                             target_column_name: str, bag_size: int, baseline: str,
                             autoencoder_layers: Optional[List[int]], random_seed, dev, task_type, prefix, multiple_runs=False) -> str:
    model_name = get_model_name(baseline, autoencoder_layers)
    folder_name = "runs"
    if multiple_runs:
        folder_name = "multiple_" + folder_name
    if dev:
        folder_name = "dev_runs"
    if prefix:
        model_save_directory = os.path.join(
            os.path.dirname(__file__),
            folder_name,
            task_type,
            f"seed_{random_seed}",
            dataset,
            data_embedded_column_name,
            embedding_model_name,
            target_column_name,
            f"bag_size_{bag_size}",
            model_name,
            prefix,
        )
    else:
        model_save_directory = os.path.join(
            os.path.dirname(__file__),
            folder_name,
            task_type,
            f"seed_{random_seed}",
            dataset,
            data_embedded_column_name,
            embedding_model_name,
            target_column_name,
            f"bag_size_{bag_size}",
            model_name,
        )
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
    return model_save_directory

def get_model_name(baseline: str, autoencoder_layers: Optional[List[int]]) -> str:
    if autoencoder_layers:
        return baseline + "_" + "_".join([str(layer) for layer in autoencoder_layers])
    else:
        return baseline


def read_data_split(data_dir, embedding_model, split):
    df = pd.read_pickle(os.path.join(data_dir, embedding_model, f"{split}.pickle"))
    df = df.reset_index(drop=True)
    return df


def create_bag_masks(df, bag_size, bag_embedded_column_name):
    bag_masks = torch.zeros(
        (df.shape[0], df[bag_embedded_column_name][0].shape[0]), dtype=torch.bool
    )
    bag_masks[:, :bag_size] = 1
    return bag_masks


# def preprocess_dataframe(
#         df: pd.DataFrame,
#         dataframe_set: str,
#         label: str,
#         train_dataframe_mean: Optional[float],
#         train_dataframe_median: Optional[float],
#         train_dataframe_std: Optional[float],
#         task_type: str,
#         extra_columns: Optional[List[str]] = [],
# ):
#     # from IPython import embed; embed()
#     df = df[["bag_embeddings", "bag", "bag_mask", label] + extra_columns]
#     df = df.dropna()
#     df = df.reset_index(drop=True)
#     label2id = None
#     id2label = None
#     if task_type == "regression":
#         new_label = label
#     else:
#         if label in REGRESSION_LABELS:
#             new_label = f"{label}_encoded"
#             # 1 std before mean classifies as 0, between 1 std before and 1 std after mean classifies as 1, 1 std after
#             # mean classifies as 2

#             # 0 if below mean and 1 if above or equal to mean
#             # df[new_label] = df[label].apply(lambda x: 0 if x < train_dataframe_mean else 1)

#             # 0 if below median and 1 if above or equal to median
#             df[new_label] = df[label].apply(lambda x: 0 if x < train_dataframe_median else 1)
#         else:
#             new_label = f"{label}_encoded"
#             if dataframe_set == "train":
#                 df[new_label] = SKLEARN_LABEL_ENCODER.fit_transform(df[label])
#             else:
#                 df[new_label] = SKLEARN_LABEL_ENCODER.transform(df[label])

#             label2id = {label: idx for idx, label in enumerate(SKLEARN_LABEL_ENCODER.classes_.tolist())}
#             id2label = {idx: label for idx, label in enumerate(SKLEARN_LABEL_ENCODER.classes_.tolist())}
#     df = df[["bag_embeddings", "bag", "bag_mask", new_label] + extra_columns]
#     df = df.rename(columns={new_label: "labels"})
#     return df, label2id, id2label


def get_df_mean_median_std(df, label):
    """
    Receives a dataframe and a label and returns the mean, median and std of the label in the dataframe.
    :param df: dataframe
    :param label: label
    :return: mean, median and std
    """
    if label in REGRESSION_LABELS:
        return df[label].dropna().mean(), df[label].dropna().median(), df[label].dropna().std()
    return None, None, None


# def create_preprocessed_dataframes(train_dataframe: pd.DataFrame, val_dataframe: pd.DataFrame,
#                                    test_dataframe: pd.DataFrame, label: str, task_type: str, extra_columns: Optional[List[str]] = []):
#     train_dataframe_mean, train_dataframe_median, train_dataframe_std = get_df_mean_median_std(
#         train_dataframe, label
#     )

#     train_dataframe, label2id, id2label = preprocess_dataframe(df=train_dataframe, dataframe_set="train", label=label,
#                                                                train_dataframe_mean=train_dataframe_mean,
#                                                                train_dataframe_median=train_dataframe_median,
#                                                                train_dataframe_std=train_dataframe_std,
#                                                                task_type=task_type,
#                                                                extra_columns=extra_columns)
#     val_dataframe, _, _ = preprocess_dataframe(df=val_dataframe, dataframe_set="val", label=label,
#                                                train_dataframe_mean=train_dataframe_mean,
#                                                train_dataframe_median=train_dataframe_median,
#                                                train_dataframe_std=train_dataframe_std, task_type=task_type,
#                                                extra_columns=extra_columns)
#     test_dataframe, _, _ = preprocess_dataframe(df=test_dataframe, dataframe_set="test", label=label,
#                                                 train_dataframe_mean=train_dataframe_mean,
#                                                 train_dataframe_median=train_dataframe_median,
#                                                 train_dataframe_std=train_dataframe_std, task_type=task_type,
#                                                 extra_columns=extra_columns)

#     return train_dataframe, val_dataframe, test_dataframe, label2id, id2label


class EarlyStopping:
    def __init__(self, models_dir: str, save_model_name, trace_func, patience: int = 7, verbose: bool = False,
                 delta: float = 0.0, descending: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.models_dir = models_dir
        self.trace_func = trace_func
        self.descending = descending

        if save_model_name:
            self.model_address = os.path.join(self.models_dir, save_model_name)

    def __call__(self, val_loss, model):
        if self.descending:
            score = -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Score changed ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.val_loss_min = val_loss
        if self.model_address:
            torch.save(model.state_dict(), self.model_address)


def get_model(model_path: str, ensemble: bool = False):
    if ensemble:
        model_path = os.path.join(model_path, "only_ensemble_loss_sweep_best_rl_model.pt")
        p_model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model_state_dict = {}
        for k in p_model_state_dict.keys():
            if k.startswith("task_model."):
                model_state_dict[k.split("task_model.")[1]] = p_model_state_dict[k]
    else:
        model_path = os.path.join(model_path, "best_model.pt")
        model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model_name = model_path.split("/")[-2]
    baseline = model_name.split("_")[0]
    autoencoder_layers = list(map(int, model_name.split("_")[1:]))
    if "MLP" in model_name:
        args = Namespace(
            **{
                "dropout_p": 0.5,
                "input_dim": model_state_dict["mlp.0.weight"].size()[1],
                "hidden_dim": model_state_dict["mlp.0.weight"].size()[0],
                "number_of_classes": model_state_dict["mlp.3.bias"].size()[0],
                "autoencoder_layer_sizes": autoencoder_layers,
                "baseline": baseline,
            }
        )
    else:
        args = Namespace(
            **{
                "input_dim": autoencoder_layers[-1],
                "dropout_p": 0.5,
                "n_hidden_sets": model_state_dict["fc1.weight"].size()[1],
                "n_elements": model_state_dict["Wc"].size()[1] // model_state_dict["fc1.weight"].size()[1],
                "number_of_classes": model_state_dict["fc2.bias"].size()[0],
                "autoencoder_layer_sizes": autoencoder_layers,
                "baseline": baseline,
            }
        )

    model = create_mil_model(args)
    model.load_state_dict(model_state_dict)
    return model, args

def get_balanced_weights(labels):
    label_set = list(set(labels))
    label_set.sort()
    perfect_balance_weights = [len(labels)/labels.count(element) for element in label_set]

    sample_weights = [perfect_balance_weights[t] for t in labels]
    return sample_weights

def get_task_stats(df_train: pd.DataFrame, raw_col_name: str):
    """Calculates mean, median, std for a given raw column in the training dataframe."""
    if raw_col_name not in df_train.columns:
        raise ValueError(f"Raw column '{raw_col_name}' not found in training data.")
    series = df_train[raw_col_name].dropna().astype(float) # Ensure numeric for stats
    return series.mean(), series.median(), series.std()

def preprocess_single_task_inplace(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    task_config: Dict[str, Any],
    logger,
    train_stats: Optional[Dict[str, float]] = None # For binned classification or standardization
):
    """
    Preprocesses labels for a single task across train, val, test sets.
    Adds a new column like '{task_name}_processed' to each DataFrame.
    Returns preprocessing artifacts (scaler or label_encoder specific to this task, label2id, num_classes).
    """
    task_name = task_config['name']
    raw_col_name = task_config['raw_column_name']
    task_type = task_config['type']
    processed_col_name = f"{task_name}_processed"

    logger.info(f"Preprocessing task: '{task_name}' (Type: {task_type}, Raw Col: '{raw_col_name}')")

    task_artifacts = {'type': task_type, 'scaler': None, 'label_encoder': None, 'label2id': None, 'number_of_classes': None}

    if task_type == "regression_standardized":
        scaler = StandardScaler()
        df_train[processed_col_name] = scaler.fit_transform(df_train[[raw_col_name]].astype(float))
        if not df_val.empty:
            df_val[processed_col_name] = scaler.transform(df_val[[raw_col_name]].astype(float))
        if not df_test.empty:
            df_test[processed_col_name] = scaler.transform(df_test[[raw_col_name]].astype(float))
        task_artifacts['scaler'] = scaler
        logger.info(f"  Task '{task_name}': Standardized. Train Mean: {scaler.mean_[0]:.2f}, Train Std: {scaler.scale_[0]:.2f}")

    elif task_type == "classification_direct":
        le = LabelEncoder() # Use a new LabelEncoder for each 'classification_direct' task
        # Fit on all unique values across splits to ensure consistent encoding
        all_unique_labels = pd.concat([
            df_train[raw_col_name],
            df_val[raw_col_name] if not df_val.empty else pd.Series(dtype=str),
            df_test[raw_col_name] if not df_test.empty else pd.Series(dtype=str)
        ]).astype(str).unique()
        le.fit(all_unique_labels)

        df_train[processed_col_name] = le.transform(df_train[raw_col_name].astype(str))
        if not df_val.empty:
            df_val[processed_col_name] = le.transform(df_val[raw_col_name].astype(str))
        if not df_test.empty:
            df_test[processed_col_name] = le.transform(df_test[raw_col_name].astype(str))

        task_artifacts['label_encoder'] = le
        task_artifacts['label2id'] = {label: idx for idx, label in enumerate(le.classes_)}
        task_artifacts['id2label'] = {idx: label for label, idx in task_artifacts['label2id'].items()}
        task_artifacts['number_of_classes'] = len(le.classes_)
        logger.info(f"  Task '{task_name}': Label encoded. Classes: {task_artifacts['number_of_classes']}, Mapping: {task_artifacts['label2id']}")

    elif task_type == "classification_binned_median":
        if train_stats is None or 'median' not in train_stats:
            raise ValueError(f"Train median required for binned classification of task '{task_name}'.")

        median_val = train_stats['median']
        df_train[processed_col_name] = df_train[raw_col_name].apply(lambda x: 0 if float(x) < median_val else 1)
        if not df_val.empty:
            df_val[processed_col_name] = df_val[raw_col_name].apply(lambda x: 0 if float(x) < median_val else 1)
        if not df_test.empty:
            df_test[processed_col_name] = df_test[raw_col_name].apply(lambda x: 0 if float(x) < median_val else 1)

        # For binned, label2id is simple, and we can create a dummy LabelEncoder if needed by other parts
        le = LabelEncoder().fit([0,1]) # Dummy encoder for classes 0 and 1
        task_artifacts['label_encoder'] = le
        task_artifacts['label2id'] = {0: 0, 1: 1} # Or more descriptive if needed, e.g. {'BelowMedian':0, 'AboveEqMedian':1}
        task_artifacts['id2label'] = {v: k for k, v in task_artifacts['label2id'].items()}
        task_artifacts['number_of_classes'] = 2
        logger.info(f"  Task '{task_name}': Binarized by median ({median_val:.2f}). Classes: 2")

    else:
        raise ValueError(f"Unsupported task_type: '{task_type}' for task '{task_name}'")

    return task_artifacts


def create_mtl_dict_column_inplace(df: pd.DataFrame, task_configs: List[Dict[str, Any]], logger):
    """Adds/updates 'multi_task_labels' column with a dictionary of processed labels."""
    def create_label_dict(row):
        label_d = {}
        for tc in task_configs:
            task_name = tc['name']
            processed_col = f"{task_name}_processed"
            if processed_col in row:
                label_d[task_name] = row[processed_col]
            else:
                # This case should not happen if preprocess_single_task_inplace ran correctly for all tasks
                logger.warning(f"Processed column {processed_col} not found for row. Setting None for task {task_name}.")
                label_d[task_name] = None
        return label_d

    if not df.empty:
        df['multi_task_labels'] = df.apply(create_label_dict, axis=1)


# This function will replace your existing create_preprocessed_dataframes
# It assumes train_df, val_df, test_df are ALREADY SPLIT and passed to it.
# If splitting needs to happen inside, that logic needs to be integrated from your original.
def main_preprocess_for_mtl(
    args, # Should contain args.task_configs, output paths, dataset_name
    train_df_raw: pd.DataFrame,
    val_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    logger
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Main preprocessing function for MTL.
    Processes multiple tasks as defined in args.task_configs.
    Adds individual '{task_name}_processed' columns and a 'multi_task_labels' dict column.
    Saves processed dataframes and a dictionary of all preprocessing artifacts.
    """
    if not hasattr(args, 'task_configs') or not args.task_configs:
        logger.error("args.task_configs is not defined. Cannot preprocess multi-task labels.")
        raise ValueError("task_configs must be provided in args.")

    # Deep copy to avoid modifying original raw dataframes passed if they are used elsewhere
    train_df = train_df_raw.copy()
    val_df = val_df_raw.copy() if not val_df_raw.empty else pd.DataFrame()
    test_df = test_df_raw.copy() if not test_df_raw.empty else pd.DataFrame()

    all_task_artifacts = {}

    for task_config in args.task_configs:
        task_name = task_config['name']
        raw_col = task_config['raw_column_name']

        # Check if raw column exists
        for df_check, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
             if not df_check.empty and raw_col not in df_check.columns:
                msg = f"Raw column '{raw_col}' for task '{task_name}' not found in {name} DataFrame."
                logger.error(msg)
                raise ValueError(msg)

        train_stats_for_task = None
        if task_config['type'] == "classification_binned_median" or task_config['type'] == "regression_standardized_MANUAL": # if you want manual standardization
            mean, median, std = get_task_stats(train_df, raw_col)
            train_stats_for_task = {'mean': mean, 'median': median, 'std': std}
            # Store these base stats if needed, e.g. for reverting standardization or context
            all_task_artifacts[f"{task_name}_train_stats"] = train_stats_for_task

        task_artifacts_single = preprocess_single_task_inplace(
            train_df, val_df, test_df, task_config, logger, train_stats_for_task
        )
        all_task_artifacts[task_name] = task_artifacts_single

        # Update original task_config with determined number_of_classes if classification
        if task_artifacts_single.get('number_of_classes') is not None:
            # Find the task_config in args.task_configs list and update it
            for tc_original in args.task_configs:
                if tc_original['name'] == task_name:
                    tc_original['output_dim'] = task_artifacts_single['number_of_classes']
                    break

    # Create the 'multi_task_labels' dictionary column
    create_mtl_dict_column_inplace(train_df, args.task_configs)
    if not val_df.empty:
        create_mtl_dict_column_inplace(val_df, args.task_configs, logger)
    if not test_df.empty:
        create_mtl_dict_column_inplace(test_df, args.task_configs, logger)

    logger.info("Added 'multi_task_labels' dictionary column to DataFrames.")

    # --- Saving DataFrames and Preprocessing Info ---
    # Ensure output paths are defined in args (e.g., args.output_train_file_path)
    output_dir = os.path.dirname(args.output_train_file_path)
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_pickle(args.output_train_file_path)
    logger.info(f"Processed training data saved to {args.output_train_file_path}")
    if not val_df.empty and hasattr(args, 'output_val_file_path'):
        val_df.to_pickle(args.output_val_file_path)
        logger.info(f"Processed validation data saved to {args.output_val_file_path}")
    if not test_df.empty and hasattr(args, 'output_test_file_path'):
        test_df.to_pickle(args.output_test_file_path)
        logger.info(f"Processed test data saved to {args.output_test_file_path}")

    mtl_info_filename = f"{getattr(args, 'dataset_name', 'dataset')}_mtl_preprocessing_info.pkl"
    mtl_info_path = os.path.join(output_dir, mtl_info_filename)
    with open(mtl_info_path, 'wb') as f:
        pickle.dump(all_task_artifacts, f)
    logger.info(f"Multi-task preprocessing artifacts saved to {mtl_info_path}")

    return train_df, val_df, test_df, all_task_artifacts


# Remove or comment out the old single-task preprocess_dataframe and create_preprocessed_dataframes
# def preprocess_dataframe(...): # OLD VERSION
# def create_preprocessed_dataframes(...): # OLD VERSION