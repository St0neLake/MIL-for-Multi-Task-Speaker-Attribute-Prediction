import json
import os
import pickle
import random
from argparse import Namespace
from typing import Optional, List, Dict
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from yaml.loader import SafeLoader
from sklearn.cluster import KMeans

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


def get_classification_metrics(model, dataloader, device, target_task_name: str, average='macro', detailed=False): # ADDED target_task_name
    model.eval()
    all_y_task = []
    all_y_hat_task = []
    all_y_hat_prob_task = []

    for batch in dataloader:
        x, y_dict = batch[0], batch[1] # y_dict is a dict of label tensors {'task1': tensor, 'task2': tensor}
        x = x.to(device)

        output_dict = model(x) # model(x) should return a dict of output tensors {'task1': logits, 'task2': logits}

        # Use target_task_name to get the specific task's output and labels
        if target_task_name not in output_dict:
            available_keys = list(output_dict.keys()) if isinstance(output_dict, dict) else "Output is not a dict"
            raise ValueError(f"Task '{target_task_name}' not found in model output keys. Available: {available_keys}")
        if not isinstance(y_dict, dict) or target_task_name not in y_dict:
            available_keys = list(y_dict.keys()) if isinstance(y_dict, dict) else "Labels are not a dict"
            raise ValueError(f"Task '{target_task_name}' not found in label dictionary keys. Available: {available_keys}")

        output_task = output_dict[target_task_name]
        y_task = y_dict[target_task_name] # This is the tensor for the target task

        # Assuming output_task are logits for classification
        y_prob_task = F.softmax(output_task, dim=1)
        y_hat_task_batch = torch.argmax(output_task, dim=1).cpu()

        all_y_hat_task.extend(y_hat_task_batch.tolist())
        all_y_task.extend(y_task.cpu().tolist()) # Ensure y_task is on CPU and converted
        all_y_hat_prob_task.extend(y_prob_task.cpu().detach().tolist())

    if not all_y_task: # Handle empty evaluation case (e.g., if dataloader was empty)
        print(f"Warning: No data processed for task '{target_task_name}' in get_classification_metrics. Returning zero metrics.")
        metrics = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "f1_micro": 0.0, "auc": 0.0}
        if not detailed:
            return metrics
        return metrics, [], [], []

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_y_task, all_y_hat_task, average=average, zero_division=0
    )
    f1_micro = f1_score(all_y_task, all_y_hat_task, average="micro", zero_division=0)
    acc = accuracy_score(all_y_task, all_y_hat_task)

    all_y_np = np.array(all_y_task)
    all_y_hat_prob_np = np.array(all_y_hat_prob_task)

    auc = 0.0 # Default AUC
    if all_y_np.size > 0 and all_y_hat_prob_np.size > 0: # Check if not empty
        if len(np.unique(all_y_np)) > 1: # Needs at least two classes in y_true for AUC
            if all_y_hat_prob_np.ndim == 2 and all_y_hat_prob_np.shape[1] >= 2: # Check for valid probability array shape
                if all_y_hat_prob_np.shape[1] == 2: # Binary case probabilities (N, 2)
                    auc = roc_auc_score(all_y_np, all_y_hat_prob_np[:, 1], average='macro')
                else: # Multi-class case probabilities (N, num_classes)
                    try:
                        auc = roc_auc_score(all_y_np, all_y_hat_prob_np, average='macro', multi_class='ovr')
                    except ValueError as e:
                        print(f"Warning: Could not compute AUC for task {target_task_name} (multi-class): {e}. Setting AUC to 0.0. Ensure labels are correctly encoded and present in predictions.")
                        auc = 0.0
            else:
                print(f"Warning: AUC for task {target_task_name} could not be computed. Probability array shape issue: {all_y_hat_prob_np.shape}. Expected (n_samples, n_classes >= 2).")
        else:
            print(f"Warning: AUC for task {target_task_name} set to 0.0 as only one class present in true labels or labels are empty.")
    else:
        print(f"Warning: AUC for task {target_task_name} set to 0.0 due to empty labels or probabilities after processing.")

    metrics = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_micro": f1_micro,
        "auc": auc,
    }
    if not detailed:
        return metrics
    return metrics, all_y_task, all_y_hat_task, all_y_hat_prob_task


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


def preprocess_dataframe(
        df: pd.DataFrame,
        dataframe_set: str,
        target_labels: List[str],
        task_type: str,
        fitted_label_encoders: Optional[Dict[str, LabelEncoder]] = None,
        train_dataframe_medians: Optional[Dict[str, Optional[float]]] = None,
        extra_columns: Optional[List[str]] = [],
):
    required_cols = ["bag_embeddings", "bag", "bag_mask"] + target_labels + extra_columns
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col} in DataFrame for {dataframe_set} set.")

    df_processed = df[required_cols].copy()
    df_processed = df_processed.dropna(subset=target_labels).reset_index(drop=True)

    if df_processed.empty:
        print(f"Warning: DataFrame for {dataframe_set} is empty after initial dropna on target labels: {target_labels}.")
        # Prepare to return empty structures
        empty_labels_dict_col = pd.Series([{} for _ in range(len(df_processed))], dtype=object)
        df_processed["labels"] = empty_labels_dict_col
        final_columns_empty = ["bag_embeddings", "bag", "bag_mask", "labels"] + extra_columns
        df_final_empty = df_processed[final_columns_empty] # Ensure columns exist even if empty

        if dataframe_set == "train":
            return df_final_empty, {}, {}, {} # df, label2id, id2label, encoders
        else:
            return df_final_empty, {}, {} # df, label2id, id2label


    # This will store encoders fitted on train data if dataframe_set is "train"
    output_encoders = {} if dataframe_set == "train" else None # CHANGED: initialize for train
    all_label2id = {}
    all_id2label = {}
    processed_labels_dict_for_df_col = {}

    for label_name in target_labels:
        current_task_type_for_label = task_type # Use common task_type
        encoded_col_name = f"{label_name}_encoded"

        if label_name in REGRESSION_LABELS and current_task_type_for_label == "classification":
            if train_dataframe_medians is None or label_name not in train_dataframe_medians:
                raise ValueError(f"Median for '{label_name}' must be provided from training data for binarization in {dataframe_set} set.")
            median_val = train_dataframe_medians[label_name]
            if pd.isna(median_val):
                 raise ValueError(f"Median for '{label_name}' is NaN, cannot binarize for {dataframe_set} set.")

            df_processed[encoded_col_name] = df_processed[label_name].apply(lambda x: 0 if pd.isna(x) else (0 if x < median_val else 1))

            # For binarized labels, classes are [0, 1]
            temp_encoder = LabelEncoder()
            temp_encoder.classes_ = np.array(['0', '1']) # Consistent string representation
            all_label2id[label_name] = {label: idx for idx, label in enumerate(temp_encoder.classes_)}
            all_id2label[label_name] = {idx: label for idx, label in enumerate(temp_encoder.classes_)}
            if dataframe_set == "train" and output_encoders is not None:
                output_encoders[label_name] = temp_encoder # Store this effective encoder

        elif current_task_type_for_label == "classification": # Standard classification
            if dataframe_set == "train":
                label_encoder = LabelEncoder()
                df_processed[encoded_col_name] = label_encoder.fit_transform(df_processed[label_name].astype(str))
                if output_encoders is not None:
                    output_encoders[label_name] = label_encoder
            else: # val or test
                if fitted_label_encoders is None or label_name not in fitted_label_encoders:
                    raise ValueError(f"Fitted LabelEncoder for task '{label_name}' must be provided for {dataframe_set} set.")
                label_encoder = fitted_label_encoders[label_name]
                current_labels_str = df_processed[label_name].astype(str)
                # Create a mask for labels that are known to the encoder
                known_labels_mask = current_labels_str.isin(label_encoder.classes_)
                # Initialize encoded column with NaNs or a placeholder
                df_processed[encoded_col_name] = pd.Series(np.nan, index=df_processed.index)

                if known_labels_mask.any(): # If there are any known labels to transform
                    df_processed.loc[known_labels_mask, encoded_col_name] = label_encoder.transform(current_labels_str[known_labels_mask])

                if (~known_labels_mask).any():
                    unknown_found = current_labels_str[~known_labels_mask].unique()
                    print(f"Warning: {len(unknown_found)} unique unseen labels found in '{label_name}' for {dataframe_set} set (e.g., {list(unknown_found)[:5]}). Rows with these labels will have NaN for this task's encoded label.")

            all_label2id[label_name] = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            all_id2label[label_name] = {idx: label for idx, label in enumerate(label_encoder.classes_)}

        else: # Regression task
             df_processed[encoded_col_name] = df_processed[label_name].astype(float)
             # No label2id/id2label needed in the same way for regression
             all_label2id[label_name] = {}
             all_id2label[label_name] = {}

        processed_labels_dict_for_df_col[label_name] = df_processed[encoded_col_name]

    encoded_cols_to_check_for_na = [f"{lbl}_encoded" for lbl in target_labels if f"{lbl}_encoded" in df_processed.columns and task_type == "classification"] # Check only classification encoded cols
    if encoded_cols_to_check_for_na:
        df_processed.dropna(subset=encoded_cols_to_check_for_na, inplace=True)
        df_processed = df_processed.reset_index(drop=True)

    if df_processed.empty:
        print(f"Warning: DataFrame for {dataframe_set} became empty after processing all labels and dropping NaNs from encoded columns.")
        # Construct an empty 'labels' column structure
        empty_labels_series = pd.Series([{} for _ in range(len(df_processed))], index=df_processed.index, dtype=object)
        df_processed["labels"] = empty_labels_series
    else:
        # Create the 'labels' column as a dictionary of all processed labels
        df_processed["labels"] = [dict(zip(processed_labels_dict_for_df_col, t)) for t in zip(*processed_labels_dict_for_df_col.values())]


    final_columns = ["bag_embeddings", "bag", "bag_mask", "labels"] + extra_columns
    # Ensure all final columns are present
    for col in final_columns:
        if col not in df_processed.columns:
            if col == "labels" and "labels" not in df_processed: # if 'labels' couldn't be made due to empty intermediate
                 df_processed["labels"] = pd.Series([{} for _ in range(len(df_processed))], index=df_processed.index, dtype=object)
            else:
                raise ValueError(f"Final column '{col}' missing before final selection for {dataframe_set} set.")

    df_final = df_processed[final_columns].copy()

    if dataframe_set == "train":
        return df_final, all_label2id, all_id2label, output_encoders
    else:
        return df_final, all_label2id, all_id2label


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


def create_preprocessed_dataframes(train_dataframe: pd.DataFrame, val_dataframe: pd.DataFrame,
                                   test_dataframe: pd.DataFrame, target_labels: List[str], task_type: str, # task_type common for all
                                   extra_columns: Optional[List[str]] = []):

    train_dataframe_means = {}
    train_dataframe_medians = {}
    train_dataframe_stds = {}

    # Calculate stats for regression labels OR labels to be binarized FROM TRAINING DATA ONLY
    for label_name in target_labels:
        if (label_name in REGRESSION_LABELS and task_type == "classification") or task_type == "regression":
            mean, median, std = get_df_mean_median_std(train_dataframe, label_name)
            if median is not None: # Median is crucial for binarization
                train_dataframe_medians[label_name] = median
            if mean is not None: # Store if available
                train_dataframe_means[label_name] = mean
                train_dataframe_stds[label_name] = std

    # Process training data: fit encoders and get mappings
    train_dataframe_processed, train_label2id_map, train_id2label_map, fitted_encoders = preprocess_dataframe(
        df=train_dataframe, dataframe_set="train", target_labels=target_labels,
        task_type=task_type,
        # fitted_label_encoders=None, # Not needed for train
        train_dataframe_medians=train_dataframe_medians, # Pass medians for binarization
        # train_dataframe_means=train_dataframe_means, # Not directly used in preprocess_dataframe currently
        # train_dataframe_stds=train_dataframe_stds,   # Not directly used
        extra_columns=extra_columns
    )

    # Process validation data: use fitted encoders and train stats
    val_dataframe_processed, _, _ = preprocess_dataframe(
        df=val_dataframe, dataframe_set="val", target_labels=target_labels,
        task_type=task_type,
        fitted_label_encoders=fitted_encoders, # Pass the encoders from train
        train_dataframe_medians=train_dataframe_medians,
        extra_columns=extra_columns
    )

    # Process test data: use fitted encoders and train stats
    test_dataframe_processed, _, _ = preprocess_dataframe(
        df=test_dataframe, dataframe_set="test", target_labels=target_labels,
        task_type=task_type,
        fitted_label_encoders=fitted_encoders, # Pass the encoders from train
        train_dataframe_medians=train_dataframe_medians,
        extra_columns=extra_columns
    )

    return train_dataframe_processed, val_dataframe_processed, test_dataframe_processed, train_label2id_map, train_id2label_map



class EarlyStopping:
    def __init__(self, models_dir: str, save_model_name: Optional[str], trace_func,
                 patience: int = 7, verbose: bool = False,
                 delta: float = 0.0, descending: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.models_dir = models_dir
        self.trace_func = trace_func
        self.descending = descending

        self.best_metric_value = -np.Inf if descending else np.Inf
        self.best_epoch = -1

        if save_model_name:
            self.model_address = os.path.join(self.models_dir, save_model_name)
        else:
            self.model_address = None # Handle cases where no model saving is needed

    def __call__(self, current_metric_value, model, epoch: Optional[int] = None):
        improved = False

        if current_metric_value is None:
            if self.verbose:
                self.trace_func(f"EarlyStopping: current_metric_value is None at epoch {epoch if epoch is not None else 'N/A'}. No improvement recorded.")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(f"EarlyStopping: Stopping training due to patience limit with None metric. Best metric was {self.best_metric_value:.6f} at epoch {self.best_epoch}.")
            return # Exit if current metric is None to avoid comparison error

        if self.descending: # Higher is better
            if current_metric_value > self.best_metric_value + self.delta:
                improved = True
        else: # Lower is better (ascending)
            if current_metric_value < self.best_metric_value - self.delta:
                improved = True

        if improved:
            if self.verbose:
                old_best_for_log = "N/A" if (self.best_metric_value == -np.Inf or self.best_metric_value == np.Inf) else f"{self.best_metric_value:.6f}"
                self.trace_func(
                    f"EarlyStopping: Metric improved from {old_best_for_log} to {current_metric_value:.6f}."
                )
            self.best_metric_value = current_metric_value
            if epoch is not None:
                self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience} (Best: {self.best_metric_value:.6f} at epoch {self.best_epoch})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(f"EarlyStopping: Stopping training. Best metric {self.best_metric_value:.6f} at epoch {self.best_epoch}.")

    def save_checkpoint(self, model):
        """Saves model when the monitored metric improves."""
        if self.model_address:
            if self.verbose:
                self.trace_func(f"EarlyStopping: Saving model to {self.model_address} (Metric: {self.best_metric_value:.6f})")
            torch.save(model.state_dict(), self.model_address)
        elif self.verbose:
            self.trace_func("EarlyStopping: Improvement detected, but no model_address provided for saving.")


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

    config_file_path = os.path.join(os.path.dirname(model_path), "best_model_config.json")
    if not os.path.exists(config_file_path) and not ensemble: # For RL models, config might be one level up
        config_file_path_alt = os.path.join(os.path.dirname(os.path.dirname(model_path)), "best_model_config.json")
        if os.path.exists(config_file_path_alt):
            config_file_path = config_file_path_alt

    if os.path.exists(config_file_path):
        config = load_json(config_file_path)
        args_ns = Namespace(**config)
        if "number_of_classes" in config and "output_dims_dict" not in config and isinstance(config.get("label"), list):
            print("Warning: Loaded single-task config 'number_of_classes', but multiple labels suggest MTL. Ensure 'output_dims_dict' is correctly set in config or args for create_mil_model.")
        print(f"Warning: best_model_config.json not found at {config_file_path}. Inferring model args from path and state_dict (may be inaccurate for MTL).")
        model_name_from_path = os.path.basename(os.path.dirname(model_path)) # e.g., MeanMLP_768_256_768 or a prefix folder for RL
        if ensemble or "sweep_best_rl_model" in model_path : # if it's an RL model, path is deeper
             model_name_from_path = os.path.basename(os.path.dirname(os.path.dirname(model_path)))

        baseline = model_name_from_path.split("_")[0]
        autoencoder_layers_str = model_name_from_path.split("_")[1:]
        autoencoder_layers = [int(x) for x in autoencoder_layers_str if x.isdigit()] if any(s.isdigit() for s in autoencoder_layers_str) else None

        output_dims_dict_inferred = {}
        for key in model_state_dict.keys():
            if "task_heads" in key and "3.bias" in key: # Example key structure
                task_name = key.split('.')[1]
                output_dims_dict_inferred[task_name] = model_state_dict[key].size(0)

        if not output_dims_dict_inferred: # Fallback if no task_heads found
            num_classes_fallback = model_state_dict.get("mlp.3.bias", model_state_dict.get("fc2.bias", torch.tensor([1]))).size(0) # default to 1 if not found
            output_dims_dict_inferred = {"default_task": num_classes_fallback}

        args_dict_fallback = {
            "baseline": baseline,
            "autoencoder_layer_sizes": autoencoder_layers,
            "dropout_p": 0.5, # default
            "input_dim": model_state_dict.get(f"{baseline.lower() if 'MLP' in baseline else 'base_network.net'}.0.weight", next(iter(model_state_dict.values()))).size(1),
            "hidden_dim": model_state_dict.get(f"{baseline.lower() if 'MLP' in baseline else 'task_heads.placeholder'}.0.weight", next(iter(model_state_dict.values()))).size(0), # Placeholder
            "output_dims_dict": output_dims_dict_inferred # This is crucial for MTL
        }
        if "repset" in baseline and "fc1.weight" in model_state_dict:
            args_dict_fallback["n_hidden_sets"] = model_state_dict["fc1.weight"].size()[1]
            args_dict_fallback["n_elements"] = model_state_dict["Wc"].size()[1] // model_state_dict["fc1.weight"].size()[1]

        args_ns = Namespace(**args_dict_fallback)

    model = create_mil_model(args_ns) # create_mil_model MUST handle args_ns.output_dims_dict
    model.load_state_dict(model_state_dict, strict=False) # Use strict=False if some keys might not match (e.g. old single task to new MTL)
    return model, args_ns

def get_balanced_weights(labels):
    label_set = list(set(labels))
    label_set.sort()
    perfect_balance_weights = [len(labels)/labels.count(element) for element in label_set]


    sample_weights = [perfect_balance_weights[t] for t in labels] # Assign sample weights based on the sorted labels and perfect balance weights
    return sample_weights

def get_task_representations_from_activations(task_model, dataloader, task_names, device, task_to_cluster_id_map: dict):
    # Check if the model is the clustered version, otherwise this logic won't work.
    if not hasattr(task_model, 'cluster_trunks'):
        print("Error: get_task_representations_from_activations called on a non-clustered model.")
        # You might want to have a separate representation function for non-clustered models
        # or handle this case gracefully. For now, we return an empty dict.
        return {}

    task_model.eval()
    task_activation_accumulators = {task: [] for task in task_names}

    with torch.no_grad():
        for batch_x, _, _, _ in dataloader:
            batch_x = batch_x.to(device)

            # 1. Get aggregated bag features from the shared base_network
            shared_instance_embeddings = task_model.base_network(batch_x)
            x_aggregated_bag_features = task_model.aggregate(shared_instance_embeddings)

            # 2. For each task, route through its trunk, then get activations from the final head
            for task_name in task_names:
                cluster_id = task_to_cluster_id_map.get(task_name)

                if cluster_id is None:
                    print(f"Warning: Task {task_name} has no cluster assignment. Skipping for representation.")
                    continue

                # --- THIS IS THE KEY FIX ---
                # First, pass the shared features through the task's assigned cluster trunk
                trunk_output = task_model.cluster_trunks[cluster_id](x_aggregated_bag_features)

                # Now, get the representation from the first hidden layer of the FINAL task head,
                # using the trunk_output as its input.
                final_task_head_mlp = task_model.task_heads[task_name]

                # This assumes the first two layers of the final head are Linear and ReLU
                hidden_activations = final_task_head_mlp[1](final_task_head_mlp[0](trunk_output))
                # --- END OF FIX ---

                task_activation_accumulators[task_name].append(hidden_activations.cpu())

    task_representation_vectors = {}
    for task_name in task_names:
        if task_activation_accumulators[task_name]:
            all_activations_for_task = torch.cat(task_activation_accumulators[task_name], dim=0)
            task_representation_vectors[task_name] = torch.mean(all_activations_for_task, dim=0)
        else:
            # Fallback if no activations were collected
            hidden_dim = task_model.task_heads[task_names[0]][0].out_features
            task_representation_vectors[task_name] = torch.zeros(hidden_dim)
            print(f"Warning: No activations found for task {task_name} during representation generation.")

    return task_representation_vectors

def calculate_task_similarity_matrix(task_representation_vectors, task_names):
    # Initialize a square matrix with zeros, where the number of rows and columns is equal to the number of tasks.
    num_tasks = len(task_names)
    similarity_matrix = np.zeros((num_tasks, num_tasks))

    # Loop through each pair of tasks.
    for i in range(num_tasks):
        for j in range(num_tasks):
            if i == j:
                # If it's the same task, set the similarity to 1.0 (perfectly similar).
                similarity_matrix[i, j] = 1.0
            else:
                task_i_name = task_names[i]
                task_j_name = task_names[j]

                vec_i = task_representation_vectors.get(task_i_name)
                vec_j = task_representation_vectors.get(task_j_name)

                # If the representation vector is not found for either task, print a warning and set similarity to 0.
                if vec_i is None or vec_j is None:
                    print(f"Warning: Representation vector not found for {task_i_name} or {task_j_name}. Setting similarity to 0.")
                    similarity_matrix[i, j] = 0.0
                    continue

                # Calculate the cosine similarity between the two vectors and store it in the matrix.
                similarity = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0))
                similarity_matrix[i, j] = similarity.item()

    # Convert the similarity matrix to a pandas DataFrame for easier interpretation.
    similarity_df = pd.DataFrame(similarity_matrix, index=task_names, columns=task_names)
    return similarity_df

def assign_tasks_to_clusters(task_representation_vectors, task_names, num_clusters=2, random_state=None):
    # Check if there are enough tasks and representations
    if not task_representation_vectors or len(task_names) < num_clusters:
        print("Warning: Not enough tasks or representations to cluster, assigning all to cluster 0 or returning empty.")
        if not task_names:
            return {}
        return {task_name: 0 for task_name in task_names}

    # Extract feature vectors from representation vectors
    feature_matrix = []
    for task_name in task_names:
        if task_name in task_representation_vectors:
            feature_matrix.append(task_representation_vectors[task_name].cpu().numpy())
        else:
            print(f"Warning: Missing representation for task {task_name} in assign_tasks_to_clusters. Skipping.")

    # If not enough feature vectors, return a default assignment
    if not feature_matrix or len(feature_matrix) < num_clusters:
        print("Warning: Not enough task feature vectors for clustering after filtering. Assigning available to cluster 0.")
        return {task_name: 0 for task_name in task_representation_vectors.keys() if task_name in task_names}

    # Convert the list of feature vectors to a numpy array
    feature_matrix_np = np.array(feature_matrix)

    # If there are not enough tasks with representations, assign each to its own cluster or all to cluster 0
    if feature_matrix_np.shape[0] < num_clusters:
        print(f"Warning: Number of tasks with representations ({feature_matrix_np.shape[0]}) is less than num_clusters ({num_clusters}). Assigning each to its own cluster if possible, or all to 0.")
        if feature_matrix_np.shape[0] == 0: return {}
        return {task_names[i]: i for i in range(feature_matrix_np.shape[0])}

    # Fit a KMeans model to the feature matrix and assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init='auto')
    try:
        cluster_labels = kmeans.fit_predict(feature_matrix_np)
    except ValueError as e:
        print(f"Error during KMeans fitting: {e}. This can happen if n_samples < n_clusters. Assigning tasks to default clusters.")
        if feature_matrix_np.shape[0] < num_clusters:
             return {task_names[i]: i for i in range(feature_matrix_np.shape[0])}
        else:
            return {task_name: 0 for task_name in task_names}

    # Create a dictionary mapping each task name to its cluster ID
    task_to_cluster_id = {task_name: label for task_name, label in zip(task_names, cluster_labels)}

    # Return the dictionary of task-to-cluster assignments
    return task_to_cluster_id