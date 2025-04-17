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
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from yaml.loader import SafeLoader

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


def preprocess_dataframe(
        df: pd.DataFrame,
        dataframe_set: str,
        label: str,
        train_dataframe_mean: Optional[float],
        train_dataframe_median: Optional[float],
        train_dataframe_std: Optional[float],
        task_type: str,
        extra_columns: Optional[List[str]] = [],
):
    # from IPython import embed; embed()
    df = df[["bag_embeddings", "bag", "bag_mask", label] + extra_columns]
    df = df.dropna()
    df = df.reset_index(drop=True)
    label2id = None
    id2label = None
    if task_type == "regression":
        new_label = label
    else:
        if label in REGRESSION_LABELS:
            new_label = f"{label}_encoded"
            # 1 std before mean classifies as 0, between 1 std before and 1 std after mean classifies as 1, 1 std after
            # mean classifies as 2

            # 0 if below mean and 1 if above or equal to mean
            # df[new_label] = df[label].apply(lambda x: 0 if x < train_dataframe_mean else 1)

            # 0 if below median and 1 if above or equal to median
            df[new_label] = df[label].apply(lambda x: 0 if x < train_dataframe_median else 1)
        else:
            new_label = f"{label}_encoded"
            if dataframe_set == "train":
                df[new_label] = SKLEARN_LABEL_ENCODER.fit_transform(df[label])
            else:
                df[new_label] = SKLEARN_LABEL_ENCODER.transform(df[label])

            label2id = {label: idx for idx, label in enumerate(SKLEARN_LABEL_ENCODER.classes_.tolist())}
            id2label = {idx: label for idx, label in enumerate(SKLEARN_LABEL_ENCODER.classes_.tolist())}
    df = df[["bag_embeddings", "bag", "bag_mask", new_label] + extra_columns]
    df = df.rename(columns={new_label: "labels"})
    return df, label2id, id2label


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
                                   test_dataframe: pd.DataFrame, label: str, task_type: str, extra_columns: Optional[List[str]] = []):
    train_dataframe_mean, train_dataframe_median, train_dataframe_std = get_df_mean_median_std(
        train_dataframe, label
    )

    train_dataframe, label2id, id2label = preprocess_dataframe(df=train_dataframe, dataframe_set="train", label=label,
                                                               train_dataframe_mean=train_dataframe_mean,
                                                               train_dataframe_median=train_dataframe_median,
                                                               train_dataframe_std=train_dataframe_std,
                                                               task_type=task_type,
                                                               extra_columns=extra_columns)
    val_dataframe, _, _ = preprocess_dataframe(df=val_dataframe, dataframe_set="val", label=label,
                                               train_dataframe_mean=train_dataframe_mean,
                                               train_dataframe_median=train_dataframe_median,
                                               train_dataframe_std=train_dataframe_std, task_type=task_type,
                                               extra_columns=extra_columns)
    test_dataframe, _, _ = preprocess_dataframe(df=test_dataframe, dataframe_set="test", label=label,
                                                train_dataframe_mean=train_dataframe_mean,
                                                train_dataframe_median=train_dataframe_median,
                                                train_dataframe_std=train_dataframe_std, task_type=task_type,
                                                extra_columns=extra_columns)

    return train_dataframe, val_dataframe, test_dataframe, label2id, id2label


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