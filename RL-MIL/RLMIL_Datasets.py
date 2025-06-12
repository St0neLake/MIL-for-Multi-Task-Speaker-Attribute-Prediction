from itertools import compress
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class RLMILDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            bag_masks: torch.Tensor,
            subset: bool = True,
            task_type: str = "classification",
            instance_labels_column=None,
    ) -> None:
        self.X = df["bag_embeddings"]
        self.Y = df["labels"]
        self.bag = df["bag"]
        self.bag_true_mask = df["bag_mask"]
        self.bag_masks = bag_masks
        self.task_type = task_type
        self.subset = subset
        if instance_labels_column is not None:
            self.instance_labels = df[instance_labels_column]
        else:
            self.instance_labels = None

    def set_bag_mask(self, index: int, bag_mask: torch.Tensor) -> None:
        assert bag_mask.dtype == torch.bool, "bag_mask must be of type torch.bool"
        self.bag_masks[index] = bag_mask

    def set_subset(self, subset: bool) -> None:
        self.subset = subset

    def get_y(self, index: int):
        if self.task_type == "regression":
            return torch.tensor(self.Y[index]).float()
        elif self.task_type == "classification":
            return torch.tensor(self.Y[index]).long()

    def get_x(self, index: int):
        x = torch.tensor(self.X[index]).float()
        if self.subset:
            mask = self.bag_masks[index]
            return x[mask, :]
        else:
            return x

    def get_selected_text(self, index: int):
        mask = self.bag_true_mask[index] * self.bag_masks[index].tolist()
        selected_text = list(compress(self.bag[index], mask))
        return selected_text

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple:
        x = self.get_x(index)
        y = self.get_y(index)
        ys = np.array(self.instance_labels[index]) if self.instance_labels is not None else []
        return x, y, index, ys

class SimpleDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            task_type: str = "classification",
    ) -> None:
        self.X = np.stack(df["bag_embeddings"].tolist()).mean(axis=1)
        self.Y = df["labels"]
        self.bag = df["bag"]
        self.task_type = task_type

    def get_y(self, index: int):
        if self.task_type == "regression":
            return torch.tensor(self.Y[index]).float()
        elif self.task_type == "classification":
            return torch.tensor(self.Y[index]).long()

    def get_x(self, index: int):
        return  torch.tensor(self.X[index]).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple:
        x = self.get_x(index)
        y = self.get_y(index)
        return x, y
