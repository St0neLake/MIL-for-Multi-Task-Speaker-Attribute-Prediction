from itertools import compress
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional


class RLMILDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            bag_masks: torch.Tensor,
            task_configs: List[Dict[str, Any]],
            subset: bool = True,
            instance_labels_df_column_name: Optional[str] = None
    ) -> None:
        self.df = df

        # Ensure the multi_task_labels column exists
        self.multi_task_labels_col_name = 'multi_task_labels' # Name used in utils.py
        if self.multi_task_labels_col_name not in self.df.columns:
            raise ValueError(f"Column '{self.multi_task_labels_col_name}' not found in DataFrame. Ensure utils.py preprocessing ran.")

        self.X_col = self.df["bag_embeddings"]  # Instance embeddings column
        self.multi_task_Y_col = self.df[self.multi_task_labels_col_name] # This is a Series of dicts

        # Original attributes, kept for structural similarity if underlying data structure is similar
        self.bag_col = self.df["bag"]
        self.bag_true_mask_col = self.df["bag_mask"]

        self.bag_masks = bag_masks
        self.task_configs = task_configs
        self.subset = subset

        self.instance_labels_col = None
        if instance_labels_df_column_name and instance_labels_df_column_name in self.df.columns:
            self.instance_labels_col = self.df[instance_labels_df_column_name]

    def set_bag_mask(self, index: int, bag_mask: torch.Tensor) -> None:
        assert bag_mask.dtype == torch.bool, "bag_mask must be of type torch.bool"
        self.bag_masks[index] = bag_mask

    def set_subset(self, subset: bool) -> None:
        self.subset = subset

    def get_y_dict(self, index: int) -> Dict[str, torch.Tensor]:
        y_dict_raw = self.multi_task_Y_col.iloc[index]
        y_dict_tensors = {}

        for task_config in self.task_configs:
            task_name = task_config['name']
            task_type = task_config['type']

            # if task_name not in y_dict_raw:
            #     # This should ideally not happen if utils.py created the dicts correctly
            #     print(f"Warning: Task '{task_name}' not found in label dictionary for sample index {index}.")
            #     # Decide how to handle: error, or skip, or default tensor. For now, skip.
            #     # y_dict_tensors[task_name] = torch.empty(0) # Or some placeholder
            #     continue

            label_value = y_dict_raw[task_name]

            if label_value is None or (isinstance(label_value, float) and np.isnan(label_value)):
                if task_type.startswith('regression'):
                    label_value = 0.0
                else:
                    label_value = 0 # Default class 0

            try:
                if task_type.startswith('regression'):
                    y_dict_tensors[task_name] = torch.tensor(label_value, dtype=torch.float32)
                elif task_type.startswith('classification'):
                    y_dict_tensors[task_name] = torch.tensor(label_value, dtype=torch.long)
                else:
                    y_dict_tensors[task_name] = torch.tensor(label_value)
            except Exception as e:
                print(f"Error converting label for task {task_name} (value: {label_value}) at index {index}: {e}")
                # Fallback tensor
                y_dict_tensors[task_name] = torch.tensor(0.0 if task_type.startswith('regression') else 0,
                                                        dtype=torch.float32 if task_type.startswith('regression') else torch.long)

        return y_dict_tensors

    def get_x(self, index: int) -> torch.Tensor:
        raw_x_data = self.X_col.iloc[index]
        if isinstance(raw_x_data, list) and raw_x_data and isinstance(raw_x_data[0], (list, np.ndarray)):
             x_np = np.array(raw_x_data, dtype=np.float32)
        elif isinstance(raw_x_data, pd.Series):
             x_np = raw_x_data.to_numpy(dtype=np.float32)
        else:
             x_np = np.asarray(raw_x_data, dtype=np.float32)

        x = torch.tensor(x_np, dtype=torch.float32)

        if self.subset:
            mask = self.bag_masks[index]
            # Ensure mask is not longer than number of instances in x
            if mask.size(0) > x.size(0):
                mask = mask[:x.size(0)]
            return x[mask, :]
        else:
            return x

    def get_selected_text(self, index: int) -> List[str]:
        true_mask_for_bag = self.bag_true_mask_col.iloc[index]
        policy_mask_for_bag = self.bag_masks[index].tolist()

        min_len = min(len(true_mask_for_bag), len(policy_mask_for_bag))

        combined_mask = [bool(tm and pm) for tm, pm in zip(true_mask_for_bag[:min_len], policy_mask_for_bag[:min_len])]

        current_bag_text = self.bag_col.iloc[index]
        # Ensure current_bag_text is also sliced if shorter than combined_mask
        selected_text = list(compress(current_bag_text[:len(combined_mask)], combined_mask))
        return selected_text

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple:
        x = self.get_x(index)
        y_dict = self.get_y_dict(index)

        instance_labels_tensor = torch.empty(0, dtype=torch.long)
        if self.instance_labels_col is not None:
            raw_instance_labels = self.instance_labels_col.iloc[index]
            if raw_instance_labels is not None:
                try:
                    instance_labels_np = np.array(raw_instance_labels, dtype=np.int64)
                    if instance_labels_np.ndim > 0 and instance_labels_np.size > 0 : # Check if not empty scalar
                         instance_labels_tensor = torch.tensor(instance_labels_np, dtype=torch.long)
                except Exception as e:
                    # print(f"Warning: Could not convert instance labels for index {index}: {e}")
                    pass # Keep default empty tensor

        # The original return format was x, y, index, ys
        # We now return x, y_dict, index, instance_labels_tensor
        return x, y_dict, index, instance_labels_tensor

class SimpleDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            task_configs: List[Dict[str, Any]],
            target_task_name: str,
    ) -> None:
        self.df = df
        self.task_configs = task_configs
        self.target_task_name = target_task_name

        # Find the config for the target task
        self.current_task_config = None
        for tc in self.task_configs:
            if tc['name'] == self.target_task_name:
                self.current_task_config = tc
                break
        if self.current_task_config is None:
            raise ValueError(f"Target task '{self.target_task_name}' not found in task_configs.")

        # X is mean of bag_embeddings (as in original)
        # Ensure 'bag_embeddings' contains lists/arrays of numbers
        processed_embeddings = []
        for i in range(len(df)):
            raw_embedding_bag = df["bag_embeddings"].iloc[i]
            if isinstance(raw_embedding_bag, (list, np.ndarray)) and len(raw_embedding_bag) > 0:
                bag_np = np.array([np.array(inst, dtype=np.float32) for inst in raw_embedding_bag if isinstance(inst, (list, np.ndarray)) and len(inst)>0])
                if bag_np.ndim == 2 and bag_np.shape[0] > 0: # Ensure it's not empty after filtering
                     processed_embeddings.append(bag_np.mean(axis=0))
                else:
                     print(f"Warning: Empty or malformed bag_embeddings at index {i} for SimpleDataset. Using zeros.")
                     feature_dim = 768
                     if any(isinstance(be, (list, np.ndarray)) and len(be)>0 and isinstance(be[0], (list,np.ndarray)) and len(be[0])>0 for be in df["bag_embeddings"] ):
                         for be_sample in df["bag_embeddings"]:
                             if isinstance(be_sample, (list, np.ndarray)) and len(be_sample)>0 and isinstance(be_sample[0], (list,np.ndarray)) and len(be_sample[0])>0:
                                 feature_dim = len(be_sample[0])
                                 break
                     processed_embeddings.append(np.zeros(feature_dim, dtype=np.float32))
            else:
                feature_dim = 768
                processed_embeddings.append(np.zeros(feature_dim, dtype=np.float32))

        self.X = np.array(processed_embeddings)


        # Y comes from the 'multi_task_labels' dictionary, for the specified target_task_name
        self.multi_task_labels_col_name = 'multi_task_labels'
        if self.multi_task_labels_col_name not in self.df.columns:
            raise ValueError(f"Column '{self.multi_task_labels_col_name}' not found in DataFrame for SimpleDataset.")

        # Extract the specific task label
        self.Y_target_task = self.df[self.multi_task_labels_col_name].apply(lambda dict_row: dict_row.get(self.target_task_name))
        if self.Y_target_task.isnull().any():
            print(f"Warning: Some labels for target task '{self.target_task_name}' are None in SimpleDataset.")

    def get_y(self, index: int):
        label_value = self.Y_target_task.iloc[index]
        task_type = self.current_task_config['type']

        if label_value is None or (isinstance(label_value, float) and np.isnan(label_value)):
            # print(f"Warning: Label for task '{self.target_task_name}' is None/NaN at index {index} in SimpleDataset. Assigning default.")
            label_value = 0.0 if task_type.startswith('regression') else 0

        dtype = torch.float32 if task_type.startswith('regression') else torch.long
        return torch.tensor(label_value, dtype=dtype)

    def get_x(self, index: int):
        return  torch.tensor(self.X[index], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df) # Based on original DataFrame length

    def __getitem__(self, index: int) -> tuple:
        x = self.get_x(index)
        y = self.get_y(index)
        # SimpleDataset in original RL-MIL didn't return index or instance_labels (ys)
        return x, y
