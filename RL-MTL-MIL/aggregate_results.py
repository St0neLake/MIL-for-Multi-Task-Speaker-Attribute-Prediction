import os
import re
import torch
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader
from RLMIL_Datasets import RLMILDataset
from models import create_mil_model_with_dict
from models import PolicyNetwork
from utils import (load_json, save_json, read_data_split, 
                   get_df_mean_median_std, preprocess_dataframe, 
                   create_bag_masks, get_crossentropy, get_r2_score, get_mse,
                   get_classification_metrics, set_seed)


def load_mil_model(model_path):
    config = load_json(os.path.join(model_path, "best_model_config.json"))
    saved_model_path = os.path.join(model_path, 'best_model.pt')
    
    model = create_mil_model_with_dict(config)
    model.load_state_dict(torch.load(saved_model_path))

    return model

def load_rlmil_model(model_path, device):
    rl_config = load_json(os.path.join(model_path, "sweep_best_model_config.json"))
    task_model = load_mil_model(os.path.join(model_path, "../"))
    policy_network = PolicyNetwork(task_model=task_model,
                                   state_dim=rl_config['state_dim'],
                                   hdim=rl_config['hdim'],
                                   learning_rate=0,
                                   sample_algorithm=rl_config['sample_algorithm'],
                                   task_type=rl_config['task_type'],
                                   min_clip=rl_config['min_clip'],
                                   max_clip=rl_config['max_clip'],
                                   no_autoencoder_for_rl=rl_config['no_autoencoder_for_rl'],
                                   device=device)

    saved_model_path = os.path.join(model_path, 'sweep_best_model.pt')
    policy_network.load_state_dict(torch.load(saved_model_path))
    policy_network.to(device)
    return policy_network

def agg_all_results(device, data_dir_path, models_dir_path, bag_embedded_column_name,  task_type, config_vars=[]):

    mil_results = agg_mil_results(device, data_dir_path, models_dir_path, 
                                  bag_embedded_column_name,  task_type, config_vars)
    rlmil_results = agg_rlmil_results(device, data_dir_path, models_dir_path, 
                                      bag_embedded_column_name,  task_type, config_vars)
    
    df = pd.concat([mil_results, rlmil_results])
    return df

def agg_rlmil_results(device, data_dir_path, models_dir_path, bag_embedded_column_name,  task_type, config_vars=[], in_seeds=[0, 1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9]):

    pattern = fr"{models_dir_path}/{task_type}/seed_(\d+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/bag_size_(\d+)/([^/]+)/([^/]+)$"
    regex = re.compile(pattern)
    
    metric_name = 'f1' if task_type == 'classification' else 'r2'
    df = {
            "model": [],
            "embedding_model": [],
            "bag_size": [],
            "dataset": [],
            "label": [],
            "seed": [],
            "prefix": [],
            "test/loss": [],
            "eval/loss": [],
            "test/auc": [],
            "test/f1": [],
            "eval/auc": [],
            "eval/f1": [],
            "eval/f1_micro": [],
            "test/f1_micro": []
    }
    for k in config_vars:
        df[k] = []

    for root, _, files in os.walk(models_dir_path):
        for file in files:
            if file == "log.txt":
                # print(root)
                match = regex.search(root)
                # print(match)
                if match:
                    # print(match.groups())
                    (random_seed, 
                    dataset,
                    data_embedded_column_name,
                    embedding_model_name,
                    target_column_name,
                    bag_size,
                    model_name,
                    prefix) = match.groups()
                    if int(random_seed) not in in_seeds:
                        continue
                    metrics = {}
                    results_file = os.path.join(root, "results.json")
                    if os.path.isfile(results_file):
                        metrics = load_json(results_file)
                        metrics["seed"] = random_seed
                        metrics['prefix'] = prefix
                    if (f"eval/{metric_name}" not in metrics.keys()) or (os.path.isfile(os.path.join(root, "model_outputs.csv")) == False):
                        # print(f"Testing RL-MIL model for {dataset, model_name, target_column_name, prefix}!")
                        # try:
                        print(root)
                        set_seed(int(random_seed))
                        metrics, csv_df = generate_result_for_rlmil_model(root, device, random_seed, data_dir_path, dataset, 
                                                                data_embedded_column_name, embedding_model_name, target_column_name, 
                                                                bag_size, bag_embedded_column_name, task_type=task_type)
                        if len(metrics) != 0:
                            metrics.update({
                                "model": "rl-" + model_name.split("_")[0],
                                "embedding_model": embedding_model_name,
                                "bag_size": int(bag_size),
                                "dataset": dataset,
                                "label": target_column_name,
                                "seed": random_seed,
                                "prefix": prefix,
                                })
                            save_json(results_file, metrics)
                            csv_df.to_csv(os.path.join(root, "model_outputs.csv"), index=False)
                        # except:
                        #     print(root)
                    if len(metrics) != 0:
                        rl_config = load_json(os.path.join(root, "sweep_best_model_config.json"))
                        for k in df.keys():
                            df[k].append(metrics.get(k, rl_config.get(k, None)))
    df = pd.DataFrame(df)
    df.rename(columns={f"test/avg-{metric_name}": f"test/{metric_name}", f"eval/avg-{metric_name}": f"eval/{metric_name}"}, inplace=True)
    return df

def agg_mil_results(device, data_dir_path, models_dir_path, bag_embedded_column_name,  task_type, config_vars=[], in_seeds=[0, 1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9]):

    pattern = fr"{models_dir_path}/{task_type}/seed_(\d+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/bag_size_(\d+)/([^/]+)$"
    regex = re.compile(pattern)
    metric_name = 'f1' if task_type == 'classification' else 'r2'
    df = {
            "model": [],
            "embedding_model": [],
            "bag_size": [],
            "dataset": [],
            "label": [],
            "seed": [],
            "test/loss": [],
            "eval/loss": [],
            "test/auc": [],
            "test/f1": [],
            "eval/auc": [],
            "eval/f1": [],
            "eval/f1_micro": [],
            "test/f1_micro": []
    }
    
    for k in config_vars:
        df[k] = []
    
    for root, _, files in os.walk(models_dir_path):
        for file in files:
            if file == "log.txt":
                # print(root)
                match = regex.search(root)
                # print(match)
                if match:
                    # print(match.groups())
                    (random_seed, 
                    dataset,
                    data_embedded_column_name,
                    embedding_model_name,
                    target_column_name,
                    bag_size,
                    model_name) = match.groups()
                    if int(random_seed) not in in_seeds:
                        continue
                    metrics = {}
                    results_file = os.path.join(root, "results.json")
                    if os.path.isfile(results_file):
                        metrics = load_json(results_file)
                        metrics["seed"] = random_seed
                        
                    if (f"eval/{metric_name}" not in metrics.keys()) or (os.path.isfile(os.path.join(root, "model_outputs.csv")) == False):
                        print(f"MIL json result does not exist! {root}")
                        metrics, csv_df = generate_result_for_mil_model(root, device, random_seed, data_dir_path, dataset, 
                                                                data_embedded_column_name, embedding_model_name, target_column_name, 
                                                                bag_size, bag_embedded_column_name, task_type)
                        if len(metrics) != 0:
                            metrics.update({
                                "model": model_name.split("_")[0],
                                "embedding_model": embedding_model_name,
                                "bag_size": int(bag_size),
                                "dataset": dataset,
                                "label": target_column_name,
                                "seed": random_seed,
                            })
                            save_json(results_file, metrics)
                            csv_df.to_csv(os.path.join(root, "model_outputs.csv"), index=False)
                    if len(metrics) != 0:
                        mil_config = load_json(os.path.join(root, "best_model_config.json"))
                        for k in df.keys():
                            df[k].append(metrics.get(k, mil_config.get(k, None)))
    return pd.DataFrame(df)


def generate_result_for_rlmil_model(model_path, device, random_seed, data_dir_path, dataset, 
                                  data_embedded_column_name, embedding_model_name, target_column_name, 
                                  bag_size, bag_embedded_column_name, task_type):
    # try:
        metrics = {}
        df = {'split': [], 'label': [], 'pred': [], 'pred_prob': []}
        subset=False
        metric_name = 'f1' if task_type == 'classification' else 'r2'

        dataloaders = get_dataloaders(data_dir_path, random_seed, dataset, data_embedded_column_name, embedding_model_name, 
                                                target_column_name, bag_size, bag_embedded_column_name, subset)

        policy_network = load_rlmil_model(model_path, device)
        policy_network.eval()
        # pool_size = 1 if policy_network.sample_algorithm == 'static' else 10
        random = True if 'only_ensemble' in model_path else False
        for split, dataloader in dataloaders.items():
            data = policy_network.select_from_dataloader(dataloader, int(bag_size), random=random)
            metric, probs, labels, preds = policy_network.compute_metrics_and_details(data)
            # dataloader_pool = policy_network.create_pool_data(dataloader=dataloader, bag_size=int(bag_size), pool_size=pool_size, random=random)
            # avg_reward, loss, ensemble_reward = policy_network.expected_reward_loss(dataloader_pool)
            for metric_key, metric_value in metric.items():
                metrics.update({
                    f'{split}/{metric_key}': metric_value,
                })
            df['split'].extend([split] * len(labels))
            df['label'].extend(labels)
            df['pred'].extend(preds)
            df['pred_prob'].extend(probs)
            # metrics.update({
            #     f"{split}/loss": loss,
            #     f"{split}/avg-{metric_name}": avg_reward,
            #     f"{split}/ensemble-{metric_name}": ensemble_reward,
            #     f"{split}/{metric_name}": None,
            # })
        return metrics, pd.DataFrame(df)
    # except:
    #     return {}, []

def generate_result_for_mil_model(model_path, device, random_seed, data_dir_path, dataset, 
                                  data_embedded_column_name, embedding_model_name, target_column_name, 
                                  bag_size, bag_embedded_column_name, task_type):
    best_model = load_mil_model(model_path)
    best_model.eval()
    best_model.to(device)
    metric_name = 'f1' if task_type == 'classification' else 'r2'
    subset=True
    data_loaders = get_dataloaders(data_dir_path, random_seed, dataset, data_embedded_column_name, embedding_model_name, 
                                          target_column_name, bag_size, bag_embedded_column_name, subset)
    metrics = {}
    df = {'split': [], 'label': [], 'pred': [], 'pred_prob': []}
    for split, dataloader in data_loaders.items():
        if task_type == 'classification':
            test_loss = get_crossentropy(best_model, dataloader, device)
            metric, labels, labels_preds, labels_probs = get_classification_metrics(best_model, dataloader, device, detailed=True)
            metric['loss'] = test_loss
        else:
            test_loss = get_mse(best_model, dataloader, device) 
            metric, labels, labels_preds, labels_probs = get_r2_score(best_model, dataloader, device) # TODO: add detailed=True
            metric = {'r2' :metric, 'loss': test_loss}
            
        for metric_key, metric_value in metric.items():
            metrics.update({
                f'{split}/{metric_key}': metric_value,
            })
        df['split'].extend([split] * len(labels))
        df['label'].extend(labels)
        df['pred'].extend(labels_preds)
        df['pred_prob'].extend(labels_probs)
    df = pd.DataFrame(df)
    return metrics, df

def get_dataloaders(data_dir_path, random_seed, dataset, data_embedded_column_name, embedding_model_name, 
                        target_column_name, bag_size, bag_embedded_column_name, subset, instance_labels_column=None):
    train_dataframe = read_data_split(
        data_dir=os.path.join(
            data_dir_path,
            f'seed_{random_seed}',
            dataset,
            data_embedded_column_name,
        ),
        embedding_model=embedding_model_name,
        split="train",
    )
    
    train_dataframe_mean, train_dataframe_median, train_dataframe_std = get_df_mean_median_std(
        train_dataframe, target_column_name
    )
    extra_columns = [instance_labels_column] if instance_labels_column is not None else []
        
    data_loaders = {}
    for split in ["train", "val", "test"]:
        split_dataframe = read_data_split(
            data_dir=os.path.join(
                data_dir_path,
                f'seed_{random_seed}',
                dataset,
                data_embedded_column_name,
            ),
            embedding_model=embedding_model_name,
            split=split,
        )
        

        split_dataframe, _, _ = preprocess_dataframe(df=split_dataframe, dataframe_set=split,
                                                    label=target_column_name,
                                                    train_dataframe_mean=train_dataframe_mean,
                                                    train_dataframe_median=train_dataframe_median,
                                                    train_dataframe_std=train_dataframe_std,
                                                    task_type="classification", 
                                                    extra_columns=extra_columns)

        split_bag_masks = create_bag_masks(
            split_dataframe,
            int(bag_size),
            bag_embedded_column_name,
        )
        
        split_dataset = RLMILDataset(
            df=split_dataframe,
            bag_masks=split_bag_masks,
            subset=subset,
            task_type="classification",
            instance_labels_column=instance_labels_column,
        )
        
        split_dataloader = DataLoader(
            split_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )
        split_save_name = "eval" if split == "val" else split
        data_loaders.update({split_save_name: split_dataloader})
        
    return data_loaders

def add_formatted_column(df, metric, extra_groupby_cols=[]):
    mil_df = df[df['prefix'] == '---']
    ensemble_df = df[df['prefix'] == 'ensemble']
    rlmil_df = df[df['model'].str.startswith('RL-')]
    
    groupby_cols = ['model', 'dataset', 'label'] + extra_groupby_cols
    
    mil_df = mil_df.groupby(groupby_cols)[f'test/{metric}'].agg([np.mean, np.std]).reset_index()
    mil_df.columns = groupby_cols + [f'avg_{metric}', f'std_{metric}']
    
    ensemble_df = ensemble_df.groupby(groupby_cols)[f'test/avg-{metric}'].agg([np.mean, np.std]).reset_index()
    ensemble_df.columns = groupby_cols + [f'avg_{metric}', f'std_{metric}']
    
    rlmil_df = rlmil_df.groupby(groupby_cols).agg({f'test/avg-{metric}': [np.mean, np.std],
                                                   f'test/ensemble-{metric}': [np.mean, np.std]}).reset_index()
    rlmil_df.columns = groupby_cols + [f'avg_{metric}', f'std_{metric}', 
                                       f'avg_ensemble_{metric}', f'std_ensemble_{metric}']
    
    mil_df[f'formatted_{metric}'] = mil_df.apply(lambda x: format_column(x, metric), axis=1)
    ensemble_df[f'formatted_{metric}'] = ensemble_df.apply(lambda x: format_column(x, metric), axis=1)
    rlmil_df[f'formatted_{metric}'] = rlmil_df.apply(lambda x: format_column(x, metric), axis=1)

    df = pd.concat([mil_df, ensemble_df, rlmil_df])
    return df

def format_column(row, metric):
    if "rl" not in row['model']:
        if pd.isna(row[f'std_{metric}']):
            return f"${row[f'avg_{metric}']:.3f}$"
        return f"${row[f'avg_{metric}']:.3f}_{{(\\pm {row[f'std_{metric}']:.3f})}}$"
    if "static" in row['prefix'] or row['prefix'] == 'ensemble':
        if pd.isna(row[f'std_{metric}']):
            return f"${row[f'avg_{metric}']:.3f}$"
        return f"${row[f'avg_{metric}']:.3f}_{{(\\pm {row[f'std_{metric}']:.3f})}}$"
    if pd.isna(row[f'std_{metric}']):
        return f"${row[f'avg_{metric}']:.3f}$ / ${row[f'avg_ensemble_{metric}']:.3f}$"
    return f"${row[f'avg_{metric}']:.3f}_{{(\\pm {row[f'std_{metric}']:.3f})}}$ / ${row[f'avg_ensemble_{metric}']:.3f}_{{(\\pm {row[f'std_ensemble_{metric}']:.3f})}}$"

    
def save_dataset_table(df, dataset, metric, prefix, extra_index=[]):
    temp_df = df[df['dataset'] == dataset].reset_index(drop=True)
    column_format = 'll' + 'c' * len(temp_df['label'].unique())
    temp_df = temp_df.pivot_table(index=['model'] + extra_index, 
                        columns='label', 
                        values=f'formatted_{metric}', 
                        aggfunc='first')
    temp_df = temp_df.sort_values(by=['prefix'])
    
    temp_df.reset_index().to_latex(f'./tables_latex/{prefix}_{dataset}.tex', 
                                                                column_format=column_format, 
                                                                index=False)

def save_dataset_table_for_all(df, metric, prefix, extra_index=[]):
    temp_df = df
    column_format = 'll' + 'c' * len(temp_df['label'].unique())
    temp_df = temp_df.pivot_table(index=['model'] + extra_index, 
                        columns='label', 
                        values=f'formatted_{metric}', 
                        aggfunc='first')
    temp_df = temp_df.sort_values(by=['prefix'])
    
    temp_df.reset_index().to_latex(f'./tables_latex/{prefix}.tex', 
                                    column_format=column_format, 
                                    index=False)