import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize

from logger import get_logger
from utils import set_seed, save_pickle

# import nltk
# nltk.download('punkt')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cls_pooling(model_output) -> torch.Tensor:
    """
    Receives the model output in shape (batch_size, sequence_length, hidden_size)
    and returns the CLS embedding for each sample in the batch in shape (batch_size, hidden_size).
    With default config of RoBERTa, hidden_size = 768 and sequence_length = 64.
    So the input shape is (batch_size, 64, 768) and output shape is (batch_size, 768).
    """
    return model_output.last_hidden_state[:, 0]


def eos_pooling(model_output, input_ids, eos_token_id):
    hidden_states = model_output.last_hidden_state
    eos_mask = input_ids.eq(eos_token_id)

    if len(torch.unique(eos_mask.sum(1))) > 1:
        raise ValueError("All examples must have the same number of <eos> tokens.")
    return hidden_states[eos_mask, :].view(
        hidden_states.size(0), -1, hidden_states.size(-1)
    )[:, -1, :]


def get_embeddings(tokenizer, model, device: torch.device, embedding_model_name: str, bag_of_text: list[str]) -> np.ndarray:
    """
    Receives a list of strings (bags of text) and returns the CLS embeddings for each bag.
    Input shape is equal to the bag size (at most BAG_SIZE) and output shape is (BAG_SIZE, hidden_size).
    CLS pooling is also possible:
        # cls_embedding = cls_pooling(model_output).cpu().detach().numpy()
        # embeddings.append(cls_embedding)
    """

    batch_size = 100

    num_batch = len(bag_of_text) // batch_size
    if len(bag_of_text) % batch_size != 0:
        num_batch += 1
    embeddings = []
    for i in range(num_batch):
        if i == num_batch:
            batch = bag_of_text[i * batch_size:]
        else:
            batch = bag_of_text[i * batch_size: (i + 1) * batch_size]
        token_ids = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        model_output = model(**token_ids)

        if embedding_model_name == 'sentence-transformers/paraphrase-xlm-r-multilingual-v1':
            mean_embedding = (
                mean_pooling(model_output, token_ids["attention_mask"])
                .cpu()
                .detach()
                .numpy()
            )
            embeddings.append(mean_embedding)
        else:
            eos_embedding = (
                eos_pooling(model_output, token_ids["input_ids"], tokenizer.eos_token_id)
                .cpu()
                .detach()
                .numpy()
            )
            embeddings.append(eos_embedding)


    return np.concatenate(embeddings)


def create_bag(row, tokenizer, model, device: torch.device, embedding_model_name: str, text_column: str, bag_size: int,
               bag_heuristic: str):
    bag = row[text_column]
    if bag_heuristic == "longest":
        # Current heuristic is to sort the tweets by length and take the top BAG_SIZE
        bag = sorted(bag, key=len, reverse=True)
    bag = bag[:bag_size]
    random.shuffle(bag)
    embeddings = get_embeddings(tokenizer, model, device, embedding_model_name, bag)

    # Pad all the embeddings to the same size as BAG_SIZE
    bag_embeddings = np.zeros((bag_size, embeddings.shape[1]))
    bag_mask = np.zeros(bag_size)

    bag_embeddings[: embeddings.shape[0]] = embeddings
    bag_mask[: embeddings.shape[0]] = 1

    row["bag"] = bag
    row["bag_embeddings"] = bag_embeddings
    row["bag_mask"] = bag_mask
    return row


def customized_political_opinion_yourmorals_incas(political_opinion: str):
    """
    Here is the distribution of the labels in political_opinion column:
    Liberal                         404
    Moderate, middle of the road    265
    Conservative                    106
    Something else                   83
    Slightly liberal                  2
    Very liberal                      1

    I want to merge "Slighly liberal" and "Very liberal" with "Liberal" and drop "Something else"
    """
    if pd.isna(political_opinion):
        return np.nan
    else:
        if "liberal" in political_opinion.lower():
            return "liberal"
        elif political_opinion == "Something else":
            return np.nan
        else:
            return political_opinion.lower()

def convert_age_to_bins_incas(df: pd.DataFrame):
    bins = [16, 20, 30, 40, 50, 60, float('inf')]
    bin_names = ["16-20", "21-30", "31-40", "41-50", "51-60", "61 and above"]
    df = df.rename(columns={'age': 'old_age'})
    df['age'] = pd.cut(df['old_age'], bins=bins, labels=bin_names, right=False)
    return df

def convert_education_to_bins_incas(df: pd.DataFrame):
    education_mapping = {
        'Master': 'Master/Doctorate',
        'Doctorate': 'Master/Doctorate',
        'Some College': 'Bachelor/Some College',
        'Bachelor': 'Bachelor/Some College',
        'High School': 'High School/Less than high school',
        'Less than high school': 'High School/Less than high school'
        }
    df = df.rename(columns={'education': 'old_education'})
    df['education'] = df['old_education'].replace(education_mapping)
    return df

def convert_political_orientation_to_bins_incas(df: pd.DataFrame):
    def make_customized_political_orientation(x):
        if pd.isna(x):
            return np.nan
        else:
            if x in [1, 2, 3]:
                return "Left_123"
            elif x == 4:
                return "Neutral_4"
            else:
                return "Right_567"
    df = df.rename(columns={'political_orientation': 'old_political_orientation'})
    df["political_orientation"] = df["old_political_orientation"].apply(make_customized_political_orientation)
    return df

def convert_ladder_to_bins_incas(df: pd.DataFrame):
    df = df.rename(columns={'ladder': 'old_ladder'})
    df['ladder'] = pd.qcut(df['old_ladder'], 3, labels=['lad_bin_1', 'lad_bin_2', 'lad_bin_3'])
    return df

def convert_religion_to_bins_incas(df: pd.DataFrame):
    df = df.rename(columns={"religion": "old_religion"})
    df['religion'] = df['old_religion'].apply(lambda x: 'Nonreligious' if x == 'Nonreligious' else 'Religious')
    return df

def create_dataset(dataset_name: str, embedding_model: str, random_seed: int, device, text_column: str,
                   whole_bag_size: int, num_pos_samples: int = 2):
    tqdm.pandas()
    # The random seed for dataset creation
    random.seed(random_seed)

    if embedding_model == 'twhin-bert-base':
        embedding_model = "Twitter/twhin-bert-base"
    elif embedding_model == 'paraphrase-xlm-r-multilingual-v1':
        embedding_model = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

    tokenizer = AutoTokenizer.from_pretrained(embedding_model, model_max_length=512)
    model = AutoModel.from_pretrained(embedding_model).to(device)

    # args.dataset is either 'political', 'facebook', or 'incas_yourmorals' or 'incas'
    if dataset_name == 'political_data_with_age':
        bag_heuristic = ""

        df = pd.read_csv(os.path.join(os.getcwd(), "data", "political_data.csv"))
        df = df[df["congress"] >= 108].reset_index(drop=True)
        # Let's eval the text column to make them a list of strings
        df[text_column] = df[text_column].apply(eval)

        df = df.rename(columns={"age": "old_age"})
        # Let's create a new column called age
        bins = [27, 40, 55, 70, float('inf')]
        labels = ["Young Adult", "Middle-aged", "Senior", "Elderly"]
        df['age'] = pd.cut(df['old_age'], bins=bins, labels=labels, right=True)

    elif dataset_name == 'facebook':
        bag_heuristic = ""

        df = pd.read_json(os.path.join(os.getcwd(), "data", "facebook", "processed_data", "full_dataset_clean.jsonl"),
                          lines=True)

        df = df.rename(columns={"Care": "care",
                                "Fairness": "fairness",
                                "Loyalty": "loyalty",
                                "Authority": "authority",
                                "Purity": "purity"
                                })

    elif dataset_name == 'yourmorals_incas':
        bag_heuristic = "longest"

        df = pd.read_csv(os.path.join(os.getcwd(), "data", "yourmorals_incas_data.csv"))
        for col in ['timeline_tweets', 'timeline_retweets', 'timeline_replies', 'timeline_quotes', 'timeline_merged_chronologically', 'timeline_cleaned_tweets']:
            df[col] = df[col].apply(eval)

    elif dataset_name == 'incas':
        bag_heuristic = "longest"

        df = pd.read_csv(os.path.join(os.getcwd(), "data", "incas_data.csv"))
        for col in ['timeline_tweets', 'timeline_retweets', 'timeline_replies', 'timeline_quotes', 'timeline_merged_chronologically', 'timeline_cleaned_tweets']:
            df[col] = df[col].apply(eval)

        df = convert_age_to_bins_incas(df)
        df = convert_education_to_bins_incas(df)
        df = convert_ladder_to_bins_incas(df)
        df = convert_political_orientation_to_bins_incas(df)
        df = convert_religion_to_bins_incas(df)

    elif dataset_name.startswith('jigsaw'):
        bag_heuristic = ""

        df = pd.read_csv('./data/jigsaw/train.csv')
        df['hate'] = (df['severe_toxic'] + df['toxic'] + df['obscene'] + df['identity_hate'] + df['insult'] + df['threat']) > 0

        # df['hate'] = df['hate'].astype(int)
        df_hate = df[df['hate'] == 1].reset_index(drop=True)
        df_non_hate = df[df['hate'] == 0].reset_index(drop=True)
        df_hate = df_hate.sample(frac = 1, random_state=random_seed).reset_index(drop=True)
        df_non_hate = df_non_hate.sample(frac = 1, random_state=random_seed).reset_index(drop=True)

        df = synthetic_dataset(df_pos=df_hate, df_neg=df_non_hate,
                               whole_bag_size=whole_bag_size,
                               num_pos_samples=num_pos_samples,
                               random_seed=random_seed, text_column=text_column, labels_column='hate')
    elif dataset_name == 'essays':
        bag_heuristic = ""

        df = pd.read_csv(os.path.join(os.getcwd(), "data", "essays.csv"))

        # lowercase the column names
        df.columns = df.columns.str.lower()
        # rename the columns
        df = df.rename(columns={
            '#authid': 'author_id',
            'cext': 'extraversion',
            'cneu': 'neuroticism',
            'cagr': 'agreeableness',
            'ccon': 'conscientiousness',
            'copn': 'openness',
        })

        df['author_id'] = df['author_id'].str.strip()
        df['text'] = df['text'].str.strip()

        # change all of the 'y' values to 1 and all of the 'n' values to 0 and change the data type to int
        df['extraversion'] = df['extraversion'].map({'y': 1, 'n': 0}).astype(int)
        df['neuroticism'] = df['neuroticism'].map({'y': 1, 'n': 0}).astype(int)
        df['agreeableness'] = df['agreeableness'].map({'y': 1, 'n': 0}).astype(int)
        df['conscientiousness'] = df['conscientiousness'].map({'y': 1, 'n': 0}).astype(int)
        df['openness'] = df['openness'].map({'y': 1, 'n': 0}).astype(int)

        df['sentences'] = df['text'].apply(sent_tokenize)

        # Shuffle the sentences for each author using a random seed of 42
        df['sentences'] = df['sentences'].apply(lambda sentences: pd.DataFrame(sentences).sample(frac=1, random_state=random_seed)[0].tolist())
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    df[f"num_{text_column}"] = df[text_column].str.len()
    df = df[df[f"num_{text_column}"] > 19].reset_index(drop=True)

    df = df.progress_apply(create_bag, args=(tokenizer, model, device, embedding_model, text_column, whole_bag_size, bag_heuristic), axis=1)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_seed, shuffle=True)
    test_df, val_df = train_test_split(
        test_df, test_size=0.5, random_state=random_seed, shuffle=True
    )

    dataset_path = os.path.join(os.getcwd(), "data", f"seed_{random_seed}", dataset_name, text_column, embedding_model)
    save_pickle(os.path.join(dataset_path, 'train.pickle'), train_df)
    save_pickle(os.path.join(dataset_path, 'val.pickle'), val_df)
    save_pickle(os.path.join(dataset_path, 'test.pickle'), test_df)

def synthetic_dataset(df_pos, df_neg, whole_bag_size, num_pos_samples, random_seed, text_column, labels_column):
    num_neg_samples = whole_bag_size - num_pos_samples
    df = {
    text_column: [],
    f"{labels_column}s": [],
    labels_column: []
    }
    total_samples = min(df_pos.shape[0]//num_pos_samples, df_neg.shape[0]//(num_neg_samples+whole_bag_size)) // 2
    for i in range(total_samples):
        # pos sample
        pos_samples = df_pos.iloc[num_pos_samples*i:num_pos_samples*(i+1)]
        neg_samples = df_neg.iloc[num_neg_samples*i:num_neg_samples*(i+1)]
        bag = pd.concat([pos_samples, neg_samples]).reset_index(drop=True)
        bag = bag.sample(frac = 1).reset_index(drop=True)
        df[text_column].append(bag[text_column].tolist())
        df[f"{labels_column}s"].append(bag[labels_column].tolist())
        df[labels_column].append(int(sum(bag[labels_column].tolist()) > 0))
        # neg sample
        bag = df_neg.iloc[whole_bag_size*(i+total_samples):whole_bag_size*(i+total_samples+1)]
        bag = bag.sample(frac = 1).reset_index(drop=True)
        df[text_column].append(bag[text_column].tolist())
        df[f"{labels_column}s"].append(bag[labels_column].tolist())
        df[labels_column].append(int(sum(bag[labels_column].tolist()) > 0))
    df = pd.DataFrame(df)
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument(
        "--data_embedded_column_name",
        type=str,
        default="timeline_cleaned_tweets",
        required=False,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--whole_bag_size",
        type=int,
        default=100,
        required=True,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        required=False,
    )
    parser.add_argument(
        "--num_pos_samples",
        type=int,
        default=2,
        required=False,
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    if args.dataset == 'jigsaw':
        args.dataset  = f"{args.dataset}_{args.num_pos_samples}"
    dataset_path = os.path.join(os.getcwd(), "data", f"seed_{args.random_seed}", args.dataset,
                                args.data_embedded_column_name,
                                args.embedding_model)
    os.makedirs(dataset_path, exist_ok=True)
    logger = get_logger(dataset_path)
    logger.info(f"{args=}")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    set_seed(args.random_seed)

    # If dataset os not available in ./data, then create it
    train_pickle_path = os.path.join(dataset_path, "train.pickle")
    val_pickle_path = os.path.join(dataset_path, "val.pickle")
    test_pickle_path = os.path.join(dataset_path, "test.pickle")
    if not os.path.exists(dataset_path) or not os.path.isfile(train_pickle_path) or not os.path.isfile(
            val_pickle_path) or not os.path.isfile(test_pickle_path):
        logger.info(f'{os.path.exists(dataset_path)=}')
        logger.info(f'{os.path.isfile(train_pickle_path)=}')
        logger.info(f'{os.path.isfile(val_pickle_path)=}')
        logger.info(f'{os.path.isfile(test_pickle_path)=}')
        logger.info(f'Creating dataset with random seed {args.random_seed}')
        create_dataset(dataset_name=args.dataset, embedding_model=args.embedding_model,
                       random_seed=args.random_seed, text_column=args.data_embedded_column_name,
                       device=device, whole_bag_size=args.whole_bag_size, num_pos_samples=args.num_pos_samples)
        logger.info(f'Dataset was created with random seed {args.random_seed}')
    else:
        logger.info(f'Dataset with random seed {args.random_seed} already exists')
