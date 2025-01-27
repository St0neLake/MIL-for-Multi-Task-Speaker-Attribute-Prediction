import argparse
import os

import pandas as pd
from tqdm import tqdm
import os

tqdm.pandas()

parser = argparse.ArgumentParser()
parser.add_argument("--hein_daily_raw_data_dir", help="Location of downloaded data (hein-daily)")
parser.add_argument("--hein_daily_preprocessed_data_dir", help="Location of output data files")
parser.add_argument("--congress_terms_raw_data_address", help="Location of downloaded data (congress-terms.csv)")
parser.add_argument("--final_data_address", help="Location of the final output data with age")
parser.add_argument("--years", help="Congressional years", required=True, type=str, nargs='+')

ARGS = parser.parse_args()


def preprocess_hein_daily(**kwargs) -> None:
    """
    Preprocesses speeches by merging them with metadata, filtering out speeches by independent speakers,
    and saving the preprocessed data as a CSV file.

    Args:
        **kwargs: Keyword arguments containing the following parameters:
            - hein_daily_preprocessed_data_dir (str): Directory path to save the preprocessed data.
            - hein_daily_raw_data_dir (str): Directory path where the raw data is located.
            - years (list): List of years to preprocess.

    Returns:
        None
    """
    
    if not os.path.isdir(kwargs["hein_daily_preprocessed_data_dir"]):
        os.makedirs(kwargs["hein_daily_preprocessed_data_dir"])

    for year in kwargs["years"]:
        print(f"Preprocessing {year}...")
        speeches_path = os.path.join(kwargs["hein_daily_raw_data_dir"], f"speeches_{year}.txt")
        meta_path = os.path.join(kwargs["hein_daily_raw_data_dir"], f"{year}_SpeakerMap.txt")
        speeches = pd.read_csv(
            speeches_path,
            sep="|",
            engine="python",
            encoding="Latin1",
            quoting=3,
            on_bad_lines="skip",
        )
        meta = pd.read_csv(meta_path, sep="|")

        df = speeches.merge(meta, how="inner", on="speech_id")

        df = df[df.party.isin({"R", "D"})]  # No Independents!!!

        # We need csvs!
        df.groupby(["speakerid", "state", "gender", "party", "chamber", "firstname", "lastname"])["speech"].apply(
            list
        ).reset_index().rename(columns={"speech": "text"}).to_csv(
            os.path.join(kwargs["hein_daily_preprocessed_data_dir"], f"all_{year}.csv"), index=False
        )


def load_congress_terms(**kwargs) -> pd.DataFrame:
    """
    Load and preprocess the congress terms data.

    Args:
        **kwargs: Additional keyword arguments.
            congress_terms_raw_data_address (str): The file path of the raw congress terms data.
            hein_daily_preprocessed_data_dir (str): The directory path of the preprocessed data.

    Returns:
        pd.DataFrame: The preprocessed congress terms data.
    """
    congress_terms_df = pd.read_csv(kwargs["congress_terms_raw_data_address"])
    # In chamber change "house" to "H" and change "senate" to "S"
    congress_terms_df['chamber'] = congress_terms_df['chamber'].replace(['house', 'senate'], ['H', 'S'])
    assert set(congress_terms_df['chamber']) == {'H', 'S'}

    years = []
    for csv_file in [file for file in os.listdir(kwargs["hein_daily_preprocessed_data_dir"]) if file.endswith('.csv')]:
        # Let's skip if it was 114th congress since we don't have the data for it
        if csv_file.split('.')[0].split('_')[1] == '114':
            continue
        years.append(csv_file.split('.')[0].split('_')[1])

    years.sort()

    # Keep only congress years that we have data for
    congress_terms_df = congress_terms_df[congress_terms_df['congress'].isin([int(year) for year in years])]

    return congress_terms_df


def make_first_letter_uppercase(string: str) -> str:
  return string[0].upper() + string[1:].lower()


def load_hein_daily(**kwargs) -> pd.DataFrame:
    """
    Load the Hein Daily dataset.

    Args:
        **kwargs: Additional keyword arguments.
            hein_daily_preprocessed_data_dir (str): Directory path of the preprocessed data.

    Returns:
        pd.DataFrame: The loaded Hein Daily dataset.
    """

    hein_daily_df = pd.DataFrame()

    # For each file in all_data_dir read the csv file, add congress using the file name (e.g. all_097.csv) and append to hein_daily_df
    for csv_file in [file for file in os.listdir(kwargs["hein_daily_preprocessed_data_dir"]) if file.endswith('.csv')]:
        congress = int(csv_file.split('.')[0].split('_')[1])
        df = pd.read_csv(os.path.join(kwargs["hein_daily_preprocessed_data_dir"], csv_file))
        df['congress'] = congress
        hein_daily_df = pd.concat([hein_daily_df, df], ignore_index=True)

    # apply make_first_letter_uppercase to the "firstname" and "lastname" columns in hein_daily_df
    hein_daily_df['firstname'] = hein_daily_df['firstname'].apply(make_first_letter_uppercase)
    hein_daily_df['lastname'] = hein_daily_df['lastname'].apply(make_first_letter_uppercase)

    return hein_daily_df


def join_dataframes(congress_terms_df: pd.DataFrame, hein_daily_df: pd.DataFrame, final_output_address: str) -> pd.DataFrame:
    """
    Joins two dataframes based on specified columns and saves the result to a CSV file.

    Args:
        congress_terms_df (pd.DataFrame): The dataframe containing congress terms data.
        hein_daily_df (pd.DataFrame): The dataframe containing daily data.
        final_output_address (str): The file path to save the joined dataframe.

    Returns:
        pd.DataFrame: The joined dataframe.

    Raises:
        AssertionError: If the specified merge columns have different data types.

    """
    merge_cols = ['congress', 'firstname', 'lastname', 'chamber', 'state', 'party']
    for col in merge_cols:
        assert hein_daily_df[col].dtype == congress_terms_df[col].dtype, f'{col} has different dtypes'

    joined_df = hein_daily_df.merge(congress_terms_df, on=['congress', 'firstname', 'lastname', 'chamber', 'state', 'party'], how='inner', suffixes=('_hein', '_congress'))
    joined_df = joined_df.reset_index(drop=True)
    joined_df['num_speeches'] = joined_df['text'].progress_apply(lambda x: len(eval(x)))

    # Let's save it to a csv file named political_data.csv
    joined_df.to_csv(final_output_address, index=False)

if __name__ == "__main__":
    # if the hein_daily_preprocessed_data_dir already exists, and it contains all of the files from all_097.csv to all_114.csv
    # then we don't need to preprocess_hein_daily and we are ready to join the dataframes
    files_exist = True
    if os.path.isdir(ARGS.hein_daily_preprocessed_data_dir):
        for file in os.listdir(ARGS.hein_daily_preprocessed_data_dir):
            if file.endswith('.csv'):
                if file not in [f'all_{year}.csv' for year in ARGS.years]:
                    print(f'{file} is not in: ' + ', '.join([f'all_{year}.csv' for year in ARGS.years]))
                    files_exist = False
                    break
    else:
        files_exist = False

    if files_exist:
        print('hein_daily_preprocessed_data_dir already exists and contains all of the files from all_097.csv to all_114.csv')
    else:
        print('Preprocessing Hein Daily...')
        preprocess_hein_daily(**vars(ARGS))

    congress_terms_df = load_congress_terms(**vars(ARGS))
    hein_daily_df = load_hein_daily(**vars(ARGS))

    # Join the dataframes
    joined_df = join_dataframes(congress_terms_df, hein_daily_df, ARGS.final_data_address)

    print('Done!')
