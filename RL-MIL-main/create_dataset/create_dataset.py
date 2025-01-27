import os
import re
import unicodedata

import pandas as pd
from tqdm import tqdm


def preprocess_text(text: str) -> str:
    """
    Cleans tweet text.
    1. Removes URLs.
    2. Removes mentions.
    3. Removes hashtags.
    4. Removes emojis.
    5. Removes extra spaces and punctuation.
    """

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 2. Remove mentions
    text = re.sub(r"@\w+", "", text)

    # 3. Remove hashtags
    text = re.sub(r"#", "", text)

    # 4. Remove emojis
    text = "".join(c for c in text if unicodedata.category(c) != "So")

    # 5. Remove extra spaces and punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,?!]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # 6. Strip leading and trailing whitespace
    text = text.strip()

    return text


if __name__ == "__main__":
    DATA_PATH = os.path.join("/".join(os.getcwd().split("/")[:-1]), "data")
    question_mappings = {
        "analytics.Care": [1, 2, 3, 4, 5, 6],
        "analytics.Equality": [7, 8, 9, 10, 11, 12],
        "analytics.Proportionality": [13, 14, 15, 16, 17, 18],
        "analytics.Loyalty": [19, 20, 21, 22, 23, 24],
        "analytics.Authority": [25, 26, 27, 28, 29, 30],
        "analytics.Purity": [31, 32, 33, 34, 35, 36],
    }

    twitter_users_answered_MFQ_dataframe = pd.read_csv(
        "UsersLoggedInWithTwitterMFQAnswers.csv", low_memory=False
    )

    # from twitter_users_answered_MFQ_dataframe, we only need the columns having "analytics" in them, the columns having "responses.mfq_{number}" in them and the column "user_info.twitterUsername"
    twitter_users_answered_MFQ_dataframe = twitter_users_answered_MFQ_dataframe[
        [
            column
            for column in twitter_users_answered_MFQ_dataframe.columns
            if "analytics" in column or "responses.mfq_" in column
        ]
        + ["user_info.twitterUsername"]
        ]

    twitter_users_demographics_dataframe = pd.read_csv(
        "UsersLoggedInWithTwitterWithDemographics.csv", low_memory=False
    )

    # From twitter_users_demographics_dataframe, we only need the columns having "user_demographics" in them, and the column "user_info.twitterUsername"
    twitter_users_demographics_dataframe = twitter_users_demographics_dataframe[
        [
            column
            for column in twitter_users_demographics_dataframe.columns
            if "user_demographics" in column
        ]
        + ["user_info.twitterUsername"]
        ]

    # Merge the two dataframes on the column "user_info.twitterUsername"
    merged_dataframe = pd.merge(
        twitter_users_answered_MFQ_dataframe,
        twitter_users_demographics_dataframe,
        on="user_info.twitterUsername",
    )

    # merged_dataframe should have unique values for the column "user_info.twitterUsername"
    merged_dataframe.drop_duplicates(subset="user_info.twitterUsername", inplace=True)

    data = []

    for file in tqdm(os.listdir("user_timelines")):
        if file.endswith(".csv"):
            twitter_handle = file.split(".")[0]
            # Read the csv file. It has only one column named "tweets". If it is empty, then continue
            if len(pd.read_csv(f"user_timelines/{file}")) == 0:
                print(f"User {twitter_handle} has no tweets")
                continue
            moral_values = merged_dataframe[
                merged_dataframe["user_info.twitterUsername"] == twitter_handle
                ][
                [
                    "analytics.Care",
                    "analytics.Equality",
                    "analytics.Proportionality",
                    "analytics.Loyalty",
                    "analytics.Authority",
                    "analytics.Purity",
                ]
            ]

            try:
                # Assert that shape of moral_values is (1, 6)
                assert moral_values.shape == (1, 6)

                # Change dtype of moral_values to double
                moral_values = moral_values.astype("double")
            except AssertionError:
                print(f"Error in {twitter_handle}")
                print(f"Shape of moral_values: {moral_values.shape}")
            except ValueError:
                # Get moral values which have "missing" values
                missing_moral_values = moral_values[
                    moral_values.isin(["missing"]).any(axis=1)
                ]
                # Get the column names of the missing moral values
                missing_moral_values_columns = missing_moral_values.columns[
                    missing_moral_values.isin(["missing"]).any()
                ]
                print(f"Error in {twitter_handle}")
                print(f"Missing moral values: {missing_moral_values_columns}")
                user_mfq_values = merged_dataframe[
                    merged_dataframe["user_info.twitterUsername"] == twitter_handle
                    ]
                for missing_moral_values_column in missing_moral_values_columns:
                    foundation_value = 0
                    counter = 0
                    for question_num in question_mappings[missing_moral_values_column]:
                        print(f"Question number: {question_num}")
                        print(
                            f"Value: {user_mfq_values[f'responses.mfq_{question_num}'].values[0]}"
                        )
                        if pd.isna(
                                user_mfq_values[f"responses.mfq_{question_num}"].values[0]
                        ):
                            continue

                        foundation_value += user_mfq_values[
                            f"responses.mfq_{question_num}"
                        ].values[0]
                        counter += 1
                    if counter == 0:
                        moral_values[missing_moral_values_column] = pd.NA
                    else:
                        foundation_value /= counter
                        moral_values[missing_moral_values_column] = foundation_value

                    print(
                        f"Fixed {missing_moral_values_column} for {twitter_handle} to {foundation_value}"
                    )

                # Change dtype of moral_values to double
                moral_values = moral_values.astype("double")

            # Extract the moral values (if the length of the moral values is 0, then set the value to pd.NA)
            care = (
                moral_values["analytics.Care"].values[0]
                if len(moral_values["analytics.Care"]) > 0
                else pd.NA
            )
            equality = (
                moral_values["analytics.Equality"].values[0]
                if len(moral_values["analytics.Equality"]) > 0
                else pd.NA
            )
            proportionality = (
                moral_values["analytics.Proportionality"].values[0]
                if len(moral_values["analytics.Proportionality"]) > 0
                else pd.NA
            )
            loyalty = (
                moral_values["analytics.Loyalty"].values[0]
                if len(moral_values["analytics.Loyalty"]) > 0
                else pd.NA
            )
            authority = (
                moral_values["analytics.Authority"].values[0]
                if len(moral_values["analytics.Authority"]) > 0
                else pd.NA
            )
            purity = (
                moral_values["analytics.Purity"].values[0]
                if len(moral_values["analytics.Purity"]) > 0
                else pd.NA
            )

            original_tweets = []
            retweets = []
            quotes = []
            replies = []
            merged_chronologically = []
            cleaned_timeline_tweets = []
            timeline_tweets = pd.read_csv(f"user_timelines/{file}").to_dict()
            if len(timeline_tweets) == 0:
                continue
            for timeline_tweet in timeline_tweets["tweets"].values():
                timeline_tweet_dict = eval(timeline_tweet)

                timeline_tweet_full_text = timeline_tweet_dict["full_text"]

                if "retweeted_status" in timeline_tweet_dict:
                    retweets.append(timeline_tweet_full_text)
                elif timeline_tweet_dict["is_quote_status"]:
                    quotes.append(timeline_tweet_full_text)
                elif timeline_tweet_dict["in_reply_to_status_id"] is not None:
                    replies.append(timeline_tweet_full_text)
                else:
                    original_tweets.append(timeline_tweet_full_text)

                merged_chronologically.append(timeline_tweet_full_text)
                cleaned_timeline_tweets.append(
                    preprocess_text(timeline_tweet_full_text)
                )

            row_to_append = {
                "twitter_handle": twitter_handle,
                "care": care,
                "equality": equality,
                "proportionality": proportionality,
                "loyalty": loyalty,
                "authority": authority,
                "purity": purity,
                "timeline_tweets": original_tweets,
                "timeline_retweets": retweets,
                "timeline_replies": replies,
                "timeline_quotes": quotes,
                "timeline_merged_chronologically": merged_chronologically,
                "timeline_cleaned_tweets": cleaned_timeline_tweets,
            }

            for column in merged_dataframe.columns:
                if column in [
                    "user_info.twitterUsername",
                    "analytics.Care",
                    "analytics.Equality",
                    "analytics.Proportionality",
                    "analytics.Loyalty",
                    "analytics.Authority",
                    "analytics.Purity",
                ]:
                    continue
                elif "responses." in column:
                    continue
                elif "user_demographics." in column:
                    # Drop the user_demographics. part of the column name
                    new_column_name = ".".join(column.split(".")[1:])
                    row_to_append[new_column_name] = (
                        merged_dataframe[
                            merged_dataframe["user_info.twitterUsername"]
                            == twitter_handle
                            ][column].values[0]
                        if len(
                            merged_dataframe[
                                merged_dataframe["user_info.twitterUsername"]
                                == twitter_handle
                                ][column]
                        )
                           > 0
                        else pd.NA
                    )

            data.append(row_to_append)

    dataframe = pd.DataFrame(data)
    dataframe.to_csv(f"{DATA_PATH}/yourmorals_data.csv", index=False)
