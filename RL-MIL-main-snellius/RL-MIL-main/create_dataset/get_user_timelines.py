import os
import threading
import time

import pandas as pd
import tweepy
from tqdm import tqdm
from tweepy.errors import NotFound, TooManyRequests


def log(batch_number: int, message: str) -> None:
    """
    Logs a message to ./logs/{batch_number}_logs.txt
    """
    # if log of batch_number exists, append to it
    # else create a new log file
    if not os.path.exists(f"./logs/{batch_number}_logs.txt"):
        with open(
                f"./logs/{batch_number}_logs.txt", "w", encoding="utf-8", buffering=1
        ) as log_file:
            log_file.write(f"{message}\n")
        return
    else:
        with open(
                f"./logs/{batch_number}_logs.txt", "a", encoding="utf-8", buffering=1
        ) as log_file:
            log_file.write(f"{message}\n")
        return


def get_user_timeline(api: tweepy.API, batch_number: int) -> None:
    log(
        batch_number=batch_number,
        message=f"Starting thread {batch_number} at {time.ctime()}",
    )
    twitter_handle_batch_dataframe = pd.read_csv(
        f"./twitter_handles/twitter_handles_batch_{batch_number}.csv"
    )

    for twitter_handle in twitter_handle_batch_dataframe["twitter_handle"]:
        # If user_id.csv already exists continue
        if os.path.exists(f"./user_timelines/{twitter_handle}.csv"):
            continue
        try:
            timeline = {}
            timeline["tweets"] = []
            counter = 0
            for tweets in tqdm(
                    tweepy.Cursor(
                        api.user_timeline,
                        screen_name=twitter_handle,
                        count=200,
                        include_rts=True,
                        exclude_replies=False,
                        tweet_mode="extended",
                    ).items(),
                    desc=f"Getting timeline of {twitter_handle}",
            ):
                timeline["tweets"].append(tweets._json)
                if counter % 10 == 0:
                    pd.DataFrame(timeline).to_csv(
                        f"./user_timelines/{twitter_handle}.csv", index=False
                    )
                counter += 1
            pd.DataFrame(timeline).to_csv(
                f"./user_timelines/{twitter_handle}.csv", index=False
            )
        except NotFound:
            log(
                batch_number=batch_number,
                message=f"User {twitter_handle} not found",
            )
            continue
        except TooManyRequests:
            if len(timeline["tweets"]) > 0:
                pd.DataFrame(timeline).to_csv(
                    f"./user_timelines/{twitter_handle}.csv", index=False
                )
            log(
                batch_number=batch_number,
                message=f"Too many requests, sleeping thread number {batch_number} for 15 minutes",
            )
            time.sleep(15 * 60)
            log(
                batch_number=batch_number,
                message=f"Thread number {batch_number} woke up",
            )
        except Exception as other_exception:
            log(
                batch_number=batch_number,
                message=f"Exception {other_exception} occurred for user {twitter_handle}",
            )
            if len(timeline["tweets"]) > 0:
                pd.DataFrame(timeline).to_csv(
                    f"./user_timelines/{twitter_handle}.csv", index=False
                )
            continue


if __name__ == "__main__":
    # There are a number of twitter_handles_batch_{batch_number}.csv files in the
    # "twitter_handles" folder. There are also a number of Twitter developer accounts
    # in the "TwitterDeveloperAccounts.csv" file. We will use the developer accounts
    # to fetch the data of the twitter handles in the batches.
    # Read the "TwitterDeveloperAccounts.csv" file
    twitter_developer_accounts_dataframe = pd.read_csv("TwitterDeveloperAccounts.csv")

    # Create an api list to store the api instances
    api_list = []
    for index, row in twitter_developer_accounts_dataframe.iterrows():
        # Get the account credentials
        consumer_key = row["consumer_key"]
        consumer_secret = row["consumer_secret"]
        access_token = row["access_token"]
        access_token_secret = row["access_token_secret"]

        # Create an OAuthHandler instance
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        # Create API instance with automatic rate limit handling
        api = tweepy.API(auth, wait_on_rate_limit=True)

        # Add the api instance to the api list
        api_list.append(api)

    threads = []
    for i, api in enumerate(api_list):
        thread = threading.Thread(target=get_user_timeline, args=(api, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("Done!")
