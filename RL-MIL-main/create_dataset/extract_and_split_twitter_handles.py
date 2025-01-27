import os

import pandas as pd

if __name__ == "__main__":
    # Read "UsersLoggedInWithTwitterWithDemographics.csv" file
    twitter_users_with_demographics_dataframe = pd.read_csv(
        "UsersLoggedInWithTwitterWithDemographics.csv", low_memory=False
    )

    # Read "UsersLoggedInWithTwitterMFQAnswers.csv" file
    twitter_users_with_mfq_answers_dataframe = pd.read_csv(
        "UsersLoggedInWithTwitterMFQAnswers.csv", low_memory=False
    )

    # Read "incasusers.csv" file
    incas_users_dataframe = pd.read_csv("incasusers.csv", low_memory=False)

    # Get all the twitter handles in column "user_info.twitterUsername" in both dataframes,
    # and all the twitter handles in column "username" in incas_users_dataframe
    # and concatenate them into a single dataframe with column name "twitter_handle"
    twitter_handles = pd.concat(
        [
            twitter_users_with_demographics_dataframe["user_info.twitterUsername"],
            twitter_users_with_mfq_answers_dataframe["user_info.twitterUsername"],
            incas_users_dataframe["username"],
        ],
    )

    # Assert twitter_handles is a pandas Series
    assert isinstance(twitter_handles, pd.Series)

    # The series column name is "twitter_handle"
    twitter_handles.name = "twitter_handle"

    # Remove all the duplicate twitter handles
    twitter_handles = twitter_handles.drop_duplicates()
    print("Number of unique twitter handles found: ", len(twitter_handles))

    # Read TwitterDeveloperAccounts.csv file and extract the number of accounts
    twitter_developer_accounts_dataframe = pd.read_csv("TwitterDeveloperAccounts.csv")
    number_of_accounts = len(twitter_developer_accounts_dataframe)

    # Print the number of accounts
    print(f"Number of Twitter Developer accounts found: {number_of_accounts}")

    # Create a folder named "twitter_handles" to store the twitter handles
    # in batches if it does not exist, else remove the folder and create it
    # again
    if os.path.exists("./twitter_handles"):
        print("Removing the existing folder ./twitter_handles")
        os.system("rm -rf ./twitter_handles")
    os.system("mkdir ./twitter_handles")

    # Split the twitter handles in number_of_accounts batches and save each batch to
    # a separate csv file named "./twitter_handles/twitter_handles_batch_{batch_number}.csv"
    batch_size = (
        len(twitter_handles) // number_of_accounts
        if len(twitter_handles) % number_of_accounts == 0
        else len(twitter_handles) // number_of_accounts + 1
    )
    for batch_number in range(number_of_accounts):
        batch = twitter_handles[
                batch_number * batch_size: (batch_number + 1) * batch_size
                ]
        batch.to_csv(
            f"./twitter_handles/twitter_handles_batch_{batch_number}.csv",
            index=False,
        )
