import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.load_data import load_data

def join_data(train_df, bids_df):
    print("\nJoining train and bids data...")
    merged_df = bids_df.merge(train_df[["bidder_id", "outcome"]], 
                               on="bidder_id", 
                               how="left")
    print(f"Merged shape: {merged_df.shape}")
    return merged_df

def handle_missing(df):
    print("\nHandling missing values...")
    df["country"] = df["country"].fillna("unknown")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df

if __name__ == "__main__":
    train_df, bids_df = load_data()
    merged_df = join_data(train_df, bids_df)
    cleaned_df = handle_missing(merged_df)
    print(cleaned_df.head())