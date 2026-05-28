import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.load_data import load_data

def check_missing(df, name):
    print(f"\n--- MISSING VALUES: {name} ---")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    print(pd.concat([missing, percent], axis=1, 
                    keys=["Missing Count", "Percent"]))
    
def check_dtypes(df, name):
    print(f"\n--- DATA TYPES: {name} ---")
    print(df.dtypes)

def check_duplicates(df, name):
    print(f"\n--- DUPLICATES: {name} ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

def check_stats(df, name):
    print(f"\n--- BASIC STATS: {name} ---")
    print(df.describe())


def validate_data(train_df, bids_df):
    check_missing(train_df, "train")
    check_missing(bids_df, "bids")
    
    check_dtypes(train_df, "train")
    check_dtypes(bids_df, "bids")
    
    check_duplicates(train_df, "train")
    check_duplicates(bids_df, "bids")
    
    check_stats(train_df, "train")
    check_stats(bids_df, "bids")

if __name__ == "__main__":
    train_df, bids_df = load_data()
    validate_data(train_df, bids_df)
