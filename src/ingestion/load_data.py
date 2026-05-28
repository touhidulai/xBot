import pandas as pd
import os

RAW_DATA_PATH = os.path.join("data", "raw")

TRAIN_FILE = os.path.join(RAW_DATA_PATH, "train.csv")
BIDS_FILE = os.path.join(RAW_DATA_PATH, "bids.csv")

def load_data():
    print("Loading train.csv...")
    train_df = pd.read_csv(TRAIN_FILE)
    
    print("Loading bids.csv...")
    bids_df = pd.read_csv(BIDS_FILE)
    
    return train_df, bids_df

def inspect_data(train_df, bids_df):
    print("\n--- TRAIN DATA ---")
    print(f"Shape: {train_df.shape}")
    print(train_df.head())

    print("\n--- BIDS DATA ---")
    print(f"Shape: {bids_df.shape}")
    print(bids_df.head())

if __name__ == "__main__":
    train_df, bids_df = load_data()
    inspect_data(train_df, bids_df)