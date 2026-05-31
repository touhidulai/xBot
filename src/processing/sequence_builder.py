import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.load_data import load_data

def filter_labelled_bids(bids_df, train_df):
    print("\nFiltering to labelled bidders only...")
    labelled_bids = bids_df[bids_df["bidder_id"].isin(train_df["bidder_id"])]
    print(f"Rows before: {len(bids_df)}")
    print(f"Rows after:  {len(labelled_bids)}")
    return labelled_bids

def handle_missing(df):
    print("\nHandling missing values...")
    df = df.copy()
    df["country"] = df["country"].fillna("unknown")
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df

def encode_columns(df):
    print("\nEncoding text columns...")
    encoder = LabelEncoder()
    
    text_columns = ["device", "country", "auction", "merchandise"]
    
    for col in text_columns:
        df[col] = encoder.fit_transform(df[col].astype(str))
        print(f"Encoded {col}: {df[col].nunique()} unique values")
    
    return df

def attach_labels(df, train_df):
    print("\nAttaching labels...")
    df = df.merge(train_df[["bidder_id", "outcome"]],
                  on="bidder_id",
                  how="left")
    print(f"Bot bids:   {len(df[df['outcome']==1.0])}")
    print(f"Human bids: {len(df[df['outcome']==0.0])}")
    return df

def sort_by_time(df):
    print("\nSorting bids by time per bidder...")
    df = df.sort_values(["bidder_id", "time"])
    print("Sorting complete.")
    return df

def add_time_differences(df):
    print("\nCalculating time differences...")
    df = df.copy()
    df["time_diff"] = df.groupby("bidder_id")["time"].diff()
    df["time_diff"] = df["time_diff"].fillna(0)
    print(f"time_diff min: {df['time_diff'].min()}")
    print(f"time_diff max: {df['time_diff'].max()}")
    print(f"time_diff mean: {df['time_diff'].mean():.2f}")
    return df

def add_behaviour_flags(df):
    print("\nAdding behaviour flags...")
    df = df.copy()
    
    df["same_auction_flag"] = (
        df.groupby("bidder_id")["auction"]
        .transform(lambda x: (x == x.shift(1)).astype(int))
    )
    
    df["same_device_flag"] = (
        df.groupby("bidder_id")["device"]
        .transform(lambda x: (x == x.shift(1)).astype(int))
    )
    
    df["same_auction_flag"] = df["same_auction_flag"].fillna(0)
    df["same_device_flag"] = df["same_device_flag"].fillna(0)
    
    print(f"Same auction rate: {df['same_auction_flag'].mean():.3f}")
    print(f"Same device rate:  {df['same_device_flag'].mean():.3f}")
    return df


def build_sequences(df, max_len=500):
    print("\nBuilding sequences...")
    
    features = ["time", "time_diff", "device", "country", 
            "auction", "merchandise",
            "same_auction_flag", "same_device_flag"]
    
    bidders = df["bidder_id"].unique()
    
    X = np.zeros((len(bidders), max_len, len(features)))
    y = np.zeros(len(bidders))
    
    for i, bidder in enumerate(bidders):
        bidder_bids = df[df["bidder_id"] == bidder]
        label = bidder_bids["outcome"].iloc[0]
        bids = bidder_bids[features].values
        length = min(len(bids), max_len)
        X[i, :length, :] = bids[:length]
        y[i] = label
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Bot sequences:   {int(y.sum())}")
    print(f"Human sequences: {int(len(y) - y.sum())}")
    return X, y

def save_sequences(X, y):
    print("\nSaving sequences...")
    output_path = os.path.join("data", "processed")
    np.save(os.path.join(output_path, "X_sequences.npy"), X)
    np.save(os.path.join(output_path, "y_labels.npy"), y)
    print(f"Saved X_sequences.npy: {X.shape}")
    print(f"Saved y_labels.npy:    {y.shape}")


if __name__ == "__main__":
    train_df, bids_df = load_data()
    bids_df = filter_labelled_bids(bids_df, train_df)
    bids_df = handle_missing(bids_df)
    bids_df = encode_columns(bids_df)
    bids_df = attach_labels(bids_df, train_df)
    bids_df = sort_by_time(bids_df)
    bids_df = add_time_differences(bids_df)
    bids_df = add_behaviour_flags(bids_df)
    X, y = build_sequences(bids_df)
    save_sequences(X, y)
    print("\nSequence building complete.")