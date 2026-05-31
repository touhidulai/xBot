import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.load_data import load_data
from src.processing.process_data import join_data, handle_missing

def engineer_features(df):
    print("\nEngineering features...")
    
    bidder_features = df.groupby("bidder_id").agg(
        total_bids        = ("bid_id", "count"),
        unique_auctions   = ("auction", "nunique"),
        unique_devices    = ("device", "nunique"),
        unique_countries  = ("country", "nunique"),
        mean_time_diff    = ("time", lambda x: x.sort_values().diff().mean())
    ).reset_index()
    
    return bidder_features

def attach_labels(bidder_features, train_df):
    print("\nAttaching bot labels...")
    final_df = bidder_features.merge(train_df[["bidder_id", "outcome"]],
                                      on="bidder_id",
                                      how="left")
    print(f"Final shape: {final_df.shape}")
    return final_df

def save_data(df):
    print("\nSaving processed data...")
    output_path = os.path.join("data", "processed", "features.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    train_df, bids_df = load_data()
    merged_df = join_data(train_df, bids_df)
    cleaned_df = handle_missing(merged_df)
    bidder_features = engineer_features(cleaned_df)
    final_df = attach_labels(bidder_features, train_df)
    save_data(final_df)
    print("\nFeature engineering complete.")
    print(final_df.head())