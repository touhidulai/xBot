import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.load_data import load_data
from src.processing.sequence_builder import (
    filter_labelled_bids,
    handle_missing,
    encode_columns,
    sort_by_time,
    add_time_differences,
    add_behaviour_flags
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class BotDetectorTransformer(nn.Module):
    def __init__(self, input_dim=8, d_model=32,
                 nhead=4, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding    = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.transformer  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dropout=dropout, batch_first=True
            ),
            num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze()

def build_test_sequences(test_df, bids_df, max_len=500):
    print("\nBuilding test sequences...")
    
    features = ["time", "time_diff", "device", "country",
                "auction", "merchandise",
                "same_auction_flag", "same_device_flag"]
    
    bidders = test_df["bidder_id"].values
    X = np.zeros((len(bidders), max_len, len(features)))
    
    for i, bidder in enumerate(bidders):
        bidder_bids = bids_df[bids_df["bidder_id"] == bidder]
        if len(bidder_bids) == 0:
            continue
        bids = bidder_bids[features].values
        length = min(len(bids), max_len)
        X[i, :length, :] = bids[:length]
    
    print(f"Test sequences shape: {X.shape}")
    return X, bidders

def load_model(model_path):
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = BotDetectorTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    threshold = checkpoint['threshold']
    print(f"Model loaded. Threshold: {threshold}")
    return model, threshold


def predict(model, X, scaler, batch_size=32):
    print("\nGenerating predictions...")
    
    # Scale using same scaler as training
    n_samples, n_steps, n_features = X.shape
    X_scaled = scaler.transform(
        X.reshape(-1, n_features)
    ).reshape(n_samples, n_steps, n_features)
    
    X_tensor = torch.FloatTensor(X_scaled)
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            probs = model(batch)
            all_probs.extend(probs.numpy())
    
    return np.array(all_probs)

def save_predictions(bidders, probs, threshold, output_path):
    print("\nSaving predictions...")
    
    predictions = pd.DataFrame({
        'bidder_id':  bidders,
        'prediction': probs
    })
    
    predictions.to_csv(output_path, index=False)
    
    print(f"Saved {len(predictions)} predictions to {output_path}")
    print(f"\nPrediction summary:")
    print(f"→ Bot (>{threshold}):   {(probs > threshold).sum()}")
    print(f"→ Human (<={threshold}): {(probs <= threshold).sum()}")
    print(f"→ Mean probability:    {probs.mean():.4f}")
    print(f"→ Max probability:     {probs.max():.4f}")
    print(f"→ Min probability:     {probs.min():.4f}")
    
    return predictions

if __name__ == "__main__":
    
    # Paths
    MODEL_PATH  = os.path.join("models", "transformer_model.pth")
    OUTPUT_PATH = os.path.join("data", "processed", "predictions.csv")
    
    # Step 1 — Load raw data
    train_df, bids_df = load_data()
    test_df = pd.read_csv(os.path.join("data", "raw", "test.csv"))
    print(f"Test bidders: {len(test_df)}")
    
    # Step 2 — Process bids (same pipeline as training)
    bids_df = handle_missing(bids_df)
    bids_df = encode_columns(bids_df)
    bids_df = sort_by_time(bids_df)
    bids_df = add_time_differences(bids_df)
    bids_df = add_behaviour_flags(bids_df)
    
    # Step 3 — Build sequences for test bidders
    X_test, bidders = build_test_sequences(test_df, bids_df)
    
    # Step 4 — Fit scaler on train sequences
    # Must use same scaling as training
    print("\nFitting scaler on training data...")
    X_train = np.load(os.path.join("data", "processed", "X_sequences.npy"))
    from sklearn.preprocessing import StandardScaler
    n_samples, n_steps, n_features = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    print("Scaler fitted.")
    
    # Step 5 — Load model and predict
    model, threshold = load_model(MODEL_PATH)
    probs = predict(model, X_test, scaler)
    
    # Step 6 — Save predictions
    predictions = save_predictions(
        bidders, probs, threshold, OUTPUT_PATH
    )
    
    print("\nPrediction complete.")
    print(predictions.head(10))