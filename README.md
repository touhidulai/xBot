# Auction Bot Detector

An end-to-end ML pipeline to detect auction bots
using the Facebook Recruiting IV dataset from Kaggle.

## Problem Statement
Online auction platforms suffer from bot bidders that:
- Drive up prices artificially (shill bidding)
- Destroy buyer trust
- Violate platform terms of service

## Dataset
Download from: https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/data
Place files in: data/raw/

Files needed:
- train.csv (2,013 labelled bidders)
- bids.csv  (7,656,334 bid records)

## Pipeline Stages
- Stage 1: Data Ingestion       → src/ingestion/load_data.py
- Stage 2: Data Validation      → src/validation/validate_data.py
- Stage 3: Sequence Building    → src/processing/sequence_builder.py
- Stage 4: Model Training       → notebooks/auction_bot_detector.ipynb
- Stage 5: Model Serving        → src/serving/ (coming soon)
- Stage 6: Monitoring           → src/monitoring/ (coming soon)

## Key Findings
- 1,984 labelled bidders (103 bots, 1,881 humans)
- Labelled bid rows: 3,071,224
- Best honest model performance: Recall 76%, F1 0.27
- Main bottleneck: only 103 labelled bot sequences

## Features Per Bid (8 total)
1. time
2. time_diff
3. device
4. country
5. auction
6. merchandise
7. same_auction_flag
8. same_device_flag

## Models Trained
- Transformer classifier (supervised)
- Autoencoder anomaly detection (unsupervised)

## Experiment History
See docs/experiments.md for full details of all 11 experiments,
including what was tried, results, and lessons learned.

## Tech Stack
- Python, Pandas, NumPy
- PyTorch (Transformer, Autoencoder)
- Scikit-learn, imbalanced-learn
- Google Colab (GPU training)

## How To Run
1. Download dataset from Kaggle into data/raw/
2. Run Stage 1: python src/ingestion/load_data.py
3. Run Stage 2: python src/validation/validate_data.py
4. Run Stage 3: python src/processing/sequence_builder.py
5. Run Stage 4: Open notebooks/auction_bot_detector.ipynb in Colab
