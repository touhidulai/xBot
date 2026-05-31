# Experiment History — Auction Bot Detector

This document tracks every experiment we ran during Stage 4 model training,
including what we tried, why, and what the results were.

---

## Dataset Summary

| Property | Value |
|---|---|
| Total bidders | 1,984 labelled |
| Bot bidders | 103 (5.1%) |
| Human bidders | 1,881 (94.9%) |
| Bid rows (labelled) | 3,071,224 |
| Bot bids | 412,416 (13.4%) |
| Human bids | 2,658,808 (86.6%) |
| Test set bots | 21 real bots |
| Features per bid | 8 |
| Sequence length | 500 bids (truncated/padded) |

---

## Features Used

| Feature | Description | Bot Signal |
|---|---|---|
| time | Raw bid timestamp | Less useful than diff |
| time_diff | Gap between consecutive bids | Bots bid in 0.3s gaps |
| device | Device fingerprint (encoded) | Bots use same device always |
| country | Bidder country (encoded) | Minor signal |
| auction | Auction ID (encoded) | Bots target same auction |
| merchandise | Item category (encoded) | Minor signal |
| same_auction_flag | 1 if same auction as previous bid | Bots repeat same auction |
| same_device_flag | 1 if same device as previous bid | Bots never switch device |

**Removed feature:** `time_of_day` — distribution was nearly uniform across all hours (global platform, different timezones), added noise not signal.

---

## Model Architecture Decisions

### Why Transformer?
- Bid sequences vary from 1 to 515,033 bids per bidder
- LSTM forgets over very long sequences
- Transformer reads all bids simultaneously — no forgetting problem
- 3,071,224 labelled bid rows — large enough for Transformer
- Bot patterns require connecting distant bids in sequence

### Why Autoencoder?
- Only 103 labelled bot sequences — very few for supervised learning
- Autoencoder only needs human sequences (1,881 available)
- Learns what "normal" bidding looks like
- Bots reconstruct poorly = high error = detected as anomaly

---

## Experiment Log

---

### Experiment 1 — Baseline Transformer
**Changes:** Basic setup, 5 features, BCEWithLogitsLoss, lr=0.001, 20 epochs, no oversampling

**Problem:** Loss = nan

**Root Cause:**
- Data not scaled (time column = trillions, device = 0-3)
- time dominated all other features
- Gradients exploded → nan

**Result:**
```
Loss: nan
Accuracy: 94.83% (predicting all humans)
Bot F1: N/A
```

---

### Experiment 2 — Fixed Scaling + Correct Loss
**Changes:** Added StandardScaler, removed double sigmoid, switched to BCELoss, lr=0.0001

**Problem:** Still predicting all humans

**Root Cause:**
- Class imbalance: 94.9% humans
- Model found it easier to predict everyone as human
- Gets 94.83% accuracy without detecting any bots

**Result:**
```
Loss: 0.1467
Accuracy: 94.83%
Bot Recall: 0%
```

---

### Experiment 3 — Weighted Loss
**Changes:** Added weighted loss (18.35x penalty for missing a bot)

**Why:** Force model to pay more attention to rare bot examples

**Result:**
```
Loss: 0.7626
Accuracy: 78.83%
Bot Precision: 0.14
Bot Recall:    0.67
Bot F1:        0.23
```
**Improvement:** Model now detecting some bots but very low precision (86% false alarms)

---

### Experiment 4 — Simple Oversampling
**Changes:** Oversampled bots from 103 → 500 using resample() (duplication)

**Why:** More bot examples = model sees bots more often during training

**Note:** This had a data leakage issue — oversampling was done BEFORE train/test split,
meaning synthetic bots appeared in both train and test sets.

**Result (optimistic due to leakage):**
```
Loss: 0.6333
Bot Precision: 0.48
Bot Recall:    0.90
Bot F1:        0.63
```
**Improvement:** Large jump in precision and F1

---

### Experiment 5 — New Features + Focal Loss
**Changes:**
- Added time_diff (gap between bids)
- Added same_auction_flag
- Added same_device_flag
- Switched to Focal Loss (alpha=0.75, gamma=2)
- input_dim: 5 → 8

**Why Focal Loss:** Reduces loss for easy examples (obvious humans), focuses learning on hard examples (rare bots)

**Result:**
```
Loss: 0.0500
Accuracy: 88.87%
Bot Precision: 0.56
Bot Recall:    0.85
Bot F1:        0.67
```

---

### Experiment 6 — Threshold Tuning
**Changes:** Same model as Exp 5, tested different classification thresholds

**Why:** Default threshold 0.5 may not be optimal for imbalanced data

**Result:**
```
Threshold  Precision  Recall   F1
0.3        0.534      0.860    0.659
0.4        0.559      0.850    0.675  ← best recall balance
0.5        0.691      0.670    0.680  ← best F1
0.6        0.917      0.440    0.595
0.7        0.905      0.190    0.314
```

---

### Experiment 7 — time_of_day Feature + LR Scheduler + 50 Epochs
**Changes:**
- Added time_of_day feature (hour 0-23)
- Learning rate scheduler (StepLR, step=10, gamma=0.5)
- Increased epochs: 20 → 50
- input_dim: 8 → 9

**Note:** Results appeared very good but this still had the data leakage issue from Exp 4.

**Result (still optimistic due to leakage):**
```
Loss: 0.0232
Accuracy: 96.69%
Bot F1 at threshold 0.4: 0.856
Bot Recall: 0.890
Bot Precision: 0.824
```

---

### Experiment 8 — Fixed Data Leakage (SMOTE + Correct Split Order)
**Changes:**
- Fixed split order: train_test_split FIRST, then oversample
- Switched from simple resample to SMOTE (generates synthetic sequences)
- Smaller model: d_model=64→32, num_layers=2→1, dropout=0.1→0.3
- Added weight_decay=0.01

**Why SMOTE:** Simple oversampling duplicates same 82 bots repeatedly → model memorises them.
SMOTE generates synthetic bot sequences → more diverse training data.

**Result (honest, no leakage):**
```
Train Loss: 0.0779  Test Loss: 0.0772  Gap: 0.0019
Status: No overfitting ✅

Bot Precision: 0.15
Bot Recall:    0.76
Bot F1:        0.25
```
**Key finding:** F1 dropped from 0.856 to 0.25 — previous good results were due to data leakage.
Real performance is F1 = 0.25.

---

### Experiment 9 — Added Overfitting Check
**Changes:** Added test loss tracking inside training loop, loss plot

**Finding:** Train loss and test loss stay close together → no overfitting.
Problem is NOT overfitting — it is fundamental data scarcity (only 103 real bots).

---

### Experiment 10 — Removed time_of_day (Noisy Feature)
**Changes:** Removed time_of_day from features, input_dim: 9 → 8

**Why:** Investigated time_of_day distribution — nearly uniform across all 24 hours.
Global platform with different timezones = no day/night pattern.
Feature adds noise not signal.

**Result:**
```
Bot F1: 0.27 (tiny improvement from removing noise)
```
**Conclusion:** Removing noisy feature helped slightly but root cause is data scarcity.

---

### Experiment 11 — Autoencoder Anomaly Detection
**Approach:** Train only on human sequences, detect bots by high reconstruction error

**Why:** Supervised learning struggles with 103 bot examples.
Autoencoder only needs human data (1,881 sequences available).

**Architecture:**
- Encoder: 8 features → 8
- Bottleneck: 500×8 → 32 → 500×8
- Decoder: 8 → 8 features
- Loss: Masked MSE (ignores padding zeros)

**Result:**
```
Human reconstruction error: Mean=0.4219
Bot reconstruction error:   Mean=1.0566
Bot mean is 2.5x higher ✅ — separation exists

Threshold Analysis:
0.7        Precision=0.151  Recall=0.762  F1=0.252
1.1        Precision=0.196  Recall=0.476  F1=0.278
```
**Conclusion:** Similar performance to Transformer. Both limited by data scarcity.

---

## Overall Results Summary

| Experiment | Bot F1 | Bot Recall | Bot Precision | Notes |
|---|---|---|---|---|
| Exp 1 | N/A | N/A | N/A | nan loss |
| Exp 2 | N/A | 0% | N/A | All humans predicted |
| Exp 3 | 0.23 | 0.67 | 0.14 | Weighted loss |
| Exp 4 | 0.63 | 0.90 | 0.48 | Data leakage! |
| Exp 5 | 0.67 | 0.85 | 0.56 | New features + Focal Loss |
| Exp 6 | 0.675 | 0.85 | 0.559 | Threshold tuned |
| Exp 7 | 0.856 | 0.890 | 0.824 | Data leakage! |
| Exp 8 | 0.25 | 0.76 | 0.15 | **Honest result, fixed leakage** |
| Exp 9 | 0.25 | 0.76 | 0.15 | No overfitting confirmed |
| Exp 10 | 0.27 | 0.76 | 0.16 | Removed noisy feature |
| Exp 11 (AE) | 0.25 | 0.76 | 0.15 | Autoencoder approach |

---

## Root Cause Analysis

The fundamental bottleneck is **data scarcity**:

```
Training set: 82 real bot sequences
Test set:     21 real bot sequences

No matter what technique we apply:
→ Oversampling = model memorises same 82 bots
→ SMOTE = synthetic bots may not match real patterns
→ Both approaches hit the same ceiling: F1 ~0.25-0.27
```

**What would actually improve results:**
1. More labelled bot data (hundreds of bot examples minimum)
2. Semi-supervised learning using 7.6M unlabelled bids
3. Rule-based layer on top of ML (hard rules for obvious bots)
4. Active learning — label more data based on model uncertainty

---

## Lessons Learned

1. **Data leakage is subtle** — oversampling BEFORE split inflates results significantly
2. **Scaling order matters** — fit scaler on train, transform test separately
3. **Accuracy is misleading** — 94.9% accuracy by predicting all humans
4. **Data beats model complexity** — more bot labels > better architecture
5. **Overfitting ≠ poor results** — our model generalises well but lacks signal
6. **Real-world ML is messy** — textbook results rarely appear in practice
