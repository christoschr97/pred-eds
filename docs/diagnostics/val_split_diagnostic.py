
"""
Diagnostic ONLY — not part of the publication code, not saved into src/.

Question: how much does the reported Full-Model test AUPRC change if early
stopping / checkpoint selection / LR scheduling are driven by a validation
split carved out of the TRAINING period, instead of the held-out test set
that the published number is computed on?

Both variants share: same data file, same feature set (16), same
architecture (AttentionLSTM, hidden=64, layers=2, dropout=0.2), same
num_epochs=50/batch=64/lr=0.001/patience=10, same seeds [0,1,2] (matching
run_ablations.py's actual seed loop, range(num_seeds)).

Variant "original" replicates src/models.py::train_lstm exactly (monitors
test loss/AUPRC every epoch) -- this is a sanity-check replicate of the
published Full Model number (0.8088 mean test AUPRC per results/ablations_5c.json).

Variant "corrected" carves the last 15% of the TRAIN period (by month) off
as a validation split. Scaler is fit on the reduced train split only.
Early stopping, best-checkpoint selection, and the LR scheduler all key off
validation loss. X_test/y_test are touched exactly once, after training is
complete and the best checkpoint (by validation loss) is loaded.
"""
import sys, os, json, time, copy
sys.path.insert(0, "/Users/christoschristodoulou/research_projects/disact-pred-eds/src")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from data_prep import (
    DATA_FILE, load_acled_data, create_monthly_aggregation_with_networks,
    create_temporal_sequences, temporal_split
)
from models import AttentionLSTM

DATA_PATH = "/Users/christoschristodoulou/research_projects/disact-pred-eds/data/" + DATA_FILE
SEEDS = [0, 1, 2]
NUM_EPOCHS = 50
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 10
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
DEVICE = "cpu"

t0 = time.time()
df = load_acled_data(DATA_PATH)
monthly_df = create_monthly_aggregation_with_networks(df)
X, y, region_ids, regions, month_targets = create_temporal_sequences(monthly_df, sequence_length=6)
print(f"[prep done in {time.time()-t0:.1f}s] X={X.shape} pos_rate={y.mean():.4f}")

# ---- Standard 70/30 temporal split (identical to the published pipeline) ----
X_train_raw, y_train_full, X_test_raw, y_test = temporal_split(X, y, region_ids, month_targets, split_ratio=0.7)

unique_months_sorted = np.sort(np.unique(month_targets))
cutoff_idx = int(len(unique_months_sorted) * 0.7)
cutoff_month = unique_months_sorted[cutoff_idx]
train_months_sorted = unique_months_sorted[unique_months_sorted < cutoff_month]

# ---- Three-way split for the corrected variant: carve last 15% of TRAIN months as validation ----
val_cutoff_idx = int(len(train_months_sorted) * 0.85)
val_cutoff_month = train_months_sorted[val_cutoff_idx]

train_mask_full = month_targets < cutoff_month
train2_mask = month_targets < val_cutoff_month
val_mask = (month_targets >= val_cutoff_month) & (month_targets < cutoff_month)

X_train2_raw = X[train2_mask]; y_train2 = y[train2_mask]
X_val_raw = X[val_mask]; y_val = y[val_mask]

print(f"\nSplit sizes: train(orig)={len(X_train_raw)}  train2(corrected)={len(X_train2_raw)}  "
      f"val={len(X_val_raw)}  test={len(X_test_raw)}")
print(f"Positive rate: train2={y_train2.mean():.4f}  val={y_val.mean():.4f}  test={y_test.mean():.4f}")

def fit_scaler(X_fit, *X_apply):
    scaler = StandardScaler()
    flat = X_fit.reshape(-1, X_fit.shape[2])
    scaler.fit(flat)
    out = []
    for Xa in (X_fit,) + X_apply:
        out.append(scaler.transform(Xa.reshape(-1, Xa.shape[2])).reshape(Xa.shape))
    return out, scaler

(X_train_orig, X_test_orig), _ = fit_scaler(X_train_raw, X_test_raw)
(X_train2, X_val, X_test_corr), _ = fit_scaler(X_train2_raw, X_val_raw, X_test_raw)

def make_model(input_dim):
    return AttentionLSTM(input_dim=input_dim, hidden_dim=HIDDEN, num_layers=NUM_LAYERS, dropout=DROPOUT)

def train_variant(X_train, y_train, X_monitor, y_monitor, X_final_test, y_final_test, seed, monitor_name):
    """Generic trainer: early-stops/checkpoints on (X_monitor, y_monitor);
    evaluates once at the end on (X_final_test, y_final_test)."""
    np.random.seed(seed); torch.manual_seed(seed)
    pos_weight = (1 - y_train.mean()) / y_train.mean()
    model = make_model(X_train.shape[2]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    Xtr = torch.FloatTensor(X_train).to(DEVICE); ytr = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
    Xmo = torch.FloatTensor(X_monitor).to(DEVICE); ymo = torch.FloatTensor(y_monitor).unsqueeze(1).to(DEVICE)

    best_monitor_loss = float('inf'); best_state = None; patience_counter = 0
    stopped_epoch = NUM_EPOCHS - 1
    monitor_auprc_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            b = idx[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            out = model(Xtr[b])
            loss = criterion(out, ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mo_out = model(Xmo)
            mo_loss = criterion(mo_out, ymo).item()
            mo_auprc = average_precision_score(ymo.cpu().numpy(), torch.sigmoid(mo_out).cpu().numpy())
        monitor_auprc_history.append(mo_auprc)

        scheduler.step(mo_loss)
        if mo_loss < best_monitor_loss:
            best_monitor_loss = mo_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            stopped_epoch = epoch
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xte = torch.FloatTensor(X_final_test).to(DEVICE)
        test_probs = torch.sigmoid(model(Xte)).cpu().numpy()
        Xtrain_eval = torch.FloatTensor(X_train).to(DEVICE)
        train_probs = torch.sigmoid(model(Xtrain_eval)).cpu().numpy()

    return {
        'monitor': monitor_name,
        'seed': seed,
        'stopped_epoch': stopped_epoch,
        'best_monitor_auprc': max(monitor_auprc_history),
        'test_auprc': average_precision_score(y_final_test, test_probs),
        'test_auroc': roc_auc_score(y_final_test, test_probs),
        'test_brier': brier_score_loss(y_final_test, test_probs),
        'train_auprc': average_precision_score(y_train, train_probs),
    }

results = {'original_test_monitored': [], 'corrected_val_monitored': []}
for seed in SEEDS:
    t1 = time.time()
    r_orig = train_variant(X_train_orig, y_train_full, X_test_orig, y_test, X_test_orig, y_test,
                            seed, 'test_loss (as published)')
    results['original_test_monitored'].append(r_orig)
    print(f"[seed {seed}] ORIGINAL  (monitor=test):  stopped@{r_orig['stopped_epoch']:2d}  "
          f"test_AUPRC={r_orig['test_auprc']:.4f}  test_AUROC={r_orig['test_auroc']:.4f}  "
          f"({time.time()-t1:.1f}s)")

    t1 = time.time()
    r_corr = train_variant(X_train2, y_train2, X_val, y_val, X_test_corr, y_test,
                            seed, 'val_loss (corrected)')
    results['corrected_val_monitored'].append(r_corr)
    print(f"[seed {seed}] CORRECTED (monitor=val):   stopped@{r_corr['stopped_epoch']:2d}  "
          f"test_AUPRC={r_corr['test_auprc']:.4f}  test_AUROC={r_corr['test_auroc']:.4f}  "
          f"({time.time()-t1:.1f}s)")

with open("/tmp/val_split_diagnostic.json", "w") as f:
    json.dump({
        'note': 'diagnostic only, not published, not saved into src/',
        'seeds': SEEDS,
        'split_sizes': {'train_orig': int(len(X_train_raw)), 'train2_corrected': int(len(X_train2_raw)),
                         'val': int(len(X_val_raw)), 'test': int(len(X_test_raw))},
        'results': results,
    }, f, indent=1)
print("\nSaved to /tmp/val_split_diagnostic.json")
print(f"TOTAL TIME: {time.time()-t0:.1f}s")
