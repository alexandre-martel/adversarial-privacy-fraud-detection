import argparse
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import average_precision_score, confusion_matrix, classification_report

from ..utils import set_seed, load_dataset, get_datasets, standardize, compute_scale_pos_weight, summarize
from src.baselines.mlp_class import MLP

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    ys = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits)
        probs.append(prob.cpu().numpy())
        ys.append(yb.cpu().numpy())

    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return ys, probs


def main():
    parser = argparse.ArgumentParser(description="Baseline MLP - Credit Card Fraud")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data loading / preprocessing
    X, y, feat_names = load_dataset(args.data_path)
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)
    X_tr_s, X_val_s, X_te_s, scaler, q_low, q_high = standardize(X_tr, X_val, X_te)

    # to tensors / data loaders
    Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xval = torch.tensor(X_val_s, dtype=torch.float32)
    yval = torch.tensor(y_val, dtype=torch.float32)
    Xte = torch.tensor(X_te_s, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xval, yval), batch_size=4096)
    test_loader  = DataLoader(TensorDataset(Xte, yte),  batch_size=4096)

    # instantiate model, loss, optimizer
    model = MLP(in_dim=X_tr_s.shape[1]).to(device)
    pos_weight = torch.tensor([compute_scale_pos_weight(y_tr)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    for epoch in range(1, args.epochs + 1):
        
        tr_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        yv, pv = predict_proba(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train loss: {tr_loss:.5f} | val PR-AUC: {average_precision_score(yv, pv):.4f}")



    # final evaluation
    yv, pv = predict_proba(model, val_loader, device)
    summarize(yv, pv, title="Baseline MLP - Validation")
    yt, pt = predict_proba(model, test_loader, device)
    summarize(yt, pt, title="Baseline MLP - Test")


    # save model and preprocessing objects for later use in attacks
    torch.save(model.state_dict(), "mlp_baseline.pt")
    joblib.dump(scaler, "scaler.joblib")

    np.savez("q_bounds.npz", low=q_low, high=q_high)
    np.save("X_test.npy", X_te_s)
    np.save("y_test.npy", y_te)

    print("\nSaved: mlp_baseline.pt, scaler.joblib, q_bounds.npz, X_test.npy, y_test.npy")

    
if __name__ == "__main__":
    main()