# fgsm_attack.py
import argparse
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
)

from src.utils import compute_scale_pos_weight, summarize
from src.baselines.mlp_class import MLP

# THe low and high bounds for the FGSM attack are computed as the 0.5% and 99.5% quantiles of each feature 
# on the train set, after standardization. It allows to clip the perturbed samples, to avoid creating unrealistic 
# samples that would be too simple to change the prediction.
def fgsm_attack_batch(model, loss_fn, Xb, yb, epsilon, low, high):

    """Generate FGSM adversarial examples: 
        X_adv = X + eps * sign(dloss/dX), then clip by column.

    Args:
        model: PyTorch model to attack
        loss_fn: loss function used to compute the gradient (should be the same, if possible, as the one used for training)
        Xb: batch of input samples (tensor, shape [batch_size, n_features])
        yb: batch of target labels (tensor, shape [batch_size])
        epsilon: perturbation strength (float)
        low: tensor of shape (1, n_features) with the per-feature lower bounds (quantiles) 
        high: tensor of shape (1, n_features) with the per-feature upper bounds (quantiles)
    """

    Xb = Xb.clone().detach().to("cpu").requires_grad_(True)
    yb = yb.to("cpu")

    model.zero_grad()
    logits = model(Xb)
    loss = loss_fn(logits, yb)
    loss.backward()

    with torch.no_grad():
        X_adv = Xb + epsilon * Xb.grad.sign()
        X_adv = torch.max(torch.min(X_adv, high), low)

    return X_adv.detach()

def main():
    parser = argparse.ArgumentParser(description="FGSM sur MLP - Credit Card Fraud")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Perturbation strength for FGSM attack")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for generating adversarial examples")
    parser.add_argument("--model_folder", type=str, default="baseline_model", help="Folder where the baseline model and preprocessing objects are saved")
    args = parser.parse_args()

    X_test = np.load(f"{args.model_folder}/X_test.npy")
    y_test = np.load(f"{args.model_folder}/y_test.npy")
    scaler = joblib.load(f"{args.model_folder}/scaler.joblib")

    q = np.load(f"{args.model_folder}/q_bounds.npz")
    q_low, q_high = q["low"], q["high"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X_test.shape[1]).to(device)
    state = torch.load(f"{args.model_folder}/mlp_baseline.pt", map_location=device)

    model.load_state_dict(state)

    # pos_weight for the loss used in the attack
    pos_weight = compute_scale_pos_weight(y_test)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch_size)

    # Evaluation before attack
    model.eval()
    with torch.no_grad():
        logits = []
        ys = []
        for xb, yb in test_loader:
            lb = model(xb.to(device))
            logits.append(lb.cpu())
            ys.append(yb)
        logits = torch.cat(logits)
        ys = torch.cat(ys).numpy()
        proba = torch.sigmoid(logits).numpy()
    summarize(ys, proba, title="MLP - Test (avant attaque)")

    # FGSM + evaluation after attack
    low = torch.tensor(q_low, dtype=torch.float32, device=device).unsqueeze(0)
    high = torch.tensor(q_high, dtype=torch.float32, device=device).unsqueeze(0)

    adv_probs = []
    adv_targets = []
    for xb, yb in test_loader:
        Xb_adv = fgsm_attack_batch(model, loss_fn, xb, yb, args.epsilon, low, high)
        with torch.no_grad():
            p = torch.sigmoid(model(Xb_adv)).cpu().numpy()
        adv_probs.append(p)
        adv_targets.append(yb.numpy())

    adv_probs = np.concatenate(adv_probs)
    adv_targets = np.concatenate(adv_targets)
    summarize(adv_targets, adv_probs, title=f"MLP - Test (FGSM, eps={args.epsilon})")

    # Focus recall on fraud class (positive class), which is the most important metric for this use case. 
    # We want to see how much it drops under attack.
    y_pred_clean = (proba >= 0.5).astype(int)
    y_pred_adv = (adv_probs >= 0.5).astype(int)
    def recall_pos(y, yhat):
        tp = ((y==1) & (yhat==1)).sum()
        fn = ((y==1) & (yhat==0)).sum()
        return 0.0 if (tp+fn)==0 else tp/(tp+fn)
    r_before = recall_pos(ys, y_pred_clean)
    r_after  = recall_pos(adv_targets, y_pred_adv)
    print(f"\nVariation of the Recall (fraud class) under FGSM: {r_after - r_before:+.4f} (before={r_before:.4f}, after={r_after:.4f})")


if __name__ == "__main__":
    main()