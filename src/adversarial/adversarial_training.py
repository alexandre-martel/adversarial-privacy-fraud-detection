# adversarial_training.py
import argparse
import numpy as np
import joblib
import os 

from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.utils import (
    set_seed, load_dataset, get_datasets, standardize, 
    compute_scale_pos_weight, summarize, 
    plot_training_history, plot_evaluation_results, plot_epsilon_study, 
    recall_pos)
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba
from src.adversarial.fsgm_attack import fgsm_attack_batch

# Training epoch with mixed clean + adversarial samples, with a given ratio of adversarial in the batch.
# The base ratio is 1/2, but we can change it to have more or less adversarial samples in the batch.
def train_epoch_mixed(model, loader, optimizer, loss_fn, epsilon, low, high, device, mix_ratio=0.5):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        model.eval()  # Set model to eval mode for attack generation
        # generate adversarial examples for the whole batch
        xb_adv = fgsm_attack_batch(model, loss_fn, xb, yb, epsilon, low, high)

        model.train()  # Back to train mode for the optimization step
        # Mix clean and adversarial examples
        n = xb.size(0)
        n_adv = int(mix_ratio * n)
        idx = torch.randperm(n, device=device)

        xb_mix = torch.cat([xb[idx[:n - n_adv]], xb_adv[idx[:n_adv]]], dim=0)
        yb_mix = torch.cat([yb[idx[:n - n_adv]], yb[idx[:n_adv]]], dim=0)

        optimizer.zero_grad()
        logits = model(xb_mix)
        loss = loss_fn(logits, yb_mix)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb_mix.size(0)

    return total_loss / len(loader.dataset)



def main():
    model_folder = "results/adversarial_model"
    os.makedirs(model_folder, exist_ok=True)

    parser = argparse.ArgumentParser(description="Adversarial Training (MLP against FGSM) - Credit Card Fraud")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1, help="Perturbation strength for FGSM attack")
    parser.add_argument("--mix-ratio", type=float, default=0.5, help="Ratio of adversarial samples in the mixed training batches")
    parser.add_argument("--with_epsilon_study", default=False, action="store_true", help="If set to False, only run the attack for the specified epsilon, without doing a study over multiple epsilons.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, feat_names = load_dataset(args.data_path)
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)
    X_tr_s, X_val_s, X_te_s, scaler, q_low, q_high = standardize(X_tr, X_val, X_te)

    Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    Xval = torch.tensor(X_val_s, dtype=torch.float32)
    yval = torch.tensor(y_val, dtype=torch.float32)
    Xte = torch.tensor(X_te_s, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xval, yval), batch_size=4096)
    test_loader  = DataLoader(TensorDataset(Xte, yte),  batch_size=4096)

    model = MLP(in_dim=X_tr_s.shape[1]).to(device)
    pos_weight = torch.tensor([compute_scale_pos_weight(y_tr)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    history = {"train_loss": [], "val_prauc": []}

    # Bornes clamp for FSM attack
    low = torch.tensor(q_low, dtype=torch.float32, device=device).unsqueeze(0)
    high = torch.tensor(q_high, dtype=torch.float32, device=device).unsqueeze(0)

    # Adversarial Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch_mixed(model, train_loader, optimizer, loss_fn,
                                    epsilon=args.epsilon, low=low, high=high,
                                    device=device, mix_ratio=args.mix_ratio)
        yv, pv = predict_proba(model, val_loader, device)
        history["train_loss"].append(tr_loss)
        history["val_prauc"].append(average_precision_score(yv, pv))
        print(f"Epoch {epoch:02d} | train loss: {tr_loss:.5f} | val PR-AUC: {average_precision_score(yv, pv):.4f}")
    
    plot_training_history(history, save_path=model_folder)
    
    yv, pv = predict_proba(model, val_loader, device)
    # summarize(yv, pv, title="Adversarial MLP - clean Validation")
    yt, pt = predict_proba(model, test_loader, device)
    summarize(yt, pt, title="Adversarial MLP - clean Test")
    plot_evaluation_results(yt, pt, save_path=model_folder)
    os.rename(os.path.join(model_folder, "pr_curve.png"), os.path.join(model_folder, "clean_pr_curve.png"))
    os.rename(os.path.join(model_folder, "confusion_matrix.png"), os.path.join(model_folder, "clean_confusion_matrix.png"))

    # Adversarial evaluation on test
    adv_probs = []
    adv_targets = []
    model.eval()
    for xb, yb in test_loader:
        Xb_adv = fgsm_attack_batch(model, loss_fn, xb, yb, args.epsilon, low, high)
        with torch.no_grad():
            p = torch.sigmoid(model(Xb_adv.to(device))).cpu().numpy()
        adv_probs.append(p)
        adv_targets.append(yb.numpy())
    adv_probs = np.concatenate(adv_probs)
    adv_targets = np.concatenate(adv_targets)
    summarize(adv_targets, adv_probs, title=f"Adversarial MLP - attack Test (FGSM, eps={args.epsilon})")
    plot_evaluation_results(adv_targets, adv_probs, save_path=model_folder)
    os.rename(f"{model_folder}/pr_curve.png", f"{model_folder}/adv_pr_curve.png")
    os.rename(f"{model_folder}/confusion_matrix.png", f"{model_folder}/adv_confusion_matrix.png")
    
    # save model and preprocessing objects 
    torch.save(model.state_dict(), f"{model_folder}/adversarial_mlp.pt")
    joblib.dump(scaler, f"{model_folder}/scaler.joblib")
    np.savez(f"{model_folder}/q_bounds.npz", low=q_low, high=q_high)
    print("\n Saved: adversarial_mlp.pt, scaler.joblib, q_bounds.npz")
    
    if args.with_epsilon_study:
        print("\n--- Starting Epsilon Study on Robust Model ---")
        eps_list = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1]
        study_recalls = []
        study_aucs = []

        model.eval()
        for eps in eps_list:
            adv_probs = []
            adv_targets = []
            
            for xb, yb in test_loader:
                xb_adv = fgsm_attack_batch(model, loss_fn, xb, yb, eps, low, high)
                with torch.no_grad():
                    p = torch.sigmoid(model(xb_adv.to(device))).cpu().numpy()
                adv_probs.append(p)
                adv_targets.append(yb.numpy())

            adv_probs = np.concatenate(adv_probs)
            adv_targets = np.concatenate(adv_targets)
            
            cur_recall = recall_pos(adv_targets, (adv_probs >= 0.5).astype(int))
            cur_auc = average_precision_score(adv_targets, adv_probs)
            
            study_recalls.append(cur_recall)
            study_aucs.append(cur_auc)
            print(f"Eps: {eps:.2f} | Recall: {cur_recall:.4f} | PR-AUC: {cur_auc:.4f}")
        
        plot_epsilon_study(eps_list, study_recalls, study_aucs, save_path=f"{model_folder}")


if __name__ == "__main__":
    main()