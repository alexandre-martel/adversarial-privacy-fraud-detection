import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import average_precision_score
import joblib

# Custom project utilities
from src.utils import (
    set_seed, load_dataset, get_datasets, standardize, 
    summarize, plot_evaluation_results
)
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def train_one_epoch_dp_adv(model, loader, optimizer, loss_fn, device, eps_adv):
    """
    Performs one epoch of DP-Adversarial training.
    Combines FGSM attack generation with DP-SGD weight updates.
    """
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # Generate Adversarial Attack (FGSM) 
        xb.requires_grad = True
        outputs = model(xb)
        # We use a standard loss for the attack generation
        temp_loss = nn.BCEWithLogitsLoss()(outputs, yb.view_as(outputs))
        
        # Get gradient of the input
        model.zero_grad()
        temp_loss.backward()
        data_grad = xb.grad.data
        
        # Create the adversarial example
        perturbed_data = xb + eps_adv * data_grad.sign()
        perturbed_data = perturbed_data.detach() # Detach to stop tracking for the attack
        
        # DP-SGD Update on Adversarial Data 
        # Now we update the weights using the perturbed data.
        optimizer.zero_grad()
        outputs_adv = model(perturbed_data)
        
        # We use the weighted loss (pos_weight) here for better fraud detection
        loss = loss_fn(outputs_adv, yb.view_as(outputs_adv))
        
        # Opacus captures this backward pass to apply clipping and noise
        loss.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser(description="DP-Adversarial Training (Triple Tension)")
    parser.add_argument("--epsilon-dp", type=float, default=8.0, help="Target Privacy budget")
    parser.add_argument("--epsilon-adv", type=float, default=0.1, help="Adversarial attack strength")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    model_dir = "results/dp_adversarial_model"
    os.makedirs(model_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading and Preprocessing
    X, y, _ = load_dataset(args.data_path)
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)
    X_tr_s, X_val_s, X_te_s, scaler, _, _ = standardize(X_tr, X_val, X_te)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), 
                                            torch.tensor(y_tr, dtype=torch.float32)), 
                              batch_size=args.batch_size, shuffle=True)
    
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32)), batch_size=1024)

    model = MLP(in_dim=X_tr_s.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Weighted loss for imbalanced fraud data
    pos_weight = torch.tensor([500.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Apply Opacus Privacy Engine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.epsilon_dp,
        target_delta=1e-5,
        epochs=args.epochs,
        max_grad_norm=1.0, 
    )

    print(f"Starting DP-Adversarial Training")
    print(f"Target Privacy (eps_dp): {args.epsilon_dp} | Attack Strength (eps_adv): {args.epsilon_adv}")
    print("-" * 50)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        train_one_epoch_dp_adv(model, train_loader, optimizer, loss_fn, device, args.epsilon_adv)
        
        current_eps = privacy_engine.get_epsilon(1e-5)
        yt, pt = predict_proba(model, test_loader, device)
        prauc = average_precision_score(yt, pt)
        
        print(f"Epoch {epoch:02d} | Test PR-AUC: {prauc:.4f} | Spent epsilon_dp: {current_eps:.2f}")

    # Final Evaluation
    summarize(yt, pt, title="DP-Adversarial Model - Final Results")
    plot_evaluation_results(yt, pt, save_path=model_dir)
    torch.save(model.state_dict(), f"{model_dir}/dp_adv_model.pt")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

if __name__ == "__main__":
    main()