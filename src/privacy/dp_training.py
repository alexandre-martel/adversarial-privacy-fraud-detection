import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import joblib

from src.utils import (
    set_seed, load_dataset, get_datasets, standardize, 
    plot_training_history, plot_evaluation_results, summarize
)
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def plot_privacy_tradeoff(history, save_path):
    """
    Visualizes the Privacy-Utility Trade-off.
    Plots how the PR-AUC (Utility) evolves as the Epsilon budget (Privacy) is spent.
    """
    epochs = range(1, len(history['epsilons']) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Epsilon budget
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Epsilon_dp', color='tab:red')
    ax1.plot(epochs, history['epsilons'], color='tab:red', marker='o', label='Privacy Budget (epsilon_dp)')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Performance (PR-AUC)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test PR-AUC', color='tab:blue')
    ax2.plot(epochs, history['val_prauc'], color='tab:blue', marker='s', label='Utility (PR-AUC)')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('Privacy-Utility Trade-off (Differential Privacy Audit)')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "privacy_utility_tradeoff.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Differential Privacy (DP) Training for Fraud Detection")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target privacy budget")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Output directory setup
    model_dir = "results/dp_model"
    os.makedirs(model_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loading and Preprocessing
    X, y, _ = load_dataset(args.data_path)
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_datasets(X, y, test_size=0.2, val_size=0.2, random_state=args.seed)
    X_tr_s, X_val_s, X_te_s, scaler, _, _ = standardize(X_tr, X_val, X_te)

    # Wrapping data in Loaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), 
                                            torch.tensor(y_tr, dtype=torch.float32)), 
                              batch_size=args.batch_size, shuffle=True)
    
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te_s, dtype=torch.float32), 
                                           torch.tensor(y_te, dtype=torch.float32)), batch_size=1024)

    # Model Initialization
    model = MLP(in_dim=X_tr_s.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # We use pos_weight because DP noise makes class imbalance much harder to handle.
    # A value of 500 is roughly the ratio of (Non-Fraud / Fraud).
    pos_weight = torch.tensor([500.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

    # Opacus Privacy Engine Setup
    # This transforms the model, optimizer, and loader to be DP-compliant.
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.epsilon,
        target_delta=1e-5,        # Probability of a privacy breach (usually 1/N)
        epochs=args.epochs,
        max_grad_norm=1.0,         # Clips individual gradients to limit sample impact
    )

    history = {"val_prauc": [], "epsilons": []}

    print(f"Starting DP Training (Target Epsilon_dp: {args.epsilon})")
    print("-" * 30)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            logits = model(xb)
            loss = loss_fn(logits, yb.view_as(logits))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Track current Privacy Budget spent
        current_eps = privacy_engine.get_epsilon(1e-5)
        
        # Evaluate on Test Set
        yt, pt = predict_proba(model, test_loader, device)
        prauc = average_precision_score(yt, pt)
        
        history["val_prauc"].append(prauc)
        history["epsilons"].append(current_eps)
        
        print(f"Epoch {epoch:02d} | Test PR-AUC: {prauc:.4f} | Spent epsilon: {current_eps:.2f}")

    # Final Results and Visualization
    plot_privacy_tradeoff(history, model_dir)
    
    print("\nTraining Complete. Final Evaluation:")
    summarize(yt, pt, title="Differentially Private Model - Final Results")
    plot_evaluation_results(yt, pt, save_path=model_dir)

    # Save the DP-trained weights
    torch.save(model.state_dict(), f"{model_dir}/dp_model.pt")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Model saved in {model_dir}/dp_model.pt")

if __name__ == "__main__":
    main()