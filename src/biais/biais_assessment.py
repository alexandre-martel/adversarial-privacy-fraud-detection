import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, precision_recall_fscore_support

from src.utils import set_seed
from src.baselines.mlp_class import MLP
from src.baselines.baseline_mlp import predict_proba

def main():
    parser = argparse.ArgumentParser(description="Fairness Audit - Assessing Bias in Fraud Detection")
    parser.add_argument("--model-folder", type=str, default="baseline_model")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for converting probabilities to binary predictions")
    args = parser.parse_args()
    
    output_dir = "bias_assessment_results"
    os.makedirs(output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data and target labels
    X_test = np.load(f"{args.model_folder}/X_test.npy")
    y_test = np.load(f"{args.model_folder}/y_test.npy")
    
    # Extract 'Amount' feature (last column before class) to define groups for bias assessment
    # We use this as our sensitive attribute for the audit
    try:
        amt_idx = list(X_test[0]).index('Amount')
        print(f"'Amount' column found at index: {amt_idx}")
    except ValueError:
        amt_idx = -2
        print("'Amount' column not found by name, falling back to last column before class.")
        
    amounts = X_test[:, amt_idx] 
    median_amt = np.median(amounts)
    
    # Define groups: Small vs Large transactions
    mask_small = amounts <= median_amt
    mask_large = amounts > median_amt

    # Load Trained MLP Model
    model = MLP(in_dim=X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(f"{args.model_folder}/mlp_baseline.pt", map_location=device))
    
    # Run Inference
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                           torch.tensor(y_test, dtype=torch.float32)), 
                             batch_size=args.batch_size)
    
    y_true, y_probs = predict_proba(model, test_loader, device)
    y_pred = (y_probs >= 0.5).astype(int)

    # Compute Recall per Group (Assessment)
    # Recall is the most critical metric: are we missing more frauds in small transactions?
    rec_small = recall_score(y_true[mask_small], y_pred[mask_small])
    rec_large = recall_score(y_true[mask_large], y_pred[mask_large])

    
    print("\n" + "="*30)
    print("      FAIRNESS AUDIT RESULTS      ")
    print("="*30)
    print(f"Median Split Threshold: {median_amt:.4f}")
    print(f"Recall (Small Amounts): {rec_small:.4f}")
    print(f"Recall (Large Amounts): {rec_large:.4f}")
    print(f"Recall Gap (Bias):      {abs(rec_large - rec_small):.4f}")
    print("="*30)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true[mask_small], y_pred[mask_small], average='binary')
    precision_l, recall_l, f1_l, _ = precision_recall_fscore_support(y_true[mask_large], y_pred[mask_large], average='binary')

    plt.figure(figsize=(8, 6))
    plt.bar(['Small Transactions', 'Large Transactions'], [rec_small, rec_large], color=['skyblue', 'salmon'])
    plt.ylabel('Recall (Fraud Detection Rate)')
    plt.title('Recall Disparity across Transaction Amounts')
    plt.savefig(f"{output_dir}/bias_assessment.png")
    
    labels = ['Small Amounts', 'Large Amounts']
    recalls = [recall, recall_l]
    precisions = [precision, precision_l]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, recalls, width, label='Recall (Security)', color='skyblue')
    ax.bar(x + width/2, precisions, width, label='Precision (User Experience)', color='lightcoral')

    ax.set_ylabel('Scores')
    ax.set_title(f'Fairness Gap at Threshold {args.threshold}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.savefig(f"{output_dir}/precision_recall_comparison.png")
    
if __name__ == "__main__":
    main()