# Adversarial Fraud Detection on MLP

A credit card fraud detection project using MLP that implements defenses against three issues: adversarial attacks (FGSM), privacy attacks (DG-SDG), and bias mitigation

## Explanations and Mathematical Theory

### Adversarial Attack (FGSM) /Defense Method (Adversarial training)

#### Adversarial Attack : FGSM
#### Adversarial Defense : Adversarial training
#### Results

### Privacy Attack (Membership Ineference Attack) /Defense Method (DP-SGD)

#### Privacy Attack : Membership Ineference Attack
#### Privacy Defense : DP-SGD
#### Results


### Bias & Mitigation 

## Download and Run

### Run baselines 

Run training of the baseline MLP model and summarize on val/test data.

```bash
python src/baselines/baseline.py
```

Arguments : 

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data-path` | Path for the data in data/filename.csv | `data/creditcard.csv` |
| `--seed` | Seed for randomness | `9` |
| `--learning-rate` | Learning rate of the trainng **XGBoost model only** | `0.05` |




