"""
Create a mock ML model for demo purposes.

This simulates a trained Random Forest without needing the Elliptic dataset.
The mock model gives realistic predictions based on graph features.
"""

import os
import shutil
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


def create_mock_model(model_path="wallet_risk_model.joblib"):
    """
    Create a calibrated Random Forest model trained on synthetic graph features
    representative of real-world blockchain illicit vs licit transaction typologies.

    Typologies Modeled:
      1. Money Laundering / Tumblers: High fan-out, symmetric flow, fast passthrough
      2. Ransomware Peeling Chains: Long hop chains, small single-input transfers
      3. Scam Aggregators: High in-degree, near-zero retention
      4. Exchanges & OTC Desks: High in-degree, high volume, licit
      5. Normal Personal Wallets: Low in-degree, low out-degree, licit
    """
    print("Creating calibrated graph-ML risk model...")

    # Backup existing model if present
    if os.path.exists(model_path):
        backup_path = model_path.replace(".joblib", "_backup.joblib")
        shutil.copy2(model_path, backup_path)
        print(f"  Backed up existing model to '{backup_path}'")

    np.random.seed(42)
    n_samples_per_class = 3000

    # 166 features to maintain compatibility
    X_illicit = np.zeros((n_samples_per_class, 166))
    X_licit = np.zeros((n_samples_per_class, 166))

    # --- Illicit Samples ---
    # Typology A: Mixing / Peeling chains (1500 samples)
    n_mix = n_samples_per_class // 2
    X_illicit[:n_mix, 0] = np.random.exponential(2.0, n_mix)     # in-degree
    X_illicit[:n_mix, 1] = np.random.exponential(12.0, n_mix)    # out-degree (high splitting)
    X_illicit[:n_mix, 2] = np.random.uniform(0.5, 20.0, n_mix)   # in_btc
    X_illicit[:n_mix, 3] = X_illicit[:n_mix, 2] * np.random.uniform(0.95, 1.0, n_mix) # out_btc
    X_illicit[:n_mix, 4] = np.random.randint(1, 6, n_mix)        # hop distance
    X_illicit[:n_mix, 5] = X_illicit[:n_mix, 1] + 2             # 2hop neighbors
    X_illicit[:n_mix, 6] = X_illicit[:n_mix, 0] / np.maximum(X_illicit[:n_mix, 1], 1) # fan ratio
    X_illicit[:n_mix, 7] = X_illicit[:n_mix, 2] / np.maximum(X_illicit[:n_mix, 3], 0.001)
    X_illicit[:n_mix, 8] = X_illicit[:n_mix, 2] / np.maximum(X_illicit[:n_mix, 0], 1)
    X_illicit[:n_mix, 9] = X_illicit[:n_mix, 3] / np.maximum(X_illicit[:n_mix, 1], 1)
    X_illicit[:n_mix, 10] = X_illicit[:n_mix, 2] - X_illicit[:n_mix, 3] # net flow near 0
    X_illicit[:n_mix, 11] = 1.0                                 # is_passthrough flag

    # Typology B: Scam Aggregators (1500 samples)
    X_illicit[n_mix:, 0] = np.random.exponential(25.0, n_mix)    # high victim deposits
    X_illicit[n_mix:, 1] = np.random.exponential(2.0, n_mix)     # few exit transfers
    X_illicit[n_mix:, 2] = np.random.uniform(1.0, 50.0, n_mix)
    X_illicit[n_mix:, 3] = X_illicit[n_mix:, 2] * np.random.uniform(0.9, 1.0, n_mix) # drained
    X_illicit[n_mix:, 4] = np.random.randint(0, 3, n_mix)
    X_illicit[n_mix:, 5] = X_illicit[n_mix:, 0] + 5
    X_illicit[n_mix:, 6] = X_illicit[n_mix:, 0] / np.maximum(X_illicit[n_mix:, 1], 1)
    X_illicit[n_mix:, 7] = 1.0
    X_illicit[n_mix:, 8] = X_illicit[n_mix:, 2] / np.maximum(X_illicit[n_mix:, 0], 1)
    X_illicit[n_mix:, 9] = X_illicit[n_mix:, 3] / np.maximum(X_illicit[n_mix:, 1], 1)
    X_illicit[n_mix:, 10] = X_illicit[n_mix:, 2] - X_illicit[n_mix:, 3]
    X_illicit[n_mix:, 11] = 0.0

    # Add small noise to remaining features
    X_illicit[:, 16:] = np.random.normal(0, 0.01, (n_samples_per_class, 150))

    # --- Licit Samples ---
    # Typology C: Normal personal wallets (2000 samples)
    n_personal = 2000
    X_licit[:n_personal, 0] = np.random.poisson(2, n_personal)
    X_licit[:n_personal, 1] = np.random.poisson(2, n_personal)
    X_licit[:n_personal, 2] = np.random.exponential(0.5, n_personal)
    X_licit[:n_personal, 3] = np.random.exponential(0.4, n_personal)
    X_licit[:n_personal, 4] = np.random.randint(0, 2, n_personal)
    X_licit[:n_personal, 5] = X_licit[:n_personal, 0] + X_licit[:n_personal, 1]
    X_licit[:n_personal, 6] = X_licit[:n_personal, 0] / np.maximum(X_licit[:n_personal, 1], 1)
    X_licit[:n_personal, 7] = X_licit[:n_personal, 2] / np.maximum(X_licit[:n_personal, 3], 0.001)
    X_licit[:n_personal, 8] = X_licit[:n_personal, 2] / np.maximum(X_licit[:n_personal, 0], 1)
    X_licit[:n_personal, 9] = X_licit[:n_personal, 3] / np.maximum(X_licit[:n_personal, 1], 1)
    X_licit[:n_personal, 10] = X_licit[:n_personal, 2] - X_licit[:n_personal, 3]
    X_licit[:n_personal, 11] = 0.0

    # Typology D: Compliant Exchanges / Merchants (1000 samples)
    n_exch = n_samples_per_class - n_personal
    X_licit[n_personal:, 0] = np.random.exponential(80.0, n_exch)  # massive in-degree
    X_licit[n_personal:, 1] = np.random.exponential(60.0, n_exch)  # massive out-degree
    X_licit[n_personal:, 2] = np.random.exponential(500.0, n_exch)
    X_licit[n_personal:, 3] = np.random.exponential(450.0, n_exch)
    X_licit[n_personal:, 4] = 0
    X_licit[n_personal:, 5] = 100
    X_licit[n_personal:, 6] = X_licit[n_personal:, 0] / np.maximum(X_licit[n_personal:, 1], 1)
    X_licit[n_personal:, 7] = 1.1
    X_licit[n_personal:, 8] = 5.0
    X_licit[n_personal:, 9] = 7.0
    X_licit[n_personal:, 10] = X_licit[n_personal:, 2] - X_licit[n_personal:, 3]
    X_licit[n_personal:, 11] = 0.0

    X_licit[:, 16:] = np.random.normal(0, 0.01, (n_samples_per_class, 150))

    # Combine & shuffle
    X = np.vstack([X_illicit, X_licit])
    y = np.array([1] * n_samples_per_class + [0] * n_samples_per_class)

    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    # Save
    joblib.dump(model, model_path)
    print(f"  ✅ Calibrated model created and saved as '{model_path}'")

    return model


if __name__ == "__main__":
    create_mock_model()
