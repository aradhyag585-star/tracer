"""
Retrain the ML model with enhanced multi-typology blockchain features.

This enriches the Random Forest model with calibrated graph-topological
and on-chain UTXO behavioral features for illicit and licit transaction typologies:
  1. Peeling chains & mixing services (high symmetry, rapid passthrough)
  2. Scam & ransomware aggregators (victim consolidation, rapid liquidation)
  3. High-taint intermediaries (taint propagation)
  4. Compliant exchanges & institutional vaults (high volume, bidirectional)
  5. Long-term cold storage / genesis holders (0% liquidation, accumulation)
  6. Standard personal wallets (normal small transfers)
"""

import os
import shutil
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


def generate_calibrated_training_data(n_samples=10000):
    """
    Generate calibrated feature vectors across 6 realistic blockchain typologies.
    Aligned with the 20 core graph+UTXO features extracted in analyze.py.
    """
    np.random.seed(42)
    num_features = 166
    n_per_typology = n_samples // 6

    X_list = []
    y_list = []

    # ==========================================
    # ILLICIT TYPOLOGIES (Label = 1)
    # ==========================================

    # 1. Peeling Chains & Mixers (Symmetric high fan-out, rapid passthrough, near 100% liquidation)
    X_mix = np.zeros((n_per_typology, num_features))
    X_mix[:, 0] = np.random.randint(5, 35, n_per_typology)         # in_degree
    X_mix[:, 1] = np.random.randint(5, 35, n_per_typology)         # out_degree (symmetric)
    X_mix[:, 2] = np.random.uniform(1.0, 80.0, n_per_typology)     # funded_btc
    X_mix[:, 3] = X_mix[:, 2] * np.random.uniform(0.95, 1.0, n_per_typology) # spent_btc
    X_mix[:, 4] = X_mix[:, 2] - X_mix[:, 3]                       # balance near 0
    X_mix[:, 5] = X_mix[:, 3] / np.maximum(X_mix[:, 2], 0.001)    # liq_ratio ~ 0.98
    X_mix[:, 6] = X_mix[:, 0] / np.maximum(X_mix[:, 1], 1)        # fan_ratio ~ 1.0
    X_mix[:, 7] = X_mix[:, 2] / np.maximum(X_mix[:, 3], 0.001)
    X_mix[:, 8] = X_mix[:, 2] / np.maximum(X_mix[:, 0], 1)
    X_mix[:, 9] = X_mix[:, 4]                                     # net flow near 0
    X_mix[:, 10] = 1.0                                            # is_passthrough
    X_mix[:, 11] = 0.0                                            # scam_funnel
    X_mix[:, 12] = 1.0                                            # mixer_indicator
    X_mix[:, 13] = 0.0                                            # exchange_indicator
    X_mix[:, 14] = np.random.uniform(10.0, 90.0, n_per_typology)  # taint_score
    X_mix[:, 15] = np.random.uniform(0.01, 0.15, n_per_typology) # clustering
    X_mix[:, 16] = np.random.uniform(0.02, 0.20, n_per_typology) # density
    X_mix[:, 17] = X_mix[:, 0] + X_mix[:, 1] + 1                 # n_nodes
    X_mix[:, 18] = X_mix[:, 0] + X_mix[:, 1]                     # n_edges
    X_mix[:, 19] = X_mix[:, 0] + X_mix[:, 1]                     # tx_count
    X_list.append(X_mix)
    y_list.append(np.ones(n_per_typology, dtype=int))

    # 2. Scam / Ransomware Aggregators (High victim in-degree, low out-degree, >75% liquidation)
    X_scam = np.zeros((n_per_typology, num_features))
    X_scam[:, 0] = np.random.randint(8, 1500, n_per_typology)      # in_degree / victims
    X_scam[:, 1] = np.random.randint(0, 3, n_per_typology)         # out_degree (concentrated exit)
    X_scam[:, 2] = np.random.uniform(2.0, 500.0, n_per_typology)   # funded_btc
    X_scam[:, 3] = X_scam[:, 2] * np.random.uniform(0.75, 1.0, n_per_typology) # spent_btc
    X_scam[:, 4] = X_scam[:, 2] - X_scam[:, 3]
    X_scam[:, 5] = X_scam[:, 3] / np.maximum(X_scam[:, 2], 0.001) # liq_ratio >= 0.75
    X_scam[:, 6] = X_scam[:, 0] / np.maximum(X_scam[:, 1], 1)     # high fan_ratio
    X_scam[:, 7] = X_scam[:, 2] / np.maximum(X_scam[:, 3], 0.001)
    X_scam[:, 8] = X_scam[:, 2] / np.maximum(X_scam[:, 0], 1)
    X_scam[:, 9] = X_scam[:, 4]
    X_scam[:, 10] = 0.0
    X_scam[:, 11] = 1.0                                           # scam_funnel
    X_scam[:, 12] = 0.0
    X_scam[:, 13] = 0.0
    X_scam[:, 14] = np.random.uniform(0.0, 100.0, n_per_typology) # taint_score
    X_scam[:, 15] = 0.0
    X_scam[:, 16] = np.random.uniform(0.01, 0.08, n_per_typology)
    X_scam[:, 17] = np.minimum(X_scam[:, 0] + 5, 100)
    X_scam[:, 18] = np.minimum(X_scam[:, 0] + 2, 95)
    X_scam[:, 19] = X_scam[:, 0] + np.random.randint(1, 10, n_per_typology)
    X_list.append(X_scam)
    y_list.append(np.ones(n_per_typology, dtype=int))

    # 3. High-Taint Intermediary / Launderer (High taint, high liquidation)
    X_taint = np.zeros((n_per_typology, num_features))
    X_taint[:, 0] = np.random.randint(2, 15, n_per_typology)
    X_taint[:, 1] = np.random.randint(1, 6, n_per_typology)
    X_taint[:, 2] = np.random.uniform(0.5, 50.0, n_per_typology)
    X_taint[:, 3] = X_taint[:, 2] * np.random.uniform(0.85, 1.0, n_per_typology)
    X_taint[:, 4] = X_taint[:, 2] - X_taint[:, 3]
    X_taint[:, 5] = X_taint[:, 3] / np.maximum(X_taint[:, 2], 0.001)
    X_taint[:, 6] = X_taint[:, 0] / np.maximum(X_taint[:, 1], 1)
    X_taint[:, 7] = X_taint[:, 2] / np.maximum(X_taint[:, 3], 0.001)
    X_taint[:, 8] = X_taint[:, 2] / np.maximum(X_taint[:, 0], 1)
    X_taint[:, 9] = X_taint[:, 4]
    X_taint[:, 10] = 0.0
    X_taint[:, 11] = 0.0
    X_taint[:, 12] = 0.0
    X_taint[:, 13] = 0.0
    X_taint[:, 14] = np.random.uniform(40.0, 100.0, n_per_typology) # Critical taint
    X_taint[:, 15] = 0.0
    X_taint[:, 16] = 0.05
    X_taint[:, 17] = X_taint[:, 0] + X_taint[:, 1] + 1
    X_taint[:, 18] = X_taint[:, 0] + X_taint[:, 1]
    X_taint[:, 19] = X_taint[:, 0] + X_taint[:, 1]
    X_list.append(X_taint)
    y_list.append(np.ones(n_per_typology, dtype=int))

    # ==========================================
    # LICIT TYPOLOGIES (Label = 0)
    # ==========================================

    # 4. Standard Personal Wallets (Low degree, low liquidation, small volume, 0 taint)
    X_personal = np.zeros((n_per_typology, num_features))
    X_personal[:, 0] = np.random.randint(1, 6, n_per_typology)
    X_personal[:, 1] = np.random.randint(1, 6, n_per_typology)
    X_personal[:, 2] = np.random.exponential(0.5, n_per_typology)   # small amounts
    X_personal[:, 3] = X_personal[:, 2] * np.random.uniform(0.0, 0.60, n_per_typology) # low liquidation
    X_personal[:, 4] = X_personal[:, 2] - X_personal[:, 3]          # retains balance
    X_personal[:, 5] = X_personal[:, 3] / np.maximum(X_personal[:, 2], 0.001)
    X_personal[:, 6] = X_personal[:, 0] / np.maximum(X_personal[:, 1], 1)
    X_personal[:, 7] = X_personal[:, 2] / np.maximum(X_personal[:, 3], 0.001)
    X_personal[:, 8] = X_personal[:, 2] / np.maximum(X_personal[:, 0], 1)
    X_personal[:, 9] = X_personal[:, 4]
    X_personal[:, 10] = 0.0
    X_personal[:, 11] = 0.0
    X_personal[:, 12] = 0.0
    X_personal[:, 13] = 0.0
    X_personal[:, 14] = 0.0                                         # 0 taint
    X_personal[:, 15] = 0.0
    X_personal[:, 16] = 0.03
    X_personal[:, 17] = X_personal[:, 0] + X_personal[:, 1] + 1
    X_personal[:, 18] = X_personal[:, 0] + X_personal[:, 1]
    X_personal[:, 19] = X_personal[:, 0] + X_personal[:, 1]
    X_list.append(X_personal)
    y_list.append(np.zeros(n_per_typology, dtype=int))

    # 5. Cold Storage / Genesis / Long-term Holders (0% liquidation, accumulation)
    X_cold = np.zeros((n_per_typology, num_features))
    X_cold[:, 0] = np.random.randint(1, 5000, n_per_typology)       # deposits received
    X_cold[:, 1] = np.zeros(n_per_typology)                         # 0 out_degree
    X_cold[:, 2] = np.random.uniform(10.0, 100000.0, n_per_typology)
    X_cold[:, 3] = 0.0                                             # 0 spent
    X_cold[:, 4] = X_cold[:, 2]                                    # 100% retained
    X_cold[:, 5] = 0.0                                             # 0.0 liquidation
    X_cold[:, 6] = X_cold[:, 0]
    X_cold[:, 7] = 1000.0
    X_cold[:, 8] = X_cold[:, 2] / np.maximum(X_cold[:, 0], 1)
    X_cold[:, 9] = X_cold[:, 2]
    X_cold[:, 10] = 0.0
    X_cold[:, 11] = 0.0
    X_cold[:, 12] = 0.0
    X_cold[:, 13] = 0.0
    X_cold[:, 14] = 0.0
    X_cold[:, 15] = 0.0
    X_cold[:, 16] = 0.01
    X_cold[:, 17] = np.minimum(X_cold[:, 0] + 1, 50)
    X_cold[:, 18] = np.minimum(X_cold[:, 0], 50)
    X_cold[:, 19] = X_cold[:, 0]
    X_list.append(X_cold)
    y_list.append(np.zeros(n_per_typology, dtype=int))

    # 6. Verified Exchanges & Large Custodians (Massive bidirectional flow, high volume)
    X_exch = np.zeros((n_per_typology, num_features))
    X_exch[:, 0] = np.random.randint(50, 5000, n_per_typology)     # in_degree
    X_exch[:, 1] = np.random.randint(30, 3000, n_per_typology)     # out_degree
    X_exch[:, 2] = np.random.uniform(500.0, 1000000.0, n_per_typology)
    X_exch[:, 3] = X_exch[:, 2] * np.random.uniform(0.60, 0.85, n_per_typology)
    X_exch[:, 4] = X_exch[:, 2] - X_exch[:, 3]
    X_exch[:, 5] = X_exch[:, 3] / np.maximum(X_exch[:, 2], 0.001)
    X_exch[:, 6] = X_exch[:, 0] / np.maximum(X_exch[:, 1], 1)
    X_exch[:, 7] = X_exch[:, 2] / np.maximum(X_exch[:, 3], 0.001)
    X_exch[:, 8] = X_exch[:, 2] / np.maximum(X_exch[:, 0], 1)
    X_exch[:, 9] = X_exch[:, 4]
    X_exch[:, 10] = 0.0
    X_exch[:, 11] = 0.0
    X_exch[:, 12] = 0.0
    X_exch[:, 13] = 1.0                                           # exchange_indicator
    X_exch[:, 14] = 0.0
    X_exch[:, 15] = np.random.uniform(0.01, 0.10, n_per_typology)
    X_exch[:, 16] = 0.05
    X_exch[:, 17] = 200
    X_exch[:, 18] = 250
    X_exch[:, 19] = X_exch[:, 0] + X_exch[:, 1]
    X_list.append(X_exch)
    y_list.append(np.zeros(n_per_typology, dtype=int))

    # Combine
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Add minor noise to remaining features (indices 20..165)
    X[:, 20:] = np.random.normal(0, 0.005, (len(X), num_features - 20))

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    return X[shuffle_idx], y[shuffle_idx]


def retrain_with_known_criminals(model_path="wallet_risk_model.joblib"):
    """
    Train and save the calibrated multi-typology Random Forest model.
    """
    print("Training calibrated multi-typology ML model...")

    if os.path.exists(model_path):
        backup_path = model_path.replace(".joblib", "_backup.joblib")
        try:
            shutil.copy2(model_path, backup_path)
            print(f"  Backed up existing model to '{backup_path}'")
        except Exception as e:
            print(f"  [Warning] Backup failed: {e}")

    X, y = generate_calibrated_training_data(n_samples=12000)

    print(f"  Training on {len(X)} samples across 6 blockchain typologies...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=15,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"  ✅ Calibrated model trained and saved to '{model_path}'")
    return model


if __name__ == "__main__":
    retrain_with_known_criminals()
