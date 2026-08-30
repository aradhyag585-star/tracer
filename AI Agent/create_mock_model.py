"""
Create a mock ML model for demo purposes.

This simulates a trained Random Forest without needing the Elliptic dataset.
The mock model gives realistic predictions based on graph features.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


def create_mock_model():
    """
    Create a mock trained model that gives realistic predictions.

    The model is trained on synthetic data that mimics criminal patterns:
    - High fan-out (sending to many addresses) = suspicious
    - Low in-degree with high out-degree = suspicious
    - High transaction counts = suspicious
    """
    print("Creating mock trained model...")

    # Generate synthetic training data with realistic patterns
    np.random.seed(42)
    n_samples = 5000

    # Create 166 features (matching Elliptic format)
    X_illicit = np.random.randn(n_samples // 2, 166)
    X_licit = np.random.randn(n_samples // 2, 166)

    # Make illicit transactions have distinctive patterns:
    # Feature 0 (in_degree): illicit wallets tend to have fewer inputs
    X_illicit[:, 0] = np.random.exponential(2, n_samples // 2)
    X_licit[:, 0] = np.random.exponential(10, n_samples // 2)

    # Feature 1 (out_degree): illicit wallets tend to have more outputs (splitting money)
    X_illicit[:, 1] = np.random.exponential(8, n_samples // 2)
    X_licit[:, 1] = np.random.exponential(3, n_samples // 2)

    # Feature 6 (fan ratio): illicit have suspicious ratios
    X_illicit[:, 6] = np.random.uniform(0.5, 2.0, n_samples // 2)
    X_licit[:, 6] = np.random.uniform(5.0, 50.0, n_samples // 2)

    # Combine data
    X = np.vstack([X_illicit, X_licit])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    # Save it
    joblib.dump(model, "wallet_risk_model.joblib")
    print(f"✅ Mock model created and saved as 'wallet_risk_model.joblib'")
    print(f"   Model size: {np.round(joblib.load('wallet_risk_model.joblib').__sizeof__() / 1e6, 2)} MB")
    print(f"\nThe model is now ready to use!")
    print(f"Run: python analyze.py 149w62rY42aZBox8fGcmqNsXUzSStKeq8C")

    return model


if __name__ == "__main__":
    create_mock_model()
