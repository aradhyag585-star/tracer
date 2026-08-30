"""
Step 6: The main entry point — analyze_wallet(address).

This is the ONE function you plug into your demo UI.
It calls everything else and returns a clean dictionary.

Usage:
    from analyze import analyze_wallet

    result = analyze_wallet("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")

    print(result)
    # {
    #   "address": "1A1zP1...",
    #   "risk_score": 42,
    #   "risk_category": "moderate risk — inconclusive",
    #   "evidence": ["ML model gives a moderate illicit probability of 45%", ...],
    #   "graph_stats": {"nodes": 28, "edges": 45},
    #   "heuristics": {"exchange_confidence": 15, ...}
    # }
"""

import os
import numpy as np
import joblib
import networkx as nx

from blockchain_api import build_transaction_graph
from heuristics import compute_all_heuristics
from risk_classifier import classify_risk


# Path to the trained model (created by train_model.py)
MODEL_PATH = "wallet_risk_model.joblib"

# Known bad addresses — add any addresses you want flagged as "bad seeds."
# The OFAC list is loaded separately by heuristics.py.
# You can add addresses from known hacks/scams here.
KNOWN_BAD_ADDRESSES = set()


def load_model(model_path=MODEL_PATH):
    """Load the trained Random Forest model from disk."""
    if not os.path.exists(model_path):
        print(f"[Warning] Model file '{model_path}' not found.")
        print("Run 'python train_model.py' first to create it.")
        return None
    return joblib.load(model_path)


def get_ml_probability(model, graph, target_address):
    """
    Use the trained ML model to predict illicit probability.

    IMPORTANT: The Elliptic dataset uses anonymized numeric features that
    we can't directly extract from the blockchain API. So here we use
    proxy features derived from the graph structure.

    This is a practical compromise for a demo — in production, you'd need
    the actual Elliptic feature pipeline.

    We extract 166 features (to match the model's expected input) from
    the graph structure, padding with zeros.
    """
    if model is None:
        return 0.5  # Default — model not available

    if target_address not in graph:
        return 0.5  # No graph data

    # Extract graph-based features as a proxy
    # These won't be as accurate as the real Elliptic features, but they
    # give the model SOMETHING to work with for the demo.
    in_degree = graph.in_degree(target_address)
    out_degree = graph.out_degree(target_address)

    # Total BTC received/sent through this address in the graph
    in_btc = sum(
        graph[s][target_address].get("weight", 0)
        for s in graph.predecessors(target_address)
    )
    out_btc = sum(
        graph[target_address][r].get("weight", 0)
        for r in graph.successors(target_address)
    )

    # Hop distance from the seed address
    hop = graph.nodes[target_address].get("hop", 0)

    # Number of unique predecessors/successors within 2 hops
    neighbors_2hop = len(set(
        nx.single_source_shortest_path_length(graph, target_address, cutoff=2)
    ))

    # Build a 166-element feature vector (matching Elliptic's format)
    # First ~10 features are our real graph metrics, rest are padded with 0
    features = np.zeros(166)
    features[0] = in_degree
    features[1] = out_degree
    features[2] = in_btc
    features[3] = out_btc
    features[4] = hop
    features[5] = neighbors_2hop
    features[6] = in_degree / max(out_degree, 1)  # fan-in/out ratio
    features[7] = in_btc / max(out_btc, 0.001)    # BTC in/out ratio
    features[8] = graph.number_of_nodes()          # graph size context
    features[9] = graph.number_of_edges()

    # Get illicit probability (probability of class 1)
    try:
        probabilities = model.predict_proba(features.reshape(1, -1))
        # predict_proba returns [[prob_class_0, prob_class_1]]
        illicit_prob = probabilities[0][1]
    except Exception as e:
        print(f"  [ML prediction error]: {e}")
        illicit_prob = 0.5  # Fall back to uncertain

    return float(illicit_prob)


def analyze_wallet(address, max_hops=5, max_addresses_per_hop=10):
    """
    Analyze a Bitcoin wallet address for risk.

    This is the ONE function you call from your UI.

    Args:
        address: Bitcoin wallet address (string)
        max_hops: how many levels deep to trace (default 5)
        max_addresses_per_hop: limit per level to keep API calls reasonable

    Returns:
        dict with:
          'address':        the input address
          'risk_score':     0-100 (higher = riskier)
          'risk_category':  human-readable category string
          'evidence':       list of explanation strings
          'graph_stats':    {'nodes': N, 'edges': M}
          'heuristics':     raw heuristic scores for debugging
          'ml_probability': the ML model's raw illicit probability
    """
    print(f"\n{'='*60}")
    print(f"Analyzing wallet: {address}")
    print(f"{'='*60}")

    # Step 1: Build the transaction graph
    print("\n[1/4] Building transaction graph ...")
    graph = build_transaction_graph(
        address,
        max_hops=max_hops,
        max_addresses_per_hop=max_addresses_per_hop,
    )

    # Step 2: Run ML prediction
    print("\n[2/4] Running ML prediction ...")
    model = load_model()
    ml_prob = get_ml_probability(model, graph, address)
    print(f"  ML illicit probability: {ml_prob*100:.1f}%")

    # Step 3: Compute heuristics
    print("\n[3/4] Computing heuristic scores ...")
    heuristic_scores = compute_all_heuristics(
        graph, address, KNOWN_BAD_ADDRESSES
    )
    print(f"  Exchange confidence: {heuristic_scores['exchange_confidence']}%")
    print(f"  Taint score:        {heuristic_scores['taint_score']}%")
    print(f"  OFAC flagged:       {heuristic_scores['ofac_flagged']}")
    print(f"  Mixer pattern:      {heuristic_scores['mixer_pattern_detected']}")

    # Step 4: Classify risk
    print("\n[4/4] Computing final risk classification ...")
    classification = classify_risk(ml_prob, heuristic_scores)

    # Build the final result
    result = {
        "address": address,
        "risk_score": classification["risk_score"],
        "risk_category": classification["risk_category"],
        "evidence": classification["evidence"],
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "heuristics": heuristic_scores,
        "ml_probability": round(ml_prob, 4),
    }

    # Pretty-print the result
    print(f"\n{'='*60}")
    print(f"RESULT")
    print(f"{'='*60}")
    print(f"  Address:  {result['address']}")
    print(f"  Score:    {result['risk_score']} / 100")
    print(f"  Category: {result['risk_category']}")
    print(f"  Evidence:")
    for e in result["evidence"]:
        print(f"    • {e}")
    if not result["evidence"]:
        print(f"    (no significant risk signals detected)")
    print(f"  Graph:    {result['graph_stats']['nodes']} addresses, "
          f"{result['graph_stats']['edges']} connections traced")

    return result


# ---- Run as a script ----
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        addr = sys.argv[1]
    else:
        # Default: a well-known address for testing
        addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        print(f"No address provided. Using Satoshi's address: {addr}")

    result = analyze_wallet(addr)
