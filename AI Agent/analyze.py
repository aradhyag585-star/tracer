"""
Step 6: Main Analysis Pipeline — analyze_wallet(address).

The unified entry point for Bitcoin Wallet Risk Analysis.
Integrates on-chain graph topology, machine learning inference,
proportional taint propagation, and verified threat intelligence.

Usage:
    from analyze import analyze_wallet
    result = analyze_wallet("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
"""

import os
import numpy as np
import joblib
import networkx as nx

from blockchain_api import (
    build_transaction_graph,
    is_valid_bitcoin_address,
    get_address_summary,
)
from heuristics import compute_all_heuristics, load_ofac_addresses
from risk_classifier import classify_risk
from threat_intel import detect_foreign_blockchain, query_threat_intel
from visualize_graph import visualize_transaction_graph, create_hop_distribution_chart


MODEL_PATH = "wallet_risk_model.joblib"
KNOWN_BAD_ADDRESSES = set()


def load_known_bad_addresses_from_graph(graph):
    """
    Automatically detect known-bad addresses that appear in the traced graph.
    """
    ofac_set = load_ofac_addresses()
    bad_in_graph = set()
    for node in graph.nodes():
        if node in ofac_set:
            bad_in_graph.add(node)
    return bad_in_graph


def load_model(model_path=MODEL_PATH):
    """Load the trained Random Forest model from disk."""
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def extract_graph_features(graph, target_address, summary=None, heuristics=None, num_features=166):
    """
    Extract comprehensive graph-topological and behavioral UTXO features for ML inference.
    Aligned with the multi-typology Random Forest model in retrain_model.py.
    """
    features = np.zeros(num_features)
    if target_address not in graph and summary is None:
        return features

    # Graph degree
    in_degree = graph.in_degree(target_address) if target_address in graph else 0
    out_degree = graph.out_degree(target_address) if target_address in graph else 0

    # Graph flow
    in_btc_graph = sum(
        graph[s][target_address].get("weight", 0.0)
        for s in graph.predecessors(target_address)
    ) if target_address in graph else 0.0

    out_btc_graph = sum(
        graph[target_address][r].get("weight", 0.0)
        for r in graph.successors(target_address)
    ) if target_address in graph else 0.0

    # On-chain lifetime UTXO metrics (from summary if available)
    if summary:
        tx_count = summary.get("tx_count", in_degree + out_degree)
        funded_btc = summary.get("funded_btc", in_btc_graph)
        spent_btc = summary.get("spent_btc", out_btc_graph)
        balance_btc = summary.get("current_balance_btc", max(0.0, funded_btc - spent_btc))
        liq_ratio = summary.get("liquidation_ratio", spent_btc / max(funded_btc, 0.001))
    else:
        tx_count = in_degree + out_degree
        funded_btc = in_btc_graph
        spent_btc = out_btc_graph
        balance_btc = max(0.0, funded_btc - spent_btc)
        liq_ratio = spent_btc / max(funded_btc, 0.001)

    # Heuristic indicators
    taint = 0.0
    is_mixer = 0.0
    is_scam = 0.0
    is_exchange = 0.0
    if heuristics:
        taint = float(heuristics.get("taint_score", 0.0))
        is_mixer = 1.0 if heuristics.get("mixer_pattern_detected", False) else 0.0
        is_scam = 1.0 if heuristics.get("scam_aggregator_detected", False) else 0.0
        is_exchange = 1.0 if heuristics.get("exchange_confidence", 0) >= 65 else 0.0

    # Graph structural metrics
    try:
        clustering_coeff = nx.clustering(graph.to_undirected(), target_address) if target_address in graph else 0.0
    except Exception:
        clustering_coeff = 0.0

    n_nodes = graph.number_of_nodes() if graph is not None else 0
    n_edges = graph.number_of_edges() if graph is not None else 0
    density = n_edges / max(n_nodes * (n_nodes - 1), 1) if n_nodes > 1 else 0.0

    effective_in = max(in_degree, tx_count if out_degree == 0 and liq_ratio == 0 else in_degree)
    fan_ratio = effective_in / max(out_degree, 1)
    btc_ratio = funded_btc / max(spent_btc, 0.001)
    avg_in_size = funded_btc / max(effective_in, 1)
    net_flow = funded_btc - spent_btc
    is_passthrough = 1.0 if (in_degree >= 3 and out_degree >= 3 and liq_ratio >= 0.85) else 0.0

    # Map core 20 features
    features[0] = effective_in
    features[1] = out_degree
    features[2] = funded_btc
    features[3] = spent_btc
    features[4] = balance_btc
    features[5] = liq_ratio
    features[6] = fan_ratio
    features[7] = btc_ratio
    features[8] = avg_in_size
    features[9] = net_flow
    features[10] = is_passthrough
    features[11] = is_scam
    features[12] = is_mixer
    features[13] = is_exchange
    features[14] = taint
    features[15] = clustering_coeff
    features[16] = density
    features[17] = n_nodes
    features[18] = n_edges
    features[19] = tx_count

    return features


def get_ml_probability(model, graph, target_address, summary=None, heuristics=None):
    """
    Predict illicit probability using the trained ML model with full graph + on-chain features.
    """
    if model is None:
        return 0.35  # Neutral default

    features = extract_graph_features(
        graph,
        target_address,
        summary=summary,
        heuristics=heuristics,
        num_features=166,
    )
    try:
        probabilities = model.predict_proba(features.reshape(1, -1))
        return float(probabilities[0][1])
    except Exception:
        return 0.35


def analyze_wallet(address, max_hops=5, max_addresses_per_hop=10):
    """
    Analyze a Bitcoin wallet address for risk.

    Args:
        address: Bitcoin wallet address (string)
        max_hops: trace depth (default 5)
        max_addresses_per_hop: fan-out limit per level (default 10)

    Returns:
        Clean dictionary containing scores, evidence, metrics, and visualization paths.
    """
    print(f"\n{'='*60}")
    print(f"BITCOIN WALLET RISK ANALYZER: {address}")
    print(f"{'='*60}")

    # 1. Check for foreign blockchain addresses (ETH, TRON, SOL)
    foreign_check = detect_foreign_blockchain(address)
    if foreign_check["is_foreign"]:
        print(f"  [Notice] Foreign Blockchain Address Detected: {foreign_check['blockchain']}")
        return {
            "address": address,
            "risk_score": 0.0,
            "risk_category": "foreign blockchain address detected",
            "evidence": [foreign_check["guidance"]],
            "graph_stats": {"nodes": 0, "edges": 0},
            "heuristics": {},
            "ml_probability": 0.0,
            "visualizations": {"graph_image": None, "hop_chart": None},
            "threat_intel": None,
        }

    # 2. Validate Bitcoin address format (Base58 & Bech32/Bech32m)
    if not is_valid_bitcoin_address(address):
        print(f"  [Error] Invalid Bitcoin address format: '{address}'")
        return {
            "address": address,
            "risk_score": 0.0,
            "risk_category": "invalid address format",
            "evidence": ["The provided string is not a valid Bitcoin address format (P2PKH, P2SH, SegWit bc1q, or Taproot bc1p)."],
            "graph_stats": {"nodes": 0, "edges": 0},
            "heuristics": {},
            "ml_probability": 0.0,
            "visualizations": {"graph_image": None, "hop_chart": None},
            "threat_intel": None,
        }

    # 3. Retrieve on-chain summary statistics
    print("\n[1/5] Fetching on-chain summary & UTXO activity ...")
    summary = get_address_summary(address)
    threat_match = query_threat_intel(address)
    print(f"  Transactions: {summary['tx_count']:,} | Total Received: {summary['funded_btc']:.4f} BTC | Balance: {summary['current_balance_btc']:.4f} BTC")

    # 4. Handle completely inactive addresses (0 transactions)
    if summary["tx_count"] == 0 and not threat_match["is_illicit"]:
        print("\n[Notice] Address has 0 on-chain transactions (inactive/unused).")
        return {
            "address": address,
            "risk_score": 0.0,
            "risk_category": "inactive / unused address (no on-chain history)",
            "evidence": ["On-Chain Activity: 0 transactions recorded on the Bitcoin mainnet (inactive/unfunded address)"],
            "graph_stats": {"nodes": 0, "edges": 0},
            "heuristics": {"onchain_summary": summary, "threat_intel_match": threat_match},
            "ml_probability": 0.0,
            "visualizations": {"graph_image": None, "hop_chart": None},
            "threat_intel": threat_match if threat_match["is_known"] else None,
        }

    # 5. Build transaction graph
    print("\n[2/5] Building multi-hop transaction graph ...")
    graph = build_transaction_graph(
        address,
        max_hops=max_hops,
        max_addresses_per_hop=max_addresses_per_hop,
    )

    # 6. Compute Heuristics, Taint & Threat Intelligence
    print("\n[3/5] Computing heuristics, taint propagation & sanctions ...")
    bad_seeds = load_known_bad_addresses_from_graph(graph) | KNOWN_BAD_ADDRESSES
    heuristic_scores = compute_all_heuristics(
        graph, address, bad_seeds, onchain_summary=summary
    )

    # 7. ML Behavioral Model Prediction
    print("\n[4/5] Computing ML behavioral graph features ...")
    model = load_model()
    ml_prob = get_ml_probability(
        model, graph, address, summary=summary, heuristics=heuristic_scores
    )
    print(f"  ML Illicit Probability: {ml_prob*100:.1f}%")

    # 8. Classify Risk & Aggregate Evidence
    print("\n[5/5] Synthesizing risk score & forensic attribution ...")
    classification = classify_risk(ml_prob, heuristic_scores)

    # 9. Generate Visualizations
    try:
        graph_img = visualize_transaction_graph(graph, address)
        chart_img = create_hop_distribution_chart(graph, address)
    except Exception as e:
        print(f"  [Visualization Warning]: {e}")
        graph_img = None
        chart_img = None

    # Assemble Final Report
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
        "visualizations": {
            "graph_image": graph_img,
            "hop_chart": chart_img,
        },
        "threat_intel": threat_match if threat_match["is_known"] else None,
    }

    # Console Summary Output
    taint_pct = heuristic_scores.get("taint_score", 0.0)
    print(f"\n{'='*60}")
    print(f"FORENSIC RISK REPORT")
    print(f"{'='*60}")
    print(f"  Target Address:  {result['address']}")
    print(f"  Risk Score:      {result['risk_score']} / 100")
    print(f"  Taint Score:     {taint_pct:.1f}%")
    print(f"  Classification:  {result['risk_category'].upper()}")
    if result.get("threat_intel"):
        print(f"  Attribution:     {result['threat_intel']['entity']} ({result['threat_intel']['category']})")
    print(f"  Forensic Evidence:")
    for e in result["evidence"]:
        print(f"    • {e}")
    print(f"  Graph Topology:  {result['graph_stats']['nodes']} addresses, {result['graph_stats']['edges']} edges traced")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    import sys
    addr = sys.argv[1] if len(sys.argv) > 1 else "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    analyze_wallet(addr)
