"""
Step 4: Comprehensive Heuristic Scoring & Entity Attribution Engine.

What this does:
  - exchange_confidence: estimates if a wallet is a legitimate exchange (0-100), with safeguards against scam aggregators
  - fan_in_out_ratio: detects mixing-service (tumbler) patterns
  - scam_aggregator_check: detects victim consolidation & rapid liquidation patterns for new/unseen scams
  - taint_score: traces continuous, proportional % of funds derived from known illicit sources,
                 incorporating multi-hop distance decay, clean coin dilution, and threat intelligence profiling
  - ofac_check: checks if a wallet is on the US sanctions list
  - threat_intelligence: queries verified threat intel database + external feeds
"""

import os
import networkx as nx
from threat_intel import (
    query_threat_intel,
    load_all_threat_intelligence,
    normalize_address,
)
from blockchain_api import get_address_summary


OFAC_FILE = "ofac_addresses.txt"

# Intrinsic base taint scores for recognized threat intelligence categories
INTRINSIC_TAINT_MAP = {
    "sanctions": 100.0,
    "state_sponsored_cybercrime": 100.0,
    "sanctioned_mixer": 98.0,
    "ransomware": 95.0,
    "ransomware_laundering": 94.0,
    "exchange_hack": 92.0,
    "theft_scam": 90.0,
    "darknet_seized": 88.0,
    "darknet_market": 88.0,
    "scam_distribution": 85.0,
    "scam": 80.0,
    "verified_exchange": 0.0,
    "historical_genesis": 0.0,
}


def load_ofac_addresses(filepath=OFAC_FILE):
    """
    Load sanctioned Bitcoin addresses from the local verified OFAC file.
    Normalizes addresses (trimmed, lowercase for bech32).
    """
    addresses = set()

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    addresses.add(normalize_address(line))
    return addresses


def exchange_confidence(graph, address, summary=None, threat_info=None):
    """
    Estimate how likely this address belongs to a legitimate exchange or custodian.

    Legitimate exchanges exhibit bidirectional activity:
      - High fan-in (deposit addresses) AND high fan-out (withdrawals)
      - Substantial cumulative transaction volume
      - Not flagged as an illicit entity in threat intelligence

    Scam pots / ransomware extortion addresses typically exhibit high fan-in (many victim deposits)
    but low fan-out (consolidated withdrawal to a single laundering hop) — these are NOT exchanges!

    Score 0-100:
      0-15:   Personal wallet or suspicious collection pot
      15-40:  Active merchant or payment processor
      40-70:  High-volume service
      70-100: Verified large exchange or institutional custodian
    """
    if threat_info and threat_info.get("is_exchange"):
        return 95

    if threat_info and threat_info.get("is_illicit"):
        return 0

    if address not in graph:
        return 0

    in_degree = graph.in_degree(address)
    out_degree = graph.out_degree(address)
    tx_count = summary.get("tx_count", 0) if summary else 0
    funded_btc = summary.get("funded_btc", 0.0) if summary else 0.0

    # Inactive wallet cannot be an exchange
    if tx_count == 0 and in_degree == 0:
        return 0

    # Scam signature: high in-degree, near-zero out-degree (victim aggregator)
    if in_degree >= 4 and out_degree <= 1:
        return 0

    # Mixer signature: High symmetric passthrough with rapid fund drainage
    liq_ratio = summary.get("liquidation_ratio", 0.0) if summary else 0.0
    fan = fan_in_out_ratio(graph, address)
    if fan["is_suspicious"] and liq_ratio >= 0.85:
        return 0

    # True exchange signature: High bidirectional connectivity and significant BTC flow
    if in_degree >= 8 and out_degree >= 4 and funded_btc >= 10.0:
        return min(85, int(40 + (in_degree + out_degree) * 2))

    if in_degree >= 5 and out_degree >= 2:
        return min(50, int(20 + in_degree * 3))

    return min(25, int(in_degree * 2))


def fan_in_out_ratio(graph, address):
    """
    Calculate the fan-in to fan-out ratio for mixing-service / tumbler detection.
    Tumblers typically have high symmetric fan-in and fan-out with passthrough.
    """
    if address not in graph:
        return {"fan_in": 0, "fan_out": 0, "ratio": 0.0, "is_suspicious": False}

    fan_in = graph.in_degree(address)
    fan_out = graph.out_degree(address)

    if fan_out == 0:
        ratio = float(fan_in)
    else:
        ratio = fan_in / fan_out

    # Mixer signature: High symmetric fan-in and fan-out
    is_suspicious = (fan_in >= 5 and fan_out >= 5 and 0.3 <= ratio <= 3.0)

    return {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "ratio": round(ratio, 2),
        "is_suspicious": is_suspicious,
    }


def detect_scam_aggregator(graph, address, summary=None):
    """
    Generalized behavioral detector for newly generated or unseen scam/extortion wallets.

    Scam/Extortion Aggregator Pattern:
      1. Receives deposits from multiple independent victim wallets (fan-in >= 3 or tx_count >= 5)
      2. Extremely low dispersion / fan-out (out_degree <= 2, sweeping funds to a single consolidation hub)
      3. High fund liquidation velocity (liquidation_ratio > 0.85 — attackers quickly cash out)
    """
    if address not in graph:
        return False

    in_degree = graph.in_degree(address)
    out_degree = graph.out_degree(address)

    tx_count = summary.get("tx_count", 0) if summary else 0
    liq_ratio = summary.get("liquidation_ratio", 0.0) if summary else 0.0
    funded_btc = summary.get("funded_btc", 0.0) if summary else 0.0

    # Pattern A: Graph-based victim consolidation
    if in_degree >= 3 and out_degree <= 1 and liq_ratio >= 0.80:
        return True

    # Pattern B: On-chain high transaction victim funnel with high liquidation
    if tx_count >= 5 and out_degree <= 2 and liq_ratio >= 0.90 and funded_btc > 0.01:
        return True

    return False


def _get_node_intrinsic_profile(node, known_bad_set, threat_db, ofac_set, graph=None, onchain_summary=None):
    """
    Determine the intrinsic baseline taint and whether the node is a clean taint absorber.

    Returns:
        (intrinsic_taint: float, is_absorber: bool)
    """
    norm_node = normalize_address(node)

    # 1. Check OFAC Sanctions
    if norm_node in ofac_set:
        return 100.0, False

    # 2. Check Threat Intelligence Database
    if norm_node in threat_db:
        item = threat_db[norm_node]
        cat = item.get("category", "")
        if cat in ["verified_exchange", "historical_genesis"]:
            return 0.0, True  # Taint Sink / Absorber
        return INTRINSIC_TAINT_MAP.get(cat, 85.0), False

    # 3. Check Known Bad / Sanctioned Seeds
    if norm_node in known_bad_set or node in known_bad_set:
        return 100.0, False

    # 4. Behavioral heuristics if present in graph
    if graph is not None and node in graph:
        if detect_scam_aggregator(graph, node, onchain_summary if norm_node == normalize_address(node) else None):
            return 75.0, False
        if fan_in_out_ratio(graph, node)["is_suspicious"]:
            return 65.0, False

    return 0.0, False


def taint_score(
    graph,
    target_address,
    known_bad_addresses=None,
    max_iterations=15,
    decay_factor=0.88,
    onchain_summary=None,
):
    """
    Calculate continuous proportional taint score (0.0% to 100.0%) for a Bitcoin address.

    Mathematical Model:
      - Proportional UTXO Inflow Attribution (Haircut / Poison Mixture)
      - Multi-Hop Distance Attenuation (gamma = 0.88 per hop across downstream laundering chains)
      - On-Chain Lifetime Dilution against total received BTC
      - Threat Intelligence Intrinsic Profiling
      - Taint Absorbers (Regulated compliant exchanges reset taint to prevent false cascading)

    Formula for each node u across iterations:
      TaintedInflow(u) = sum_{p in Pred(u)} [ w(p, u) * T(p) * (gamma ^ delta_hop) ]
      EffectiveInflow(u) = max( sum_{p} w(p, u), funded_btc(u) )
      InflowTaint(u) = (TaintedInflow(u) / EffectiveInflow(u)) * 100.0
      Taint(u) = max( IntrinsicBaseline(u), InflowTaint(u), OutflowExposure(u) )

    Returns:
        float: Calibrated taint percentage (0.0 to 100.0) rounded to 1 decimal place.
    """
    if not target_address:
        return 0.0

    if known_bad_addresses is None:
        known_bad_addresses = set()

    # Load threat intelligence & OFAC sets
    ofac_set = load_ofac_addresses()
    threat_db = load_all_threat_intelligence()
    normalized_bad = {normalize_address(a) for a in known_bad_addresses}

    norm_target = normalize_address(target_address)

    # 1. Handle case where target address is not in the transaction graph
    if graph is None or target_address not in graph:
        intrinsic_val, is_absorber = _get_node_intrinsic_profile(
            target_address, normalized_bad, threat_db, ofac_set, graph=None, onchain_summary=onchain_summary
        )
        return round(intrinsic_val, 1)

    # 2. Extract intrinsic baselines and absorber flags for all graph nodes
    intrinsic_profiles = {}
    absorbers = set()

    for node in graph.nodes():
        node_summary = onchain_summary if normalize_address(node) == norm_target else None
        base_taint, is_absorber = _get_node_intrinsic_profile(
            node, normalized_bad, threat_db, ofac_set, graph=graph, onchain_summary=node_summary
        )
        intrinsic_profiles[node] = base_taint
        if is_absorber:
            absorbers.add(node)

    # 3. If target is a verified exchange or genesis wallet, it is a clean sink (0.0%)
    if target_address in absorbers:
        return 0.0

    # 4. Check if there are any tainted/suspicious nodes in the graph
    has_any_taint_seed = any(val > 0 for val in intrinsic_profiles.values())
    if not has_any_taint_seed:
        return 0.0

    # 5. Initialize iterative state
    taint_values = dict(intrinsic_profiles)

    # 6. Iterative relaxation across the directed graph
    for _ in range(max_iterations):
        max_delta = 0.0
        new_taint = dict(taint_values)

        for node in graph.nodes():
            # Taint Absorbers (Exchanges) always remain 0.0%
            if node in absorbers:
                new_taint[node] = 0.0
                continue

            base_taint = intrinsic_profiles.get(node, 0.0)

            # A. Calculate Incoming Taint Flow
            predecessors = list(graph.predecessors(node))
            inflow_taint = 0.0

            if predecessors:
                total_inflow = 0.0
                tainted_inflow = 0.0

                for pred in predecessors:
                    edge_weight = graph[pred][node].get("weight", 0.0)
                    w = edge_weight if edge_weight > 0 else 1.0
                    total_inflow += w

                    pred_taint = taint_values.get(pred, 0.0)

                    # Compute distance attenuation factor
                    hop_pred = graph.nodes[pred].get("hop", None)
                    hop_node = graph.nodes[node].get("hop", None)

                    if hop_pred is not None and hop_node is not None and hop_node > hop_pred and hop_pred >= 0:
                        delta_hop = max(1, hop_node - hop_pred)
                        decay = decay_factor ** delta_hop
                    else:
                        decay = 1.0

                    tainted_inflow += w * (pred_taint * decay)

                # Reconcile with lifetime funded BTC for target address to capture true dilution
                if normalize_address(node) == norm_target and onchain_summary:
                    lifetime_funded = onchain_summary.get("funded_btc", 0.0)
                    effective_denom = max(total_inflow, lifetime_funded) if lifetime_funded > 0 else total_inflow
                else:
                    effective_denom = total_inflow

                if effective_denom > 0:
                    inflow_taint = (tainted_inflow / effective_denom)

            # B. Calculate Outgoing Exposure (Counterparty Risk for Target)
            outflow_taint = 0.0
            if normalize_address(node) == norm_target:
                successors = list(graph.successors(node))
                if successors:
                    total_outflow = 0.0
                    tainted_outflow = 0.0
                    for succ in successors:
                        edge_weight = graph[node][succ].get("weight", 0.0)
                        w = edge_weight if edge_weight > 0 else 1.0
                        total_outflow += w

                        succ_base = intrinsic_profiles.get(succ, 0.0)
                        if succ_base >= 60.0:
                            tainted_outflow += w * succ_base * 0.70  # Exposure coefficient

                    if total_outflow > 0:
                        outflow_taint = (tainted_outflow / total_outflow)

            # Synthesize final node taint for this iteration
            final_node_taint = min(100.0, max(base_taint, inflow_taint, outflow_taint))

            delta = abs(final_node_taint - taint_values[node])
            if delta > max_delta:
                max_delta = delta

            new_taint[node] = final_node_taint

        taint_values = new_taint
        if max_delta < 0.01:
            break

    target_score = taint_values.get(target_address, 0.0)
    return round(max(0.0, min(100.0, float(target_score))), 1)


def ofac_check(address, ofac_set=None):
    """
    Strictly check if an address appears on the OFAC sanctions list.
    """
    if ofac_set is None:
        ofac_set = load_ofac_addresses()

    normalized_addr = normalize_address(address)
    return normalized_addr in ofac_set


def compute_all_heuristics(graph, target_address, known_bad_addresses=None, onchain_summary=None):
    """
    Run all heuristic checks, entity attribution, and threat intelligence queries.

    Returns dict with:
      - exchange_confidence
      - fan_in, fan_out, fan_ratio, mixer_pattern_detected
      - scam_aggregator_detected
      - taint_score
      - ofac_flagged
      - threat_intel_match: {is_known, is_illicit, entity, category, severity, description, source}
      - onchain_summary: {tx_count, funded_btc, spent_btc, current_balance_btc}
    """
    if known_bad_addresses is None:
        known_bad_addresses = set()

    # 1. Load threat intelligence & sanctions
    ofac_set = load_ofac_addresses()
    threat_intel_db = load_all_threat_intelligence()

    # Add all illicit addresses from threat intel to bad seeds
    illicit_threat_addrs = {
        addr for addr, data in threat_intel_db.items()
        if data.get("category") not in ["verified_exchange", "historical_genesis"]
    }
    all_known_bad = known_bad_addresses | ofac_set | illicit_threat_addrs

    # 2. Query target address in threat intelligence
    threat_match = query_threat_intel(target_address)
    is_ofac = ofac_check(target_address, ofac_set) or (
        threat_match["is_known"] and "OFAC" in (threat_match.get("source") or "")
    )

    # 3. Fetch on-chain summary statistics if not provided
    if onchain_summary is None:
        onchain_summary = get_address_summary(target_address)

    # 4. Graph topological & behavioral heuristics
    fan = fan_in_out_ratio(graph, target_address)
    scam_aggregator = detect_scam_aggregator(graph, target_address, onchain_summary)
    exch_conf = exchange_confidence(graph, target_address, onchain_summary, threat_match)
    taint = taint_score(
        graph,
        target_address,
        all_known_bad,
        onchain_summary=onchain_summary,
    )

    # 5. Extract intel flags
    is_intel_flagged = threat_match["is_illicit"]
    intel_sources = [threat_match["source"]] if threat_match["source"] else []
    intel_reasons = [threat_match["description"]] if threat_match["description"] else []

    return {
        "exchange_confidence": exch_conf,
        "fan_in": fan["fan_in"],
        "fan_out": fan["fan_out"],
        "fan_ratio": fan["ratio"],
        "mixer_pattern_detected": fan["is_suspicious"],
        "scam_aggregator_detected": scam_aggregator,
        "taint_score": taint,
        "ofac_flagged": is_ofac,
        "threat_intel_match": threat_match,
        "intel_flagged": is_intel_flagged,
        "intel_sources": intel_sources,
        "intel_reasons": intel_reasons,
        "onchain_summary": onchain_summary,
    }


if __name__ == "__main__":
    g = nx.DiGraph()
    g.add_edge("149w62rY42aZBox8fGcmqNsXUzSStKeq8C", "intermediary", weight=5.0)
    g.add_edge("intermediary", "target_wallet", weight=3.0)
    g.add_edge("clean_source", "target_wallet", weight=2.0)

    res = compute_all_heuristics(g, "target_wallet")
    print("Heuristic results for target_wallet:")
    for k, v in res.items():
        print(f"  {k}: {v}")
