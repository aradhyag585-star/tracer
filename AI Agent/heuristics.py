"""
Step 4: Heuristic scoring functions for wallet risk analysis.

What this does:
  - exchange_confidence: estimates if a wallet is a known exchange (0-100)
  - fan_in_out_ratio: detects mixing-service patterns
  - taint_score: traces what % of funds came from known-bad wallets
  - ofac_check: checks if a wallet is on the US sanctions list

These scores are combined with the ML model's score in risk_classifier.py.

Usage:
    from heuristics import compute_all_heuristics
    scores = compute_all_heuristics(graph, target_address, known_bad_addresses)
"""

import os
import networkx as nx


# ---- OFAC Sanctions List ----
# Download the OFAC SDN list once from:
#   https://www.treasury.gov/ofac/downloads/sdn.csv
# Or create a simple text file with one address per line for testing.
# The file is checked locally — no live API call needed.

OFAC_FILE = "ofac_addresses.txt"


def load_ofac_addresses(filepath=OFAC_FILE):
    """
    Load sanctioned Bitcoin addresses from a local file.
    Returns a set of addresses (for fast lookup).
    """
    if not os.path.exists(filepath):
        print(f"  [Warning] OFAC file '{filepath}' not found. "
              f"OFAC checks will be skipped.")
        return set()

    with open(filepath, "r") as f:
        # Strip whitespace, skip empty lines and comments
        addresses = {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }
    return addresses


def exchange_confidence(graph, address):
    """
    Estimate how likely this address belongs to a known exchange.

    Heuristic: exchanges receive from MANY different senders.
    The more incoming connections, the more likely it's an exchange.

    Returns a score 0-100:
      0-20:   Probably a personal wallet
      20-60:  Could be a service or business
      60-100: Likely an exchange or large service

    This is YOUR existing heuristic — it stays as-is.
    """
    if address not in graph:
        return 0

    in_degree = graph.in_degree(address)  # Number of unique senders

    # Simple mapping: 0 senders = 0, 50+ senders = 100
    score = min(in_degree * 2, 100)
    return score


def fan_in_out_ratio(graph, address):
    """
    Calculate the fan-in to fan-out ratio for mixing-service detection.

    Mixing services (tumblers) have a distinctive pattern:
      - HIGH fan-in (many deposits from different wallets)
      - HIGH fan-out (many withdrawals to different wallets)
      - fan_in ≈ fan_out (roughly equal)

    Returns a dict with:
      'fan_in':  number of unique senders
      'fan_out': number of unique recipients
      'ratio':   fan_in / fan_out (close to 1.0 = suspicious mixer pattern)
      'is_suspicious': True if pattern looks like a mixer

    Normal wallets have asymmetric patterns (mostly sending OR receiving).
    """
    if address not in graph:
        return {"fan_in": 0, "fan_out": 0, "ratio": 0.0, "is_suspicious": False}

    fan_in = graph.in_degree(address)
    fan_out = graph.out_degree(address)

    # Avoid division by zero
    if fan_out == 0:
        ratio = float(fan_in)  # All inflow, no outflow
    else:
        ratio = fan_in / fan_out

    # Suspicious if BOTH fan-in and fan-out are high AND roughly equal
    is_suspicious = (fan_in >= 5 and fan_out >= 5 and 0.3 <= ratio <= 3.0)

    return {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "ratio": round(ratio, 2),
        "is_suspicious": is_suspicious,
    }


def taint_score(graph, target_address, known_bad_addresses):
    """
    Calculate what percentage of a wallet's incoming funds can be traced
    back to known-bad wallets (hacks, scams, sanctioned addresses).

    Uses reverse BFS: walks backward from target_address through incoming
    edges, checking how much of the money flow originated from bad wallets.

    Returns 0-100:
      0:    No connection to known-bad wallets
      1-30: Indirect connection (many hops away)
      30+:  Significant tainted funds
      100:  The wallet itself is known-bad
    """
    if not known_bad_addresses:
        return 0.0

    # If the wallet itself is known-bad, taint = 100%
    if target_address in known_bad_addresses:
        return 100.0

    if target_address not in graph:
        return 0.0

    # Walk backward through incoming edges
    total_incoming = 0.0
    tainted_incoming = 0.0

    # Check direct predecessors (1 hop back)
    for sender in graph.predecessors(target_address):
        edge_data = graph[sender][target_address]
        amount = edge_data.get("weight", 0)
        total_incoming += amount

        if sender in known_bad_addresses:
            tainted_incoming += amount
        else:
            # Check 2nd-level predecessors (2 hops back, reduced weight)
            for sender2 in graph.predecessors(sender):
                if sender2 in known_bad_addresses:
                    # Taint dilutes with distance: 50% per hop
                    edge2 = graph[sender2][sender]
                    amount2 = edge2.get("weight", 0)
                    # Proportion of sender's funds that came from bad wallet
                    sender_total_in = sum(
                        graph[s][sender].get("weight", 0)
                        for s in graph.predecessors(sender)
                    )
                    if sender_total_in > 0:
                        taint_fraction = amount2 / sender_total_in
                        tainted_incoming += amount * taint_fraction * 0.5

    if total_incoming == 0:
        return 0.0

    return round((tainted_incoming / total_incoming) * 100, 1)


def ofac_check(address, ofac_set=None):
    """
    Check if an address is on the OFAC sanctions list.

    Returns:
      True  = address IS sanctioned (very bad!)
      False = address is not on the list
    """
    if ofac_set is None:
        ofac_set = load_ofac_addresses()

    return address in ofac_set


def compute_all_heuristics(graph, target_address, known_bad_addresses=None):
    """
    Run all heuristic checks and return a single dict with all results.

    This is the main function other scripts call.
    """
    if known_bad_addresses is None:
        known_bad_addresses = set()

    ofac_set = load_ofac_addresses()

    # Add OFAC addresses to the known-bad set
    all_known_bad = known_bad_addresses | ofac_set

    fan = fan_in_out_ratio(graph, target_address)

    return {
        "exchange_confidence": exchange_confidence(graph, target_address),
        "fan_in": fan["fan_in"],
        "fan_out": fan["fan_out"],
        "fan_ratio": fan["ratio"],
        "mixer_pattern_detected": fan["is_suspicious"],
        "taint_score": taint_score(graph, target_address, all_known_bad),
        "ofac_flagged": ofac_check(target_address, ofac_set),
    }


# Quick self-test
if __name__ == "__main__":
    # Build a tiny test graph by hand
    g = nx.DiGraph()
    g.add_edge("bad_wallet", "middleman", weight=1.5)
    g.add_edge("middleman", "target", weight=1.0)
    g.add_edge("clean_wallet", "target", weight=3.0)

    known_bad = {"bad_wallet"}

    result = compute_all_heuristics(g, "target", known_bad)
    print("Heuristic scores for 'target':")
    for key, value in result.items():
        print(f"  {key}: {value}")
    # Expected: taint_score ≈ 12.5 (some tainted money, diluted by 1 hop)
