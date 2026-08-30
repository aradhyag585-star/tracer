"""
Step 3: Fetch Bitcoin transactions from the Blockstream API.

What this does:
  - Given a wallet address, fetches its outgoing transactions
  - Follows the money through up to MAX_HOPS levels of recipients
  - Builds a NetworkX directed graph of wallet -> wallet money flow

The Blockstream API is:
  - Free (no API key, no signup)
  - Public (no authentication)
  - Rate-limited to ~10 requests/second (we add a small delay to be safe)

Usage:
    from blockchain_api import build_transaction_graph
    graph = build_transaction_graph("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
"""

import time
import requests
import networkx as nx


# Blockstream API base URL (free, no key needed)
BLOCKSTREAM_API = "https://blockstream.info/api"

# How many levels deep to trace transactions
MAX_HOPS = 5

# Delay between API calls (seconds) to avoid hitting rate limits
REQUEST_DELAY = 0.3


def get_address_txs(address):
    """
    Fetch all transactions for a Bitcoin address from Blockstream.

    Returns a list of transaction dicts, or an empty list on error.
    Each transaction has 'vin' (inputs) and 'vout' (outputs).
    """
    url = f"{BLOCKSTREAM_API}/address/{address}/txs"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        time.sleep(REQUEST_DELAY)  # Be polite to the free API
        return response.json()
    except requests.RequestException as e:
        print(f"  [API error for {address[:12]}...]: {e}")
        return []


def extract_outgoing_transfers(txs, source_address):
    """
    From a list of transactions, find ones where source_address is a sender,
    and extract recipient addresses + amounts.

    Returns a list of (recipient_address, amount_btc) tuples.
    """
    transfers = []
    for tx in txs:
        # Check if source_address is in the inputs (= sender)
        is_sender = any(
            vin.get("prevout", {}).get("scriptpubkey_address") == source_address
            for vin in tx.get("vin", [])
        )
        if not is_sender:
            continue

        # Collect all output addresses (= recipients)
        for vout in tx.get("vout", []):
            recipient = vout.get("scriptpubkey_address")
            amount_sat = vout.get("value", 0)
            # Skip change outputs back to self, and skip empty outputs
            if recipient and recipient != source_address and amount_sat > 0:
                amount_btc = amount_sat / 1e8  # Satoshis -> BTC
                transfers.append((recipient, amount_btc))

    return transfers


def build_transaction_graph(seed_address, max_hops=MAX_HOPS, max_addresses_per_hop=10):
    """
    Build a directed graph of money flow starting from seed_address.

    Each node is a wallet address.
    Each edge has a 'weight' attribute = total BTC transferred.
    Each node gets a 'hop' attribute = how many steps from the seed.

    max_addresses_per_hop limits how many addresses we trace per level
    (prevents the graph from exploding — important for the free API).
    """
    graph = nx.DiGraph()
    graph.add_node(seed_address, hop=0)

    # BFS: trace outgoing transactions level by level
    current_level = [seed_address]
    visited = {seed_address}

    for hop in range(1, max_hops + 1):
        next_level = []
        print(f"  Hop {hop}/{max_hops}: tracing {len(current_level)} addresses ...")

        for address in current_level[:max_addresses_per_hop]:
            txs = get_address_txs(address)
            transfers = extract_outgoing_transfers(txs, address)

            for recipient, amount in transfers:
                # Add or update edge weight
                if graph.has_edge(address, recipient):
                    graph[address][recipient]["weight"] += amount
                else:
                    graph.add_edge(address, recipient, weight=amount)

                # Track hop level for the recipient
                if recipient not in visited:
                    graph.nodes[recipient]["hop"] = hop
                    visited.add(recipient)
                    next_level.append(recipient)

        current_level = next_level
        if not current_level:
            print(f"  No new addresses found at hop {hop}, stopping.")
            break

    print(f"  Graph built: {graph.number_of_nodes()} addresses, "
          f"{graph.number_of_edges()} connections")
    return graph


# Quick self-test
if __name__ == "__main__":
    # Satoshi's address (the very first Bitcoin address ever)
    test_addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    print(f"Building transaction graph for {test_addr} ...")
    g = build_transaction_graph(test_addr, max_hops=2, max_addresses_per_hop=3)
    print(f"Nodes: {list(g.nodes)[:5]} ...")
    print(f"Edges: {list(g.edges(data=True))[:3]} ...")
