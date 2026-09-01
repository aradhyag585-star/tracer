"""
Step 3: Fetch Bitcoin transactions from the Blockstream API.

What this does:
  - Given a wallet address, validates address format
  - Fetches its incoming and outgoing transactions with pagination & caching
  - Follows the money through up to MAX_HOPS levels of recipients
  - Builds a NetworkX directed graph of wallet -> wallet money flow with proportional input/output attribution

The Blockstream API is:
  - Free (no API key, no signup)
  - Public (no authentication)
  - Rate-limited to ~10 requests/second (we add backoff retry and in-memory caching)

Usage:
    from blockchain_api import build_transaction_graph, is_valid_bitcoin_address
    graph = build_transaction_graph("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
"""

import re
import time
import requests
import networkx as nx


# Blockstream API base URL (free, no key needed)
BLOCKSTREAM_API = "https://blockstream.info/api"

# How many levels deep to trace transactions
MAX_HOPS = 5

# Delay between API calls (seconds) to avoid hitting rate limits
REQUEST_DELAY = 0.3

# In-memory transaction cache to avoid duplicate network calls
_TX_CACHE = {}

# Standard request headers
HEADERS = {
    "User-Agent": "BitcoinRiskAnalyzer/1.0 (Hackathon Security Suite)"
}


def is_valid_bitcoin_address(address):
    """
    Validate if a string is a valid Bitcoin address format:
      - P2PKH (Legacy): starts with 1, length 26-35, Base58 (no 0, O, I, l)
      - P2SH: starts with 3, length 26-35, Base58
      - Bech32 / Bech32m (Native SegWit / Taproot): starts with bc1, length 14-74
    """
    if not isinstance(address, str):
        return False

    addr = address.strip().strip('"\'')
    if not addr:
        return False

    # P2PKH or P2SH (Base58)
    base58_pattern = r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$'
    # Bech32 / Bech32m (SegWit v0 & Taproot v1, BIP-173 / BIP-350)
    bech32_pattern = r'^bc1[a-z0-9]{11,71}$'

    if re.match(base58_pattern, addr) or re.match(bech32_pattern, addr.lower()):
        return True

    # Testnet addresses (optional fallback)
    testnet_pattern = r'^(tb1|[2mn])[a-zA-HJ-NP-Z0-9]{25,62}$'
    if re.match(testnet_pattern, addr.lower()):
        return True

    return False


def get_address_summary(address):
    """
    Fetch on-chain balance and aggregate statistics for an address.

    Returns dict with:
      - 'tx_count': total transactions
      - 'funded_btc': total BTC ever received
      - 'spent_btc': total BTC ever spent
      - 'current_balance_btc': current balance
      - 'liquidation_ratio': % of funds liquidated (0.0 to 1.0)
    """
    addr = address.strip().strip('"\'')
    url = f"{BLOCKSTREAM_API}/address/{addr}"
    data = fetch_with_retry(url)
    if not data or not isinstance(data, dict):
        return {
            "tx_count": 0,
            "funded_btc": 0.0,
            "spent_btc": 0.0,
            "current_balance_btc": 0.0,
            "liquidation_ratio": 0.0,
        }

    chain_stats = data.get("chain_stats", {})
    funded_sat = chain_stats.get("funded_txo_sum", 0)
    spent_sat = chain_stats.get("spent_txo_sum", 0)
    tx_count = chain_stats.get("tx_count", 0)

    funded_btc = funded_sat / 1e8
    spent_btc = spent_sat / 1e8
    balance_btc = (funded_sat - spent_sat) / 1e8
    liquidation_ratio = (spent_sat / funded_sat) if funded_sat > 0 else 0.0

    return {
        "tx_count": tx_count,
        "funded_btc": round(funded_btc, 6),
        "spent_btc": round(spent_btc, 6),
        "current_balance_btc": round(max(0.0, balance_btc), 6),
        "liquidation_ratio": round(liquidation_ratio, 4),
    }


def fetch_with_retry(url, max_retries=3, backoff_factor=1.0, timeout=15):
    """
    HTTP GET request with exponential backoff for 429 (rate limits) and 5xx errors.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            if response.status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                print(f"  [Rate limited (429)] Backing off for {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff_factor * (2 ** attempt))

    return None


def get_address_txs(address, max_pages=3):
    """
    Fetch transactions for a Bitcoin address from Blockstream with pagination and in-memory caching.

    Args:
        address: Bitcoin address string
        max_pages: Maximum pages to fetch (25 txs per page, default 3 = 75 txs max)

    Returns:
        List of transaction dicts.
    """
    cache_key = f"{address}_{max_pages}"
    if cache_key in _TX_CACHE:
        return _TX_CACHE[cache_key]

    all_txs = []
    last_txid = None

    for page in range(max_pages):
        if last_txid:
            url = f"{BLOCKSTREAM_API}/address/{address}/txs/chain/{last_txid}"
        else:
            url = f"{BLOCKSTREAM_API}/address/{address}/txs"

        txs = fetch_with_retry(url)
        if not txs or not isinstance(txs, list):
            break

        all_txs.extend(txs)

        # If less than 25 returned, we've reached the end
        if len(txs) < 25:
            break

        last_txid = txs[-1].get("txid")
        if not last_txid:
            break

        time.sleep(REQUEST_DELAY)

    _TX_CACHE[cache_key] = all_txs
    return all_txs


def extract_incoming_transfers(txs, target_address):
    """
    From a list of transactions, find ones where target_address is a receiver,
    and extract sender addresses + amounts with proportional input weighting.

    Avoids double-counting when a transaction has multiple inputs or multiple outputs.

    Returns a list of (sender_address, amount_btc) tuples.
    """
    transfers = []
    for tx in txs:
        # Check if target_address is in the outputs (= receiver)
        target_vouts = [
            vout for vout in tx.get("vout", [])
            if vout.get("scriptpubkey_address") == target_address
        ]
        if not target_vouts:
            continue

        target_total_sat = sum(vout.get("value", 0) for vout in target_vouts)
        if target_total_sat <= 0:
            continue

        # Get all inputs and their values
        vins = tx.get("vin", [])
        total_vin_sat = sum(
            vin.get("prevout", {}).get("value", 0)
            for vin in vins
            if vin.get("prevout")
        )

        valid_senders = []
        for vin in vins:
            prevout = vin.get("prevout", {})
            sender = prevout.get("scriptpubkey_address")
            val = prevout.get("value", 0)
            if sender:
                valid_senders.append((sender, val))

        if not valid_senders:
            continue

        # Proportional attribution of received amount to each sender
        if total_vin_sat > 0:
            for sender, vin_val in valid_senders:
                proportion = vin_val / total_vin_sat
                amount_sat = target_total_sat * proportion
                amount_btc = amount_sat / 1e8
                if amount_btc > 0:
                    transfers.append((sender, amount_btc))
        else:
            # Equal split fallback if input values are missing
            split_btc = (target_total_sat / 1e8) / len(valid_senders)
            for sender, _ in valid_senders:
                transfers.append((sender, split_btc))

    return transfers


def extract_outgoing_transfers(txs, source_address):
    """
    From a list of transactions, find ones where source_address is a sender,
    and extract recipient addresses + amounts with proportional output attribution.

    Returns a list of (recipient_address, amount_btc) tuples.
    """
    transfers = []
    for tx in txs:
        vins = tx.get("vin", [])
        # Check if source_address is in the inputs (= sender)
        source_vins = [
            vin for vin in vins
            if vin.get("prevout", {}).get("scriptpubkey_address") == source_address
        ]
        if not source_vins:
            continue

        total_vin_sat = sum(
            vin.get("prevout", {}).get("value", 0)
            for vin in vins
            if vin.get("prevout")
        )
        source_vin_sat = sum(
            vin.get("prevout", {}).get("value", 0)
            for vin in source_vins
            if vin.get("prevout")
        )

        # Proportion of this tx funded by source_address
        source_proportion = (
            (source_vin_sat / total_vin_sat)
            if total_vin_sat > 0
            else (len(source_vins) / len(vins) if vins else 1.0)
        )

        # Collect all output addresses (= recipients)
        for vout in tx.get("vout", []):
            recipient = vout.get("scriptpubkey_address")
            amount_sat = vout.get("value", 0)
            # Skip change outputs back to self, and skip empty outputs
            if recipient and recipient != source_address and amount_sat > 0:
                amount_btc = (amount_sat * source_proportion) / 1e8  # Satoshis -> BTC
                transfers.append((recipient, amount_btc))

    return transfers


def build_transaction_graph(seed_address, max_hops=MAX_HOPS, max_addresses_per_hop=10):
    """
    Build a directed graph of money flow starting from seed_address.

    FORWARD TRACING: Traces where money WENT (outgoing transactions).
    PLUS: Fetches immediate incoming transactions for the seed address only
          (so we can calculate taint for the wallet being analyzed).

    Each node is a wallet address.
    Each edge has a 'weight' attribute = total BTC transferred.
    Each node gets a 'hop' attribute = how many steps from the seed.
    """
    graph = nx.DiGraph()
    graph.add_node(seed_address, hop=0)

    # STEP 1: Fetch incoming transactions for the SEED ADDRESS ONLY
    print(f"  Fetching incoming transactions for seed address...")
    seed_txs = get_address_txs(seed_address)
    incoming_transfers = extract_incoming_transfers(seed_txs, seed_address)

    for sender, amount in incoming_transfers[:20]:  # Limit to top 20 senders
        if sender != seed_address:  # Skip self-sends
            graph.add_edge(sender, seed_address, weight=amount)
            graph.add_node(sender, hop=-1)  # Mark as "incoming source"

    # STEP 2: BFS forward tracing - trace outgoing transactions level by level
    current_level = [seed_address]
    visited = {seed_address}

    for hop in range(1, max_hops + 1):
        next_level = []
        print(f"  Hop {hop}/{max_hops}: tracing {len(current_level)} addresses ...")

        for address in current_level[:max_addresses_per_hop]:
            txs = get_address_txs(address)
            transfers = extract_outgoing_transfers(txs, address)

            for recipient, amount in transfers:
                if graph.has_edge(address, recipient):
                    graph[address][recipient]["weight"] += amount
                else:
                    graph.add_edge(address, recipient, weight=amount)

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


if __name__ == "__main__":
    test_addr = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    print(f"Address valid: {is_valid_bitcoin_address(test_addr)}")
    print(f"Building transaction graph for {test_addr} ...")
    g = build_transaction_graph(test_addr, max_hops=2, max_addresses_per_hop=3)
    print(f"Nodes: {list(g.nodes)[:5]} ...")
    print(f"Edges: {list(g.edges(data=True))[:3]} ...")
