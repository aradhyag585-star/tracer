"""
Step 5: Risk Classifier — combines ML score + heuristics into a final verdict.

What this does:
  1. Takes the ML model's "illicit probability" (0.0 to 1.0)
  2. Takes the heuristic scores from heuristics.py
  3. Produces a weighted final risk score (0-100)
  4. Maps that score + evidence into a human-readable risk category
  5. Lists the specific evidence that led to the classification

The weighting formula:
  final_score = (ml_weight * ml_score)
              + (taint_weight * taint_score)
              + (ofac_weight * ofac_penalty)
              + (mixer_weight * mixer_penalty)
              - (exchange_weight * exchange_discount)

Usage:
    from risk_classifier import classify_risk
    result = classify_risk(ml_probability=0.7, heuristic_scores={...})
"""


# ---- Weight Configuration ----
# These control how much each signal contributes to the final score.
# You can tune these for your demo — they don't need to be "perfect."

WEIGHTS = {
    "ml_score": 0.40,       # 40% weight: the ML model's illicit probability
    "taint_score": 0.25,    # 25% weight: traced funds from known-bad wallets
    "ofac_penalty": 0.20,   # 20% weight: OFAC sanctions match (binary: 0 or 100)
    "mixer_penalty": 0.10,  # 10% weight: mixing-service pattern detected
    "exchange_discount": 0.05,  # 5% weight: reduces risk if likely an exchange
}


# ---- Risk Categories ----
# Mapped from score ranges + evidence patterns.

CATEGORIES = [
    {
        "name": "linked to sanctioned entity",
        "condition": lambda score, evidence: any("OFAC" in e for e in evidence),
        "priority": 1,  # Highest priority — overrides other categories
    },
    {
        "name": "likely money laundering",
        "condition": lambda score, evidence: (
            any("mixing" in e.lower() for e in evidence) and score >= 60
        ),
        "priority": 2,
    },
    {
        "name": "linked to theft/hack proceeds",
        "condition": lambda score, evidence: (
            any("taint" in e.lower() or "known-bad" in e.lower() for e in evidence)
            and score >= 50
        ),
        "priority": 3,
    },
    {
        "name": "linked to scam/fraud",
        "condition": lambda score, evidence: score >= 65,
        "priority": 4,
    },
    {
        "name": "likely legitimate exchange",
        "condition": lambda score, evidence: (
            any("exchange" in e.lower() for e in evidence) and score < 30
        ),
        "priority": 5,
    },
    {
        "name": "insufficient data",
        "condition": lambda score, evidence: len(evidence) <= 1,
        "priority": 6,
    },
    {
        "name": "low risk",
        "condition": lambda score, evidence: score < 30,
        "priority": 7,
    },
    {
        "name": "moderate risk — inconclusive",
        "condition": lambda score, evidence: True,  # Default fallback
        "priority": 8,
    },
]


def build_evidence_list(ml_probability, heuristics):
    """
    Generate human-readable evidence strings from the raw scores.

    Each string explains ONE piece of evidence that contributed to the score.
    These show up in the final output so the user understands WHY.
    """
    evidence = []

    # ML model evidence
    if ml_probability >= 0.7:
        evidence.append(
            f"ML model flags this as {ml_probability*100:.0f}% likely illicit "
            f"(trained on 200K labeled Bitcoin transactions)"
        )
    elif ml_probability >= 0.4:
        evidence.append(
            f"ML model gives a moderate illicit probability of "
            f"{ml_probability*100:.0f}%"
        )

    # Taint evidence
    taint = heuristics.get("taint_score", 0)
    if taint >= 50:
        evidence.append(
            f"High taint: {taint:.0f}% of incoming funds traced back to "
            f"known-bad wallets"
        )
    elif taint >= 10:
        evidence.append(
            f"Moderate taint: {taint:.0f}% of incoming funds linked to "
            f"known-bad wallets within 2 hops"
        )

    # OFAC evidence
    if heuristics.get("ofac_flagged", False):
        evidence.append(
            "OFAC MATCH: This address appears on the US Treasury sanctions list"
        )

    # Mixer evidence
    if heuristics.get("mixer_pattern_detected", False):
        fan_in = heuristics.get("fan_in", 0)
        fan_out = heuristics.get("fan_out", 0)
        evidence.append(
            f"High fan-out pattern typical of mixing services "
            f"(fan-in={fan_in}, fan-out={fan_out}, ratio="
            f"{heuristics.get('fan_ratio', 0):.1f})"
        )

    # Exchange evidence
    exchange_conf = heuristics.get("exchange_confidence", 0)
    if exchange_conf >= 60:
        evidence.append(
            f"Likely a known exchange or large service "
            f"(exchange confidence: {exchange_conf}%)"
        )
    elif exchange_conf >= 30:
        evidence.append(
            f"Moderate exchange likelihood "
            f"(exchange confidence: {exchange_conf}%)"
        )

    return evidence


def compute_final_score(ml_probability, heuristics):
    """
    Combine all signals into one weighted score (0-100).

    Each signal is scaled to 0-100, then multiplied by its weight.
    """
    # ML score: probability 0.0-1.0 → scale to 0-100
    ml_component = ml_probability * 100 * WEIGHTS["ml_score"]

    # Taint score: already 0-100
    taint_component = heuristics.get("taint_score", 0) * WEIGHTS["taint_score"]

    # OFAC: binary — either 0 or 100
    ofac_component = (100 if heuristics.get("ofac_flagged", False) else 0) * \
                     WEIGHTS["ofac_penalty"]

    # Mixer: binary — either 0 or 100
    mixer_component = (100 if heuristics.get("mixer_pattern_detected", False)
                       else 0) * WEIGHTS["mixer_penalty"]

    # Exchange discount: reduces the score (legitimate exchanges are less risky)
    exchange_component = heuristics.get("exchange_confidence", 0) * \
                         WEIGHTS["exchange_discount"]

    raw_score = (ml_component + taint_component + ofac_component
                 + mixer_component - exchange_component)

    # Clamp to 0-100
    return round(max(0, min(100, raw_score)), 1)


def determine_category(score, evidence):
    """
    Pick the most appropriate risk category based on score + evidence.

    Categories are checked in priority order (most severe first).
    The first one whose condition matches wins.
    """
    sorted_categories = sorted(CATEGORIES, key=lambda c: c["priority"])
    for cat in sorted_categories:
        if cat["condition"](score, evidence):
            return cat["name"]
    return "moderate risk — inconclusive"  # Should never reach here


def classify_risk(ml_probability, heuristic_scores):
    """
    Main classification function.

    Args:
        ml_probability: float 0.0-1.0 from the Random Forest model
                        (probability of being illicit)
        heuristic_scores: dict from heuristics.compute_all_heuristics()

    Returns:
        dict with:
          'risk_score':    int 0-100
          'risk_category': human-readable string
          'evidence':      list of explanation strings
    """
    evidence = build_evidence_list(ml_probability, heuristic_scores)
    score = compute_final_score(ml_probability, heuristic_scores)
    category = determine_category(score, evidence)

    return {
        "risk_score": score,
        "risk_category": category,
        "evidence": evidence,
    }


# Quick self-test
if __name__ == "__main__":
    # Simulate a suspicious wallet
    print("=== Test 1: Suspicious wallet ===")
    result = classify_risk(
        ml_probability=0.85,
        heuristic_scores={
            "exchange_confidence": 10,
            "fan_in": 15,
            "fan_out": 12,
            "fan_ratio": 1.25,
            "mixer_pattern_detected": True,
            "taint_score": 35,
            "ofac_flagged": False,
        }
    )
    print(f"  Score:    {result['risk_score']}")
    print(f"  Category: {result['risk_category']}")
    print(f"  Evidence:")
    for e in result["evidence"]:
        print(f"    - {e}")

    # Simulate a clean exchange wallet
    print("\n=== Test 2: Clean exchange wallet ===")
    result = classify_risk(
        ml_probability=0.05,
        heuristic_scores={
            "exchange_confidence": 85,
            "fan_in": 200,
            "fan_out": 5,
            "fan_ratio": 40.0,
            "mixer_pattern_detected": False,
            "taint_score": 0,
            "ofac_flagged": False,
        }
    )
    print(f"  Score:    {result['risk_score']}")
    print(f"  Category: {result['risk_category']}")
    print(f"  Evidence:")
    for e in result["evidence"]:
        print(f"    - {e}")

    # Simulate an OFAC-sanctioned wallet
    print("\n=== Test 3: OFAC-sanctioned wallet ===")
    result = classify_risk(
        ml_probability=0.5,
        heuristic_scores={
            "exchange_confidence": 0,
            "fan_in": 3,
            "fan_out": 2,
            "fan_ratio": 1.5,
            "mixer_pattern_detected": False,
            "taint_score": 0,
            "ofac_flagged": True,
        }
    )
    print(f"  Score:    {result['risk_score']}")
    print(f"  Category: {result['risk_category']}")
    print(f"  Evidence:")
    for e in result["evidence"]:
        print(f"    - {e}")
