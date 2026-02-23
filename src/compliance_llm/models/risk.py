from typing import Dict

RISK_WEIGHTS = {
    "AML": 0.9,
    "PRIVACY": 0.8,
    "ESG": 0.5,
    "REPORTING": 0.6,
    "AI_ACT": 0.75,
}


def risk_score(probabilities: Dict[str, float]) -> float:
    if not probabilities:
        return 0.05
    weighted = 0.0
    norm = 0.0
    for label, p in probabilities.items():
        w = RISK_WEIGHTS.get(label, 0.4)
        weighted += w * p
        norm += w
    return round(weighted / max(norm, 1e-8), 4)
