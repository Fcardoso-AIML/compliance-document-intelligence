from compliance_llm.models.risk import risk_score


def test_risk_score_bounds():
    score = risk_score({"AML": 0.9, "PRIVACY": 0.4})
    assert 0.0 <= score <= 1.0
