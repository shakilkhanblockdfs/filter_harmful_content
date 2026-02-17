# policy.py

def decide_action(risk_score: float):
    """
    Simple policy:
    > 0.95  -> BLOCK
    > 0.70  -> LIMIT + REVIEW
    > 0.40  -> WARN
    else    -> ALLOW
    """
    if risk_score > 0.95:
        return "BLOCK"
    elif risk_score > 0.70:
        return "LIMIT_AND_REVIEW"
    elif risk_score > 0.40:
        return "WARN"
    else:
        return "ALLOW"
