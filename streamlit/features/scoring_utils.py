from sklearn.preprocessing import minmax_scale

def get_match_tier(score: float) -> str:
    if score >= 0.75:
        return "High"
    elif score >= 0.50:
        return "Medium"
    else:
        return "Low"

def tier_badge_html(tier: str) -> str:
    if tier == "High":
        cls = "tier-high"
    elif tier == "Medium":
        cls = "tier-medium"
    else:
        cls = "tier-low"
    return f"<span class='{cls}'>{tier} tier</span>"

def compute_confidences(scores):
    if len(scores) <= 1:
        return [0.5 for _ in scores]
    arr = [float(s) for s in scores]
    if max(arr) == min(arr):
        return [0.5 for _ in scores]
    scaled = minmax_scale(arr)
    return scaled.tolist()
