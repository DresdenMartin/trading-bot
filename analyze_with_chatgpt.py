"""Analysis helpers: summary and portfolio scoring. Placeholder implementations (no OpenAI)."""
from typing import Any, Dict, List, Optional


def analyze_summary(symbol: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a short analysis dict: suggested_action, confidence, summary, rationale, premarket."""
    suggested = "hold"
    confidence = 0.5
    close = metrics.get("close") or 0.0
    rsi = metrics.get("rsi_14")
    news_sent = float(metrics.get("news_aggregate_sentiment") or 0.0)
    prem = metrics.get("premarket") or {}
    sector = prem.get("sector") if isinstance(prem, dict) else "UNKNOWN"

    if rsi is not None:
        try:
            r = float(rsi)
            if r < 30:
                suggested = "buy"
                confidence = 0.6
            elif r > 70:
                suggested = "sell"
                confidence = 0.6
        except (TypeError, ValueError):
            pass
    if news_sent > 0.2 and suggested == "hold":
        suggested = "buy"
        confidence = max(confidence, 0.55)
    elif news_sent < -0.2 and suggested == "hold":
        suggested = "sell"
        confidence = max(confidence, 0.55)

    summary = f"{symbol}: {suggested} (conf {int(confidence * 100)}%). Close ${close:.2f}."
    if rsi is not None:
        summary += f" RSI {rsi:.1f}."
    rationale = f"Technical and news placeholder; sector={sector}."

    return {
        "suggested_action": suggested,
        "confidence": confidence,
        "summary": summary,
        "rationale": rationale,
        "premarket": {"sector": sector},
        "signals": {},
        "recommended_size": None,
    }


def analyze_portfolio_short_term(items: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Optional LLM-based portfolio scores. Return None to use local scores only."""
    return None
