"""Polymarket prediction market signals.

Fetches active markets from the Polymarket Gamma API (no auth required)
and filters for macro/geopolitical relevance. Useful as forward-looking
crowd-sourced probability signals alongside traditional macro indicators.
"""
import json
from datetime import datetime, timezone

import requests
from src.common.logger import get_logger

log = get_logger("macro.polymarket")

GAMMA_API = "https://gamma-api.polymarket.com"

# Keywords to match against market questions (case-insensitive)
RELEVANT_KEYWORDS = [
    # Trade & tariffs
    "tariff", "trade war", "trade deal", "import", "export control",
    "sanction",
    # China / geopolitics
    "china", "taiwan", "huawei", "tsmc", "beijing",
    "russia", "ukraine", "north korea", "iran",
    # Tech / AI
    "semiconductor", "chip", "nvidia", "ai ", "artificial intelligence",
    "tech ban",
    # Macro
    "fed ", "federal reserve", "interest rate", "rate cut", "rate hike",
    "inflation", "recession", "gdp", "unemployment",
    "dollar", "yuan", "yen",
    # Energy
    "oil", "opec", "energy",
]

HEADERS = {"User-Agent": "GlobalMarketMonitor/1.0"}


def _yes_prob(market):
    """Extract YES probability from outcomePrices field."""
    try:
        prices = market.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        outcomes = market.get("outcomes", "[]")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        # Find YES outcome index
        for i, o in enumerate(outcomes):
            if str(o).strip().lower() in ("yes", "true"):
                return round(float(prices[i]), 4)
        # Fallback: first price
        return round(float(prices[0]), 4) if prices else None
    except Exception:
        return None


def _is_relevant(question):
    q = question.lower()
    return any(kw in q for kw in RELEVANT_KEYWORDS)


def fetch_polymarket_signals(limit=200, min_volume=1000):
    """Fetch relevant active prediction markets from Polymarket.

    Returns a list of dicts sorted by 24h volume descending:
        question, yes_prob, volume_24hr, end_date, url, moved (bool)
    """
    try:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={
                "limit": limit,
                "active": "true",
                "order": "volume24hr",
                "ascending": "false",
            },
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        markets = resp.json()
    except Exception as e:
        log.warning(f"Polymarket fetch failed: {e}")
        return []

    results = []
    for m in markets:
        q = m.get("question", "")
        if not q or not _is_relevant(q):
            continue

        vol = m.get("volume24hr") or m.get("volumeNum") or 0
        try:
            vol = float(vol)
        except Exception:
            vol = 0
        if vol < min_volume:
            continue

        yes_prob = _yes_prob(m)
        if yes_prob is None:
            continue

        end_date = m.get("endDateIso") or m.get("endDate", "")
        if end_date:
            try:
                end_date = end_date[:10]
            except Exception:
                pass

        slug = m.get("slug", "")
        url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"

        results.append({
            "question": q,
            "yes_prob": yes_prob,
            "volume_24hr": vol,
            "end_date": end_date,
            "url": url,
        })

    log.info(f"Polymarket: {len(results)} relevant markets from {len(markets)} fetched")
    return results


def fetch_and_save(out_path):
    """Fetch signals and save to JSON."""
    import json as _json
    signals = fetch_polymarket_signals()
    with open(out_path, "w") as f:
        _json.dump({
            "updated": datetime.now(timezone.utc).isoformat(),
            "markets": signals,
        }, f, indent=2)
    return signals
