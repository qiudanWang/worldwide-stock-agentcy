"""Global Peer Matching System.

Matches ALL companies against ALL others across all markets based on
tag similarity (sub_sector, customer_type, revenue_model).
"""
import json
from collections import defaultdict
from itertools import combinations

import pandas as pd
from src.common.config import load_yaml, get_data_path
from src.common.logger import get_logger

log = get_logger("peers.matcher")


def load_company_tags():
    """Load company tags from config."""
    return load_yaml("company_tags.yaml")


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_tag_similarity(tags_a, tags_b):
    """Compute similarity between two companies based on tags.

    Returns a score between 0 and 1.
    Weights: sub_sector=0.6, customer_type=0.2, revenue_model=0.2
    """
    sub_sector_sim = jaccard_similarity(
        set(tags_a.get("sub_sector", [])),
        set(tags_b.get("sub_sector", []))
    )
    customer_sim = jaccard_similarity(
        set(tags_a.get("customer_type", [])),
        set(tags_b.get("customer_type", []))
    )
    revenue_sim = jaccard_similarity(
        set(tags_a.get("revenue_model", [])),
        set(tags_b.get("revenue_model", []))
    )

    score = 0.6 * sub_sector_sim + 0.2 * customer_sim + 0.2 * revenue_sim
    return round(score, 4)


def _build_company_list(tags):
    """Convert tags dict to a list of (name, ticker, market, tags) tuples."""
    companies = []
    for name, info in tags.items():
        ticker = str(info.get("ticker", name))
        market = info.get("market", _infer_market(ticker))
        companies.append((name, ticker, market, info))
    return companies


def _infer_market(ticker):
    """Infer market from ticker format (legacy fallback)."""
    if isinstance(ticker, str) and ticker.isdigit():
        return "CN"
    return "US"


def build_peer_mapping(top_k=5):
    """Build global peer mapping — for each company, find top-K peers across ALL markets.

    Returns DataFrame with columns:
        ticker, market, name, peer_ticker, peer_market, peer_name, peer_score, rank
    Saves to data/global/peer_mapping.parquet.
    """
    tags = load_company_tags()
    if not tags:
        log.error("No company tags found")
        return pd.DataFrame()

    companies = _build_company_list(tags)
    log.info(f"Building global peer mapping for {len(companies)} companies")

    rows = []
    for i, (name_a, ticker_a, market_a, tags_a) in enumerate(companies):
        scores = []
        for j, (name_b, ticker_b, market_b, tags_b) in enumerate(companies):
            if i == j:
                continue
            # Skip same-market pairs (peers should be cross-market)
            if market_a == market_b:
                continue
            score = compute_tag_similarity(tags_a, tags_b)
            if score > 0:
                scores.append((name_b, ticker_b, market_b, score))

        # Sort by score descending, take top_k
        scores.sort(key=lambda x: x[3], reverse=True)
        for rank, (peer_name, peer_ticker, peer_market, score) in enumerate(scores[:top_k], 1):
            rows.append({
                "ticker": ticker_a,
                "market": market_a,
                "name": name_a,
                "peer_ticker": peer_ticker,
                "peer_market": peer_market,
                "peer_name": peer_name,
                "peer_score": score,
                "rank": rank,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        path = get_data_path("global", "peer_mapping.parquet")
        result.to_parquet(path, index=False)
        log.info(f"Saved global peer mapping: {len(result)} pairs to {path}")

        # Also save legacy format for backward compatibility
        legacy_path = get_data_path("processed", "peer_mapping.parquet")
        cn_us = result[
            ((result["market"] == "CN") & (result["peer_market"] == "US")) |
            ((result["market"] == "HK") & (result["peer_market"] == "US"))
        ].copy()
        if not cn_us.empty:
            legacy = cn_us.rename(columns={
                "name": "cn_name", "ticker": "cn_ticker",
                "peer_name": "us_name", "peer_ticker": "us_ticker",
            })[["cn_name", "cn_ticker", "us_name", "us_ticker", "peer_score", "rank"]]
            legacy.to_parquet(legacy_path, index=False)
            log.info(f"Saved legacy CN->US mapping: {len(legacy)} pairs")

    return result


def build_global_peer_groups():
    """Build peer groups — one group per sub_sector that spans 2+ markets.

    Each company can appear in multiple groups (once per sub_sector tag).
    Groups are sorted by company count descending.

    Returns list of dicts:
        [{ "group_id": int, "theme": str, "companies": [...], "pairs": [...] }, ...]
    Saves to data/global/peer_groups.json.
    """
    tags = load_company_tags()
    if not tags:
        log.error("No company tags found")
        return []

    companies = _build_company_list(tags)

    # Index companies by sub_sector
    by_sub_sector = defaultdict(list)
    for name, ticker, market, info in companies:
        for ss in info.get("sub_sector", []):
            by_sub_sector[ss].append({
                "name": name,
                "ticker": ticker,
                "market": market,
                "sub_sector": info.get("sub_sector", []),
            })

    groups = []
    group_id = 0
    for sub_sector, members in sorted(by_sub_sector.items(),
                                      key=lambda x: -len(x[1])):
        markets_in_group = set(c["market"] for c in members)
        # Only include groups that span 2+ markets
        if len(members) < 2 or len(markets_in_group) < 2:
            continue

        # Compute top pairwise scores (cross-market only)
        pairs = []
        for i, j in combinations(range(len(members)), 2):
            a, b = members[i], members[j]
            if a["market"] == b["market"]:
                continue
            score = compute_tag_similarity(
                {"sub_sector": a["sub_sector"]},
                {"sub_sector": b["sub_sector"]},
            )
            if score > 0:
                pairs.append({
                    "ticker_a": a["ticker"], "market_a": a["market"],
                    "ticker_b": b["ticker"], "market_b": b["market"],
                    "score": score,
                })

        _mo = ["CN", "US", "HK", "JP", "IN", "UK", "DE", "FR",
               "KR", "TW", "AU", "BR", "SA"]
        _mk = lambda m: _mo.index(m) if m in _mo else 99
        groups.append({
            "group_id": group_id,
            "theme": sub_sector,
            "company_count": len(members),
            "market_count": len(markets_in_group),
            "markets": sorted(markets_in_group, key=_mk),
            "companies": sorted(members, key=lambda c: _mk(c["market"])),
            "pairs": sorted(pairs, key=lambda p: -p["score"])[:20],
        })
        group_id += 1

    # Save
    path = get_data_path("global", "peer_groups.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(groups)} peer groups to {path}")

    return groups


def get_peers_for(ticker_or_name, peer_mapping_df=None):
    """Look up peers for a given company (from any market).

    Returns DataFrame of peers sorted by rank.
    """
    if peer_mapping_df is None:
        # Try global first, fall back to legacy
        try:
            path = get_data_path("global", "peer_mapping.parquet")
            peer_mapping_df = pd.read_parquet(path)
        except (FileNotFoundError, Exception):
            try:
                path = get_data_path("processed", "peer_mapping.parquet")
                peer_mapping_df = pd.read_parquet(path)
                # Legacy format — convert on the fly
                if "cn_ticker" in peer_mapping_df.columns:
                    return _legacy_get_peers(ticker_or_name, peer_mapping_df)
            except (FileNotFoundError, Exception):
                return pd.DataFrame()

    mask = (
        (peer_mapping_df["ticker"] == ticker_or_name) |
        (peer_mapping_df["name"] == ticker_or_name)
    )
    return peer_mapping_df[mask].sort_values("rank")


def _legacy_get_peers(ticker_or_name, legacy_df):
    """Handle legacy CN->US peer mapping format."""
    mask = (
        (legacy_df["cn_name"] == ticker_or_name) |
        (legacy_df["cn_ticker"] == ticker_or_name) |
        (legacy_df["us_name"] == ticker_or_name) |
        (legacy_df["us_ticker"] == ticker_or_name)
    )
    return legacy_df[mask].sort_values("rank" if "rank" in legacy_df.columns else "peer_score")


def get_cn_us_peers(peer_mapping_df=None):
    """Backward-compatible: get CN->US peer pairs only.

    Returns DataFrame in legacy format with cn_name, cn_ticker, us_name, us_ticker, peer_score, rank.
    """
    if peer_mapping_df is None:
        try:
            path = get_data_path("global", "peer_mapping.parquet")
            peer_mapping_df = pd.read_parquet(path)
        except (FileNotFoundError, Exception):
            try:
                path = get_data_path("processed", "peer_mapping.parquet")
                return pd.read_parquet(path)
            except (FileNotFoundError, Exception):
                return pd.DataFrame()

    if "cn_ticker" in peer_mapping_df.columns:
        return peer_mapping_df  # already legacy format

    cn_us = peer_mapping_df[
        (peer_mapping_df["market"] == "CN") &
        (peer_mapping_df["peer_market"] == "US")
    ].copy()

    if cn_us.empty:
        return pd.DataFrame()

    return cn_us.rename(columns={
        "name": "cn_name", "ticker": "cn_ticker",
        "peer_name": "us_name", "peer_ticker": "us_ticker",
    })[["cn_name", "cn_ticker", "us_name", "us_ticker", "peer_score", "rank"]]


# Legacy aliases
def get_cn_for_us(us_name_or_ticker, peer_mapping_df=None):
    """Look up which CN companies have a given US stock as a peer (legacy compat)."""
    cn_us = get_cn_us_peers(peer_mapping_df)
    if cn_us.empty:
        return pd.DataFrame()
    mask = (
        (cn_us["us_name"] == us_name_or_ticker) |
        (cn_us["us_ticker"] == us_name_or_ticker)
    )
    return cn_us[mask].sort_values("peer_score", ascending=False)
