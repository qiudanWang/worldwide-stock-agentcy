"""Build global peer mapping and peer groups from company tags."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.peers.peer_matcher import build_peer_mapping, build_global_peer_groups

if __name__ == "__main__":
    # Build global peer mapping
    mapping = build_peer_mapping(top_k=5)

    if mapping.empty:
        print("No peer mapping generated")
    else:
        markets = mapping["market"].nunique()
        companies = mapping["ticker"].nunique()
        print(f"\nGlobal peer mapping: {companies} companies across {markets} markets\n")

        # Show summary by market
        for market in sorted(mapping["market"].unique()):
            mkt_df = mapping[mapping["market"] == market]
            count = mkt_df["ticker"].nunique()
            peer_markets = sorted(mkt_df["peer_market"].unique())
            print(f"  {market}: {count} companies -> peers in {', '.join(peer_markets)}")

    # Build peer groups
    groups = build_global_peer_groups(threshold=0.5)
    print(f"\nPeer groups: {len(groups)} groups\n")
    for g in groups[:10]:
        tickers = [f"[{c['market']}]{c['ticker']}" for c in g["companies"][:8]]
        suffix = f"... +{g['company_count'] - 8} more" if g["company_count"] > 8 else ""
        print(f"  Group {g['group_id']} ({g['theme']}): {', '.join(tickers)} {suffix}")

    # Legacy: show CN->US pairs
    cn_us = mapping[
        (mapping["market"] == "CN") & (mapping["peer_market"] == "US")
    ] if not mapping.empty else mapping
    if not cn_us.empty:
        print(f"\nLegacy CN->US pairs: {cn_us['ticker'].nunique()} CN companies\n")
        for ticker in cn_us["ticker"].unique():
            peers = cn_us[cn_us["ticker"] == ticker]
            name = peers.iloc[0]["name"]
            peer_str = ", ".join(
                f"{r['peer_name']}({r['peer_score']:.2f})"
                for _, r in peers.head(3).iterrows()
            )
            print(f"  {name} -> {peer_str}")
