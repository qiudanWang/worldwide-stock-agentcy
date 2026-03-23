"""World Monitor integration.

World Monitor is an open-source geopolitical intelligence platform.
The hosted API requires self-deployment, so we use its curated RSS feeds
directly for geopolitical/macro news. These are the same feeds World Monitor
aggregates from 435+ sources.

Current feed list: 23 sources across global news, tech/AI, finance/macro,
geopolitics, and Asia-focused outlets.

For full API access, self-host via Vercel:
  https://github.com/koala73/worldmonitor
"""
import xml.etree.ElementTree as ET
from datetime import datetime

import requests
import pandas as pd
from src.common.logger import get_logger

log = get_logger("news.worldmonitor")

# Curated RSS feeds — geopolitics, tech, Asia/trade focused
# Sourced from World Monitor's curated feed list (435+ sources)
FEEDS = {
    # Global news
    "BBC World":        "https://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC Tech":         "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "The Guardian":     "https://www.theguardian.com/world/rss",
    "Financial Times":  "https://www.ft.com/rss/home",
    # Tech & AI
    "CNBC Tech":        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
    "Ars Technica":     "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "TechCrunch":       "https://techcrunch.com/feed/",
    "The Verge":        "https://www.theverge.com/rss/index.xml",
    "VentureBeat AI":   "https://venturebeat.com/category/ai/feed/",
    "MIT Tech Review":  "https://www.technologyreview.com/feed/",
    # Finance & macro
    "Federal Reserve":  "https://www.federalreserve.gov/feeds/press_all.xml",
    # Geopolitics
    "Foreign Policy":   "https://foreignpolicy.com/feed/",
    "The Diplomat":     "https://thediplomat.com/feed/",
    "Crisis Group":     "https://www.crisisgroup.org/rss",
    "IAEA":             "https://www.iaea.org/feeds/topnews",
    "RAND":             "https://www.rand.org/pubs/articles.xml",
    # Asia focus
    "SCMP":             "https://www.scmp.com/rss/91/feed",
    "Nikkei Asia":      "https://asia.nikkei.com/rss/feed/nar",
    "CNA Singapore":    "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml",
    "Japan Today":      "https://japantoday.com/feed/atom",
    # Topic-specific Google News searches
    "GNews Semis":      "https://news.google.com/rss/search?q=semiconductor+chip+export+control+China+when:3d&hl=en-US&gl=US&ceid=US:en",
    "GNews Trade":      "https://news.google.com/rss/search?q=trade+war+tariff+tech+when:3d&hl=en-US&gl=US&ceid=US:en",
    "GNews CN Tech":    "https://news.google.com/rss/search?q=China+tech+AI+Huawei+TSMC+when:2d&hl=en-US&gl=US&ceid=US:en",
}

HEADERS = {
    "User-Agent": "GlobalTechMonitor/1.0",
}


def fetch_rss_feed(name, url):
    """Fetch and parse a single RSS feed."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        items = []

        for item in root.iter("item"):
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date = item.findtext("pubDate", "")
            description = item.findtext("description", "")

            items.append({
                "source": name,
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "description": description[:200] if description else "",
            })

        return items
    except Exception as e:
        log.warning(f"Failed to fetch {name}: {e}")
        return []


def fetch_geopolitical_news(keyword_filter=None):
    """Fetch news from World Monitor's curated RSS sources.

    Args:
        keyword_filter: optional list of keywords to filter by (case-insensitive)

    Returns:
        DataFrame with geopolitical news items
    """
    all_items = []
    for name, url in FEEDS.items():
        items = fetch_rss_feed(name, url)
        all_items.extend(items)
        if items:
            log.info(f"  {name}: {len(items)} items")

    if not all_items:
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    df["date"] = datetime.now().strftime("%Y-%m-%d")
    log.info(f"Fetched {len(df)} geopolitical news items from {len(FEEDS)} sources")

    if keyword_filter:
        keywords_lower = [k.lower() for k in keyword_filter]

        def has_keyword(row):
            text = f"{row['title']} {row['description']}".lower()
            return any(k in text for k in keywords_lower)

        df["relevant"] = df.apply(has_keyword, axis=1)
        relevant = df[df["relevant"]]
        log.info(f"  {len(relevant)} items matched keyword filter")
    else:
        df["relevant"] = True

    return df


def fetch_geopolitical_context():
    """Fetch geopolitical context — filtered news from curated sources.

    Returns a dict with:
        news: list of relevant geopolitical news
        feed_count: number of feeds fetched
        total_items: total items fetched
        relevant_items: items matching tech/China keywords
    """
    tech_keywords = [
        "AI", "chip", "semiconductor", "export control", "China tech",
        "regulation", "tariff", "trade war", "Nvidia", "TSMC",
        "sanctions", "technology", "data center", "GPU",
        "Huawei", "DeepSeek", "SMIC", "geopolit", "supply chain",
        "interest rate", "Federal Reserve", "inflation", "CPI",
        "quantum", "biotech", "EV", "electric vehicle", "battery",
        "Taiwan", "South Korea", "Japan tech", "India tech",
    ]

    df = fetch_geopolitical_news(keyword_filter=tech_keywords)
    if df.empty:
        return {}

    relevant = df[df["relevant"]].copy()

    # Sort by pub_date descending (most recent first)
    try:
        from email.utils import parsedate_to_datetime
        relevant["_ts"] = relevant["pub_date"].apply(
            lambda d: parsedate_to_datetime(d).timestamp() if d else 0
        )
        relevant = relevant.sort_values("_ts", ascending=False).drop(columns=["_ts"])
    except Exception:
        pass  # keep original order if parsing fails

    # Deduplicate: keep at most 3 items per source so no single feed dominates
    deduped = (
        relevant.groupby("source", group_keys=False)
        .head(3)
        .reset_index(drop=True)
    )
    # Re-sort after dedupe so items are still chronological across sources
    try:
        deduped["_ts"] = deduped["pub_date"].apply(
            lambda d: parsedate_to_datetime(d).timestamp() if d else 0
        )
        deduped = deduped.sort_values("_ts", ascending=False).drop(columns=["_ts"])
    except Exception:
        pass

    context = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "feed_count": len(FEEDS),
        "total_items": len(df),
        "relevant_items": len(relevant),
        "news": deduped[["source", "title", "link", "pub_date"]].to_dict("records"),
    }

    return context
