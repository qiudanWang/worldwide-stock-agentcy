"""Sector performance analysis across all markets."""

import pandas as pd
from src.common.config import get_data_path
from src.common.logger import get_logger

log = get_logger("analysis.sector")

# Map non-English and fragmented sector names → unified English names
_SECTOR_MAP = {
    # CN (Simplified Chinese) — akshare Eastmoney board names
    "IT服务Ⅲ": "IT Services", "IT服务": "IT Services",
    "半导体": "Semiconductors", "半导体及半导体生产设备": "Semiconductors",
    "计算机设备": "Hardware & Equipment", "计算机": "Hardware & Equipment",
    "软件开发": "Software", "软件": "Software",
    "通信设备": "Telecom Equipment", "通信": "Telecommunications",
    "互联网": "Internet Platforms", "互联网平台": "Internet Platforms",
    "互联网服务": "Internet Platforms",
    "电子": "Electronic Components", "电子元器件": "Electronic Components",
    "消费电子": "Consumer Electronics",
    "光学光电子": "Optoelectronics",
    "电子化学品": "Electronic Components",
    "游戏": "Gaming",
    "电池": "EV & Clean Energy",
    "储能": "EV & Clean Energy",
    "光伏设备": "EV & Clean Energy",
    "自动化设备": "Robotics & Automation",
    "新能源": "EV & Clean Energy",
    "人工智能": "AI & Infrastructure",
    "云计算": "Cloud & SaaS",
    "大数据": "Data & Analytics",
    "网络安全": "Cybersecurity",
    "医疗器械": "Medical Devices",
    "医药商业": "Pharmaceuticals",
    "医疗服务": "Healthcare",
    "银行": "Financial Services",
    "证券": "Financial Services",
    "保险": "Financial Services",
    # TW (Traditional Chinese)
    "半導體業": "Semiconductors",
    "其他電子業": "Electronic Components",
    "電子零組件業": "Electronic Components",
    "光電業": "Optoelectronics",
    "電腦及週邊設備業": "Hardware & Equipment",
    "資訊服務業": "IT Services",
    "通信網路業": "Networking & Telecom",
    "電子通路業": "Electronics Distribution",
    # KR (Korean) — tech/core
    "반도체 제조업": "Semiconductors",
    "소프트웨어 개발 및 공급업": "Software",
    "컴퓨터 및 주변장치 제조업": "Hardware & Equipment",
    "전자부품 제조업": "Electronic Components",
    "자료처리, 호스팅, 포털 및 기타 인터넷 정보매개 서비스업": "Internet Services",
    "컴퓨터 프로그래밍, 시스템 통합 및 관리업": "IT Services",
    "전기 통신업": "Telecommunications",
    "통신 및 방송 장비 제조업": "Telecom Equipment",
    "영상 및 음향기기 제조업": "Consumer Electronics",
    "의약품 제조업": "Pharmaceuticals",
    "일반 목적용 기계 제조업": "Industrial Machinery",
    "특수 목적용 기계 제조업": "Industrial Machinery",
    "자동차 신품 부품 제조업": "Auto Parts",
    "일차전지 및 이차전지 제조업": "Batteries & Energy Storage",
    "기초 화학물질 제조업": "Chemicals",
    "기타 화학제품 제조업": "Chemicals",
    "의료용 기기 제조업": "Medical Devices",
    "기초 의약물질 제조업": "Pharmaceuticals",
    "기타 정보 서비스업": "IT Services",
    "자연과학 및 공학 연구개발업": "R&D Services",
    "전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업": "Electrical Equipment",
    "기타 전기장비 제조업": "Electrical Equipment",
    "선박 및 보트 건조업": "Shipbuilding",
    # KR (Korean) — other sectors
    "가전제품 및 정보통신장비 소매업": "Electronics Retail",
    "가정용 기기 제조업": "Home Appliances",
    "가죽, 가방 및 유사제품 제조업": "Textiles & Apparel",
    "개인 및 가정용품 수리업": "Consumer Services",
    "건물설비 설치 공사업": "Construction",
    "건축기술, 엔지니어링 및 관련 기술 서비스업": "Engineering Services",
    "경비, 경호 및 탐정업": "Security Services",
    "곡물가공품, 전분 및 전분제품 제조업": "Food Processing",
    "그외 기타 전문, 과학 및 기술 서비스업": "Professional Services",
    "그외 기타 제품 제조업": "Diversified Manufacturing",
    "금융 지원 서비스업": "Financial Services",
    "기계장비 및 관련 물품 도매업": "Industrial Wholesale",
    "기타 과학기술 서비스업": "Scientific Services",
    "기타 금속 가공제품 제조업": "Metal Fabrication",
    "기타 금융업": "Financial Services",
    "기타 비금속 광물제품 제조업": "Materials",
    "기타 식품 제조업": "Food & Beverages",
    "기타 운송관련 서비스업": "Transportation",
    "기타 전문 도매업": "Wholesale Trade",
    "무점포 소매업": "E-commerce & Retail",
    "부동산 관련 서비스업": "Real Estate",
    "부동산 임대 및 공급업": "Real Estate",
    "사업시설 유지·관리 서비스업": "Facility Management",
    "상품 종합 도매업": "Wholesale Trade",
    "상품 중개업": "Wholesale Trade",
    "생활용품 도매업": "Consumer Wholesale",
    "서적, 잡지 및 기타 인쇄물 출판업": "Media & Publishing",
    "수산물 가공 및 저장 처리업": "Food Processing",
    "신탁업 및 집합투자업": "Asset Management",
    "실내건축 및 건축마무리 공사업": "Construction",
    "악기 제조업": "Consumer Products",
    "여행사 및 기타 여행보조 서비스업": "Travel & Leisure",
    "연료 소매업": "Energy",
    "영화, 비디오물, 방송프로그램 제작 및 배급업": "Media & Entertainment",
    "유리 및 유리제품 제조업": "Materials",
    "의료용품 및 기타 의약 관련제품 제조업": "Healthcare",
    "자동차 부품 및 내장품 판매업": "Auto Parts",
    "전기 및 통신 공사업": "Infrastructure",
    "절연선 및 케이블 제조업": "Electrical Equipment",
    "종합 소매업": "Retail",
    "직물직조 및 직물제품 제조업": "Textiles & Apparel",
    "초등 교육기관": "Education",
    "측정, 시험, 항해, 제어 및 기타 정밀기기 제조업; 광학기기 제외": "Precision Instruments",
    "텔레비전 방송업": "Media & Entertainment",
    "토목 건설업": "Construction",
    "편조의복 제조업": "Textiles & Apparel",
    "플라스틱제품 제조업": "Materials",
    "회사 본부 및 경영 컨설팅 서비스업": "Business Services",
    # HK custom tags
    "cloud_saas": "Cloud & SaaS",
    "ev_new_energy_tech": "EV & Clean Energy",
    "fintech_insurtech": "Fintech",
    "gaming": "Gaming",
    "hardware_electronics": "Hardware & Equipment",
    "internet_platform": "Internet Platforms",
    "semiconductor": "Semiconductors",
    "telecom_equipment": "Telecom Equipment",
    # AU custom tags
    "data_centres_and_infrastructure": "Data Centers & Infrastructure",
    "fintech_and_payments": "Fintech",
    "hardware_and_semiconductor": "Hardware & Semiconductors",
    "healthtech": "Healthtech",
    "software": "Software",
    # US custom tags
    "AI_infrastructure": "AI & Infrastructure",
    "EV_tech": "EV & Clean Energy",
    "IT_services": "IT Services",
    "adtech_martech": "Ad Tech & Marketing",
    "cleantech_greentech": "Clean Technology",
    "cloud_infrastructure": "Cloud Infrastructure",
    "crypto_blockchain": "Crypto & Blockchain",
    "cybersecurity": "Cybersecurity",
    "data_analytics": "Data & Analytics",
    "edtech_healthtech": "Healthtech",
    "fintech": "Fintech",
    "hardware_devices": "Hardware & Devices",
    "internet_platforms": "Internet Platforms",
    "networking_telecom": "Networking & Telecom",
    "quantum_computing": "Quantum Computing",
    "robotics_automation": "Robotics & Automation",
    "space_defense_tech": "Space & Defense",
    "telehealth_biotech_IT": "Healthtech & Biotech",
    # SA / other
    "Information Technology": "Technology",
    # Watchlist group names — DE
    "semiconductor_and_hardware": "Semiconductors",
    "internet_and_telecom": "Networking & Telecom",
    "solar_tech": "Clean Technology",
    # Watchlist group names — FR
    "software_and_services": "Software",
    "cloud_and_telecom": "Cloud & SaaS",
    # Watchlist group names — IN
    "it_services": "IT Services",
    "engineering_and_rd": "R&D Services",
    "product_and_platform": "Software",
    "internet_and_digital": "Internet Platforms",
    # Watchlist group names — JP
    "semiconductor_equipment": "Semiconductors",
    "electronic_components": "Electronic Components",
    "software_it": "Software",
    "hardware": "Hardware & Equipment",
    "ev_battery_lithography": "EV & Clean Energy",
    "telecom_carriers": "Telecommunications",
    "medical_tech": "Medical Devices",
    # Watchlist group names — KR (additional)
    "electronics_and_display": "Electronic Components",
    "battery_and_ev_tech": "EV & Clean Energy",
    "entertainment_tech": "Gaming",
    # Watchlist group names — TW
    "pcb_and_substrate": "Electronic Components",
    "odm_and_systems": "Hardware & Equipment",
    "networking_and_comms": "Networking & Telecom",
    "components_and_precision": "Electronic Components",
    "storage_and_embedded": "Hardware & Devices",
    "optoelectronics": "Optoelectronics",
    # Watchlist group names — UK
    "data_and_analytics": "Data & Analytics",
    "industrial_technology": "Industrial Machinery",
    # Watchlist group names — BR
    "mobile_and_digital": "Internet Platforms",
    "hardware_and_electronics": "Hardware & Equipment",
    # Watchlist group names — SA
    "telecom": "Telecommunications",
    "technology": "Technology",
    # Generic watchlist group names (lowercase with underscores)
    "ev_and_energy": "EV & Clean Energy",
    "healthcare": "Healthcare",
    "internet_and_media": "Internet Platforms",
    "cloud_and_ai": "Cloud & SaaS",
    "semiconductors": "Semiconductors",
    "electronic_components": "Electronic Components",
    "healthtech": "Healthtech",
    "hardware": "Hardware & Equipment",
    "ai_infrastructure": "AI & Infrastructure",
    "telecom_equipment": "Telecom Equipment",
}


def _normalize_sector(name):
    """Return mapped English name, or original if not in map."""
    if pd.isna(name):
        return name
    return _SECTOR_MAP.get(str(name).strip(), str(name).strip())


def normalize_sector_df(df):
    """Re-normalize and re-aggregate an already-computed sector_performance DataFrame.

    Use this at web read time so stale parquet data gets cleaned up without
    needing a full pipeline re-run.
    """
    if df.empty:
        return df

    df = df.copy()
    df["sector"] = df["sector"].map(_normalize_sector)

    # Drop rows with NaN return or NaN sector
    df = df[df["sector"].notna() & df["avg_return_1d"].notna()]

    # Drop rows with obviously corrupted aggregate returns (>±200%)
    df = df[df["avg_return_1d"].abs() <= 2.0]

    # Re-aggregate duplicate (market, sector) pairs that merged after normalization
    ret_cols = [c for c in ["avg_return_1d", "avg_return_5d", "avg_return_20d"] if c in df.columns]
    stock_col = "stock_count" if "stock_count" in df.columns else None

    agg_parts = []
    for (market, sector), grp in df.groupby(["market", "sector"]):
        row = {"market": market, "sector": sector}
        if stock_col:
            total = grp[stock_col].sum()
            row["stock_count"] = total
            for col in ret_cols:
                row[col] = round((grp[col] * grp[stock_col]).sum() / total, 4) if total else None
        else:
            for col in ret_cols:
                row[col] = round(grp[col].mean(), 4)
        # keep top/bottom tickers from the dominant sub-row
        dominant = grp.loc[grp[stock_col].idxmax()] if stock_col else grp.iloc[0]
        for extra in ["top_ticker", "top_return_1d", "bottom_ticker", "bottom_return_1d"]:
            if extra in grp.columns:
                row[extra] = dominant.get(extra)
        agg_parts.append(row)

    result = pd.DataFrame(agg_parts)

    # Drop singletons
    if stock_col and stock_col in result.columns:
        result = result[result[stock_col] >= 2]

    return result.reset_index(drop=True)


# Maps the 60+ normalized English sector names → ~12 broad meta-sectors for
# the dashboard heatmap so that market-specific taxonomies collapse into
# comparable cross-market rows.
_META_SECTOR_MAP = {
    # Technology (generic yfinance tag used by EU/APAC/EM markets)
    "Technology":                  "Technology",
    # Semiconductors + components
    "Semiconductors":              "Semiconductors",
    "Electronic Components":       "Semiconductors",
    "Optoelectronics":             "Semiconductors",
    "Electronics Distribution":    "Semiconductors",
    "Hardware & Semiconductors":   "Semiconductors",
    # Software & IT
    "Software":                    "Software & IT",
    "IT Services":                 "Software & IT",
    "Internet Services":           "Software & IT",
    "R&D Services":                "Software & IT",
    "Cybersecurity":               "Software & IT",
    "Data & Analytics":            "Software & IT",
    # Internet & Media
    "Internet Platforms":          "Internet & Media",
    "Gaming":                      "Internet & Media",
    "Media & Entertainment":       "Internet & Media",
    "Media & Publishing":          "Internet & Media",
    "Ad Tech & Marketing":         "Internet & Media",
    # Cloud & AI
    "AI & Infrastructure":         "Cloud & AI",
    "Cloud & SaaS":                "Cloud & AI",
    "Cloud Infrastructure":        "Cloud & AI",
    "Data Centers & Infrastructure": "Cloud & AI",
    "Quantum Computing":           "Cloud & AI",
    "Robotics & Automation":       "Cloud & AI",
    # Hardware & Devices
    "Hardware & Equipment":        "Hardware",
    "Hardware & Devices":          "Hardware",
    "Consumer Electronics":        "Hardware",
    "Home Appliances":             "Hardware",
    "Precision Instruments":       "Hardware",
    # Telecom & Networks
    "Telecommunications":          "Telecom",
    "Telecom Equipment":           "Telecom",
    "Networking & Telecom":        "Telecom",
    "Communication Services":      "Telecom",
    # Fintech & Finance
    "Fintech":                     "Fintech & Finance",
    "Financial Services":          "Fintech & Finance",
    "Crypto & Blockchain":         "Fintech & Finance",
    # Healthcare & Biotech
    "Healthcare":                  "Healthcare",
    "Healthtech":                  "Healthcare",
    "Healthtech & Biotech":        "Healthcare",
    "Medical Devices":             "Healthcare",
    "Pharmaceuticals":             "Healthcare",
    "Scientific Services":         "Healthcare",
    # EV & Clean Energy
    "EV & Clean Energy":           "EV & Energy",
    "Batteries & Energy Storage":  "EV & Energy",
    "Clean Technology":            "EV & Energy",
    # Industrials
    "Industrial Machinery":        "Industrials",
    "Electrical Equipment":        "Industrials",
    "Construction":                "Industrials",
    "Engineering Services":        "Industrials",
    "Infrastructure":              "Industrials",
    "Auto Parts":                  "Industrials",
    "Shipbuilding":                "Industrials",
    "Industrial Wholesale":        "Industrials",
    "Wholesale Trade":             "Industrials",
    "Diversified Manufacturing":   "Industrials",
    "Materials":                   "Industrials",
    "Chemicals":                   "Industrials",
    # Consumer
    "Retail":                      "Consumer",
    "Textiles & Apparel":          "Consumer",
    "Food Processing":             "Consumer",
    "Consumer Wholesale":          "Consumer",
    "Electronics Retail":          "Consumer",
    "Travel & Leisure":            "Consumer",
    # Defense & Space
    "Space & Defense":             "Defense & Space",
    # Business Services
    "Business Services":           "Business Services",
    "Professional Services":       "Business Services",
    "Real Estate":                 "Business Services",
    "Security Services":           "Business Services",
    "Asset Management":            "Business Services",
    "Facility Management":         "Business Services",
}


def meta_sector_heatmap(sector_df):
    """Collapse normalized sector_performance into broad meta-sectors.

    Returns a re-aggregated DataFrame with the same schema, using
    stock-count-weighted returns. Suitable for the dashboard heatmap.
    """
    if sector_df.empty:
        return sector_df

    df = sector_df.copy()
    df["sector"] = df["sector"].map(lambda s: _META_SECTOR_MAP.get(s, s))

    ret_cols = [c for c in ["avg_return_1d", "avg_return_5d", "avg_return_20d"] if c in df.columns]
    stock_col = "stock_count" if "stock_count" in df.columns else None

    agg_parts = []
    for (market, sector), grp in df.groupby(["market", "sector"]):
        row = {"market": market, "sector": sector}
        if stock_col:
            total = grp[stock_col].sum()
            row["stock_count"] = total
            for col in ret_cols:
                row[col] = round((grp[col] * grp[stock_col]).sum() / total, 4) if total else None
        else:
            for col in ret_cols:
                row[col] = round(grp[col].mean(), 4)
        agg_parts.append(row)

    return pd.DataFrame(agg_parts).reset_index(drop=True)


def compute_sector_performance(all_market_data):
    """Compute per-sector average return grouped by market.

    Args:
        all_market_data: Combined DataFrame with columns:
            ticker, market, sector, return_1d, return_5d, return_20d

    Returns:
        DataFrame with: market, sector, avg_return_1d, avg_return_5d, avg_return_20d,
                        stock_count, top_ticker, bottom_ticker
    """
    if all_market_data.empty or "sector" not in all_market_data.columns:
        return pd.DataFrame()

    # Filter out rows without sector
    df = all_market_data[all_market_data["sector"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Drop corrupted price data: single-day returns beyond ±50% are almost
    # certainly yfinance data errors (e.g. a stock price jumping to 1e11).
    if "return_1d" in df.columns:
        before = len(df)
        df = df[df["return_1d"].abs() <= 0.50]
        dropped = before - len(df)
        if dropped:
            log.warning(f"Dropped {dropped} rows with |return_1d| > 50% (corrupted price data)")

    # Normalize sector names to English
    df["sector"] = df["sector"].map(_normalize_sector)

    # Group by market + sector
    grouped = df.groupby(["market", "sector"]).agg(
        avg_return_1d=("return_1d", "mean"),
        avg_return_5d=("return_5d", "mean"),
        avg_return_20d=("return_20d", "mean"),
        stock_count=("ticker", "count"),
    ).reset_index()

    # Find top and bottom tickers per sector
    top_tickers = []
    bottom_tickers = []
    for (market, sector), group in df.groupby(["market", "sector"]):
        if not group.empty and "return_1d" in group.columns:
            valid = group.dropna(subset=["return_1d"])
            if not valid.empty:
                top = valid.loc[valid["return_1d"].idxmax()]
                bottom = valid.loc[valid["return_1d"].idxmin()]
                top_tickers.append({
                    "market": market, "sector": sector,
                    "top_ticker": top["ticker"],
                    "top_return_1d": round(top["return_1d"], 4),
                })
                bottom_tickers.append({
                    "market": market, "sector": sector,
                    "bottom_ticker": bottom["ticker"],
                    "bottom_return_1d": round(bottom["return_1d"], 4),
                })

    if top_tickers:
        top_df = pd.DataFrame(top_tickers)
        grouped = grouped.merge(top_df, on=["market", "sector"], how="left")
    if bottom_tickers:
        bot_df = pd.DataFrame(bottom_tickers)
        grouped = grouped.merge(bot_df, on=["market", "sector"], how="left")

    # Round
    for col in ["avg_return_1d", "avg_return_5d", "avg_return_20d"]:
        grouped[col] = grouped[col].round(4)

    # Drop rows with no valid return data (NaN avg_return_1d)
    grouped = grouped[grouped["avg_return_1d"].notna()]

    # Drop singleton sectors (only 1 stock — too noisy)
    grouped = grouped[grouped["stock_count"] >= 2]

    log.info(f"Sector performance: {len(grouped)} market-sector combinations")
    return grouped


def detect_sector_rotation(current_perf, previous_perf):
    """Detect sectors that have changed rank significantly.

    Args:
        current_perf: Current sector performance (5d returns).
        previous_perf: Previous sector performance (20d returns).

    Returns:
        List of dicts describing sector rotation events.
    """
    if current_perf.empty or previous_perf.empty:
        return []

    rotations = []

    for market in current_perf["market"].unique():
        curr = current_perf[current_perf["market"] == market].copy()
        prev = previous_perf[previous_perf["market"] == market].copy()

        if curr.empty or prev.empty:
            continue

        curr_ranked = curr.sort_values("avg_return_5d", ascending=False).reset_index(drop=True)
        prev_ranked = prev.sort_values("avg_return_20d", ascending=False).reset_index(drop=True)

        curr_ranked["rank_5d"] = range(1, len(curr_ranked) + 1)
        prev_ranked["rank_20d"] = range(1, len(prev_ranked) + 1)

        merged = curr_ranked[["sector", "rank_5d", "avg_return_5d"]].merge(
            prev_ranked[["sector", "rank_20d", "avg_return_20d"]],
            on="sector", how="inner"
        )

        for _, row in merged.iterrows():
            rank_change = row["rank_20d"] - row["rank_5d"]
            if abs(rank_change) >= 3:
                direction = "rising" if rank_change > 0 else "falling"
                rotations.append({
                    "market": market,
                    "sector": row["sector"],
                    "direction": direction,
                    "rank_5d": int(row["rank_5d"]),
                    "rank_20d": int(row["rank_20d"]),
                    "rank_change": int(rank_change),
                    "return_5d": round(row["avg_return_5d"], 4),
                    "return_20d": round(row["avg_return_20d"], 4),
                })

    log.info(f"Sector rotation: {len(rotations)} events detected")
    return rotations
