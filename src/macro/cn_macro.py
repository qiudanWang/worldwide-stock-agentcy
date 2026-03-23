"""China macro data via akshare."""

import akshare as ak
import pandas as pd
from src.common.logger import get_logger

log = get_logger("macro.cn")


def _parse_month_col(series):
    """Parse akshare 月份 column like '2026年02月份' into datetime."""
    return pd.to_datetime(
        series.str.replace("年", "-", regex=False)
               .str.replace("月份", "-01", regex=False),
        format="%Y-%m-%d",
        errors="coerce",
    )


def fetch_cn_cpi():
    """Fetch China CPI MoM data."""
    try:
        df = ak.macro_china_cpi_monthly()
        if df is None or df.empty:
            return pd.DataFrame()
        # New columns: 商品, 日期 (announcement date), 今值, 预测值, 前值
        if "今值" in df.columns and "日期" in df.columns:
            df = df[["日期", "今值"]].rename(columns={"日期": "date", "今值": "value"})
        elif "全国-当月" in df.columns:
            df = df.rename(columns={"日期": "date", "全国-当月": "value"})
        else:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["indicator"] = "cn_cpi"
        df["name"] = "China CPI"
        df["country"] = "CN"
        return df[["date", "indicator", "name", "country", "value"]]
    except Exception as e:
        log.warning(f"Failed to fetch CN CPI: {e}")
        return pd.DataFrame()


def fetch_cn_ppi():
    """Fetch China PPI data."""
    try:
        df = ak.macro_china_ppi()
        if df is None or df.empty:
            return pd.DataFrame()
        # Columns: 月份, 当月, 当月同比增长, 累计
        if "月份" in df.columns and "当月同比增长" in df.columns:
            df["date"] = _parse_month_col(df["月份"])
            df = df.rename(columns={"当月同比增长": "value"})
        elif "日期" in df.columns and "当月" in df.columns:
            df = df.rename(columns={"日期": "date", "当月": "value"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            return pd.DataFrame()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["indicator"] = "cn_ppi"
        df["name"] = "China PPI"
        df["country"] = "CN"
        return df[["date", "indicator", "name", "country", "value"]]
    except Exception as e:
        log.warning(f"Failed to fetch CN PPI: {e}")
        return pd.DataFrame()


def fetch_cn_pmi():
    """Fetch China PMI data."""
    try:
        df = ak.macro_china_pmi()
        if df is None or df.empty:
            return pd.DataFrame()
        # Columns: 月份, 制造业-指数, 制造业-同比增长, 非制造业-指数, ...
        if "月份" in df.columns and "制造业-指数" in df.columns:
            df["date"] = _parse_month_col(df["月份"])
            df = df.rename(columns={"制造业-指数": "value"})
        elif "日期" in df.columns and "制造业-PMI" in df.columns:
            df = df.rename(columns={"日期": "date", "制造业-PMI": "value"})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            return pd.DataFrame()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["indicator"] = "cn_pmi"
        df["name"] = "China PMI"
        df["country"] = "CN"
        return df[["date", "indicator", "name", "country", "value"]]
    except Exception as e:
        log.warning(f"Failed to fetch CN PMI: {e}")
        return pd.DataFrame()


def fetch_cn_gdp():
    """Fetch China GDP growth data (quarterly)."""
    try:
        df = ak.macro_china_gdp()
        if df is None or df.empty:
            return pd.DataFrame()
        # Columns: 季度, 国内生产总值-同比增长, ...
        # 季度 format: '2025年第1-4季度', '2025年第1季度', etc.
        if "季度" not in df.columns or "国内生产总值-同比增长" not in df.columns:
            return pd.DataFrame()

        quarter_map = {"1": "01", "2": "04", "3": "07", "4": "10"}

        def parse_quarter(s):
            import re
            # Only take single quarters like '2025年第1季度', skip cumulative '第1-4季度'
            m = re.match(r"(\d{4})年第([1-4])季度$", str(s))
            if not m:
                return pd.NaT
            year, q = m.group(1), m.group(2)
            return pd.Timestamp(f"{year}-{quarter_map[q]}-01")

        df["date"] = df["季度"].apply(parse_quarter)
        df = df.rename(columns={"国内生产总值-同比增长": "value"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value"])
        df["indicator"] = "cn_gdp"
        df["name"] = "China GDP Growth"
        df["country"] = "CN"
        return df[["date", "indicator", "name", "country", "value"]]
    except Exception as e:
        log.warning(f"Failed to fetch CN GDP: {e}")
        return pd.DataFrame()


def fetch_cn_house_price():
    """Fetch China new home price index YoY change (akshare).

    Computes average YoY across available cities.
    Value = index - 100 to convert to percentage change.
    """
    try:
        df = ak.macro_china_new_house_price()
        if df is None or df.empty:
            return pd.DataFrame()

        date_col = next((c for c in df.columns if "日期" in str(c) or "date" in str(c).lower()), None)
        yoy_col = next((c for c in df.columns if "新建商品住宅" in str(c) and "同比" in str(c)), None)

        if not date_col or not yoy_col:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["raw"] = pd.to_numeric(df[yoy_col], errors="coerce")
        df = df.dropna(subset=["date", "raw"])

        # Average across cities, convert index (e.g. 101.5) to % (1.5)
        avg = df.groupby("date")["raw"].mean().reset_index()
        avg["value"] = avg["raw"] - 100
        avg["indicator"] = "CN House Price YoY"
        avg["name"] = "China House Price YoY"
        avg["country"] = "CN"
        return avg[["date", "indicator", "name", "country", "value"]]
    except Exception as e:
        log.warning(f"Failed to fetch CN house price: {e}")
        return pd.DataFrame()


_CN_CITIES = [
    ("北京", "Beijing"), ("上海", "Shanghai"), ("广州", "Guangzhou"), ("深圳", "Shenzhen"),
    ("天津", "Tianjin"), ("重庆", "Chongqing"), ("成都", "Chengdu"), ("杭州", "Hangzhou"),
    ("武汉", "Wuhan"), ("西安", "Xian"), ("南京", "Nanjing"), ("苏州", "Suzhou"),
    ("青岛", "Qingdao"), ("郑州", "Zhengzhou"), ("长沙", "Changsha"), ("济南", "Jinan"),
    ("合肥", "Hefei"), ("宁波", "Ningbo"), ("厦门", "Xiamen"), ("福州", "Fuzhou"),
]


def fetch_cn_house_price_cities():
    """Fetch new home price index for major CN cities (last 24 months).

    Returns DataFrame with columns:
      date, city_zh, city_en, yoy_pct, mom_pct
    where yoy_pct/mom_pct are percentage changes (index - 100).
    """
    import time
    rows = []
    cities_zh = [c[0] for c in _CN_CITIES]
    en_map = {c[0]: c[1] for c in _CN_CITIES}

    for i in range(0, len(cities_zh), 2):
        c1 = cities_zh[i]
        c2 = cities_zh[i + 1] if i + 1 < len(cities_zh) else c1
        try:
            df = ak.macro_china_new_house_price(city_first=c1, city_second=c2)
            if df is None or df.empty:
                continue
            df["date"] = pd.to_datetime(df["日期"], errors="coerce")
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=750)
            df = df[df["date"] >= cutoff].copy()
            df["yoy_pct"] = pd.to_numeric(df["新建商品住宅价格指数-同比"], errors="coerce") - 100
            df["mom_pct"] = pd.to_numeric(df["新建商品住宅价格指数-环比"], errors="coerce") - 100
            for _, row in df.dropna(subset=["date", "yoy_pct"]).iterrows():
                rows.append({
                    "date": row["date"],
                    "city_zh": row["城市"],
                    "city_en": en_map.get(row["城市"], row["城市"]),
                    "yoy_pct": round(float(row["yoy_pct"]), 2),
                    "mom_pct": round(float(row["mom_pct"]), 2) if pd.notna(row["mom_pct"]) else None,
                })
            time.sleep(0.2)
        except Exception as e:
            log.warning(f"Failed to fetch CN house price for {c1}/{c2}: {e}")

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values(["city_zh", "date"]).reset_index(drop=True)
    log.info(f"CN house price cities: {result['city_zh'].nunique()} cities, {len(result)} rows")
    return result


def fetch_all_cn_macro():
    """Fetch all China macro indicators.

    Returns combined DataFrame.
    """
    all_data = []

    for name, func in [
        ("CPI", fetch_cn_cpi),
        ("PPI", fetch_cn_ppi),
        ("PMI", fetch_cn_pmi),
        ("GDP", fetch_cn_gdp),
        ("House Price", fetch_cn_house_price),
    ]:
        df = func()
        if not df.empty:
            all_data.append(df)
            log.info(f"  CN {name}: {len(df)} data points")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)
