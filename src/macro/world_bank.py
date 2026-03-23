"""World Bank API client for global economic indicators."""

import requests
import pandas as pd
from src.common.config import load_yaml
from src.common.logger import get_logger

log = get_logger("macro.worldbank")

BASE_URL = "https://api.worldbank.org/v2"


def fetch_wb_indicator(indicator, countries, start_year=None, end_year=None):
    """Fetch a World Bank indicator for a list of countries.

    Args:
        indicator: World Bank indicator code (e.g., "NY.GDP.MKTP.KD.ZG").
        countries: List of ISO3 country codes.
        start_year: Start year (default: 5 years ago).
        end_year: End year (default: current year).

    Returns:
        DataFrame with: country, year, indicator, value
    """
    import datetime
    if end_year is None:
        end_year = datetime.datetime.now().year
    if start_year is None:
        start_year = end_year - 5

    country_str = ";".join(countries)
    url = f"{BASE_URL}/country/{country_str}/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": 1000,
        "date": f"{start_year}:{end_year}",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if len(data) < 2 or not data[1]:
            return pd.DataFrame()

        rows = []
        for item in data[1]:
            val = item.get("value")
            if val is None:
                continue
            rows.append({
                "country": item["country"]["id"],
                "country_name": item["country"]["value"],
                "year": int(item["date"]),
                "indicator": indicator,
                "indicator_name": item["indicator"]["value"],
                "value": float(val),
            })

        df = pd.DataFrame(rows)
        log.info(f"  World Bank {indicator}: {len(df)} data points")
        return df

    except Exception as e:
        log.warning(f"Failed to fetch World Bank {indicator}: {e}")
        return pd.DataFrame()


def fetch_all_wb_indicators():
    """Fetch all World Bank indicators defined in macro_indicators.yaml.

    Returns combined DataFrame.
    """
    cfg = load_yaml("macro_indicators.yaml")
    wb_cfg = cfg.get("world_bank", [])
    all_data = []

    for item in wb_cfg:
        df = fetch_wb_indicator(
            indicator=item["indicator"],
            countries=item["countries"],
        )
        if not df.empty:
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    log.info(f"Fetched {len(result)} World Bank data points")
    return result
