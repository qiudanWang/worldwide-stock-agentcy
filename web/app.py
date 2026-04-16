"""Global Tech Market — Flask Dashboard"""
import sys
import os
import json
import glob
import time
from datetime import datetime

# Inject macOS native trust store so Python uses the system keychain.
# Required on corporate networks using SSL inspection (e.g. Cisco Secure Access).
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from flask import Flask, render_template, jsonify, request, redirect, url_for, make_response

from src.common.config import get_data_path, load_yaml
from src.common.stock_info import enrich_alerts, load_name_lookup, get_board, US_STOCK_NAMES
from src.market_data.market_cap import format_market_cap, load_market_caps
from src.market_data.indices import fetch_indices
from src.analysis.sector_performance import normalize_sector_df, meta_sector_heatmap, _SECTOR_MAP
from src.news.world_monitor import FEEDS as GEO_FEEDS
from src.macro.polymarket import fetch_polymarket_signals

app = Flask(__name__)
app.jinja_env.globals["format_market_cap"] = format_market_cap

# Canonical market display order used everywhere
_MARKET_ORDER = ["CN", "US", "HK", "JP", "IN", "UK", "DE", "FR", "KR", "TW", "AU", "BR", "SA"]

def _market_sort_key(m):
    try:
        return _MARKET_ORDER.index(m)
    except ValueError:
        return 99

# ---------------------------------------------------------------------------
# i18n — language selection via cookie
# ---------------------------------------------------------------------------
_I18N_DIR = os.path.join(os.path.dirname(__file__), "i18n")
_SUPPORTED_LANGS = ["en", "zh", "ja", "ko"]
_LANG_LABELS = {"en": "EN", "zh": "中文", "ja": "日本語", "ko": "한국어"}
_TRANSLATIONS: dict = {}


def _load_translations():
    for lang in _SUPPORTED_LANGS:
        path = os.path.join(_I18N_DIR, f"{lang}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                _TRANSLATIONS[lang] = json.load(f)
        except Exception:
            _TRANSLATIONS[lang] = {}


_load_translations()


def _get_lang() -> str:
    return "en"  # Page UI is always English; language selector is in chatbox only


def _make_t(lang: str):
    strings = _TRANSLATIONS.get(lang, {})
    en_strings = _TRANSLATIONS.get("en", {})

    def _lookup(store, parts):
        val = store
        for p in parts:
            val = val.get(p) if isinstance(val, dict) else None
            if val is None:
                return None
        return val

    def t(key: str) -> str:
        """Return translated string, fallback to English, then key."""
        parts = key.split(".")
        val = _lookup(strings, parts)
        if isinstance(val, str):
            return val
        val = _lookup(en_strings, parts)
        return val if isinstance(val, str) else key

    def tv(key: str):
        """Return translated value (any type: str, list, dict). Fallback to English, then key."""
        parts = key.split(".")
        val = _lookup(strings, parts)
        if val is not None:
            return val
        val = _lookup(en_strings, parts)
        return val if val is not None else key

    return t, tv


@app.context_processor
def inject_i18n():
    lang = _get_lang()
    t, tv = _make_t(lang)
    return {
        "t": t,
        "tv": tv,
        "current_lang": lang,
        "lang_labels": _LANG_LABELS,
        "supported_langs": _SUPPORTED_LANGS,
        "global_market_codes": _MARKET_ORDER,
    }


@app.route("/set-lang/<lang>")
def set_lang(lang: str):
    if lang not in _SUPPORTED_LANGS:
        lang = "en"
    next_url = request.args.get("next", "/")
    resp = make_response(redirect(next_url))
    resp.set_cookie("lang", lang, max_age=365 * 24 * 3600, samesite="Lax")
    return resp



def load_cached_indices():
    """Load indices from saved parquet files — fast, no live API calls.

    Falls back to an empty list if no files exist yet.
    Returns same format as fetch_indices(): list of dicts with market/name/close/change_pct.
    """
    results = []
    markets_cfg = get_markets_config()
    for market_code in markets_cfg:
        path = get_data_path("markets", market_code, "indices.parquet")
        try:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
            for symbol, grp in df.groupby("symbol"):
                grp = grp.dropna(subset=["close"])
                if grp.empty:
                    continue
                last = grp.iloc[-1]
                change_pct = last.get("change_pct")
                results.append({
                    "market": market_code,
                    "symbol": symbol,
                    "name": last.get("name", symbol),
                    "close": round(float(last["close"]), 2),
                    "change_pct": round(float(change_pct), 2) if change_pct is not None and not pd.isna(change_pct) else None,
                })
        except Exception:
            pass
    return results


# Simple in-memory cache for polymarket (10-minute TTL)
_polymarket_cache = {"data": [], "ts": 0}

def _get_polymarket_signals():
    if time.time() - _polymarket_cache["ts"] < 600:
        return _polymarket_cache["data"]
    try:
        data = fetch_polymarket_signals()
        _polymarket_cache["data"] = data
        _polymarket_cache["ts"] = time.time()
    except Exception:
        data = _polymarket_cache["data"]
    return data


def load_parquet(path):
    """Load a parquet file, return empty DataFrame if not found."""
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, Exception):
        return pd.DataFrame()


def load_json(path):
    """Load a JSON file, return empty dict/list if not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, Exception):
        return {}


def _sparkline(prices):
    """Convert a list of prices to an SVG polyline points string (80x24 viewBox)."""
    if not prices or len(prices) < 2:
        return None
    lo, hi = min(prices), max(prices)
    rng = hi - lo or 1
    pts = []
    for i, v in enumerate(prices):
        x = round(i / (len(prices) - 1) * 80, 1)
        y = round((1 - (v - lo) / rng) * 24, 1)
        pts.append(f"{x},{y}")
    return " ".join(pts)


def _combined_histbars(gdp_values, gdp_years, cpi_values, cpi_years, W=80, H=24):
    """Return SVG rect dicts for a combined GDP+CPI histogram (80×24 viewBox).

    Each year gets two side-by-side bars: GDP (blue) on the left, CPI (orange)
    on the right. Both series share the same Y axis scale.
    Returns (bars, baseline_y_or_None).
    """
    all_years = sorted(set(gdp_years) | set(cpi_years))
    n = len(all_years)
    if not n:
        return [], None

    gdp_map = dict(zip(gdp_years, gdp_values))
    cpi_map = dict(zip(cpi_years, cpi_values))

    all_vals = list(gdp_values) + list(cpi_values)
    if not all_vals:
        return [], None

    lo, hi = min(all_vals), max(all_vals)
    has_neg = lo < 0 and hi > 0
    vrange = hi - lo or 1

    baseline_y = H * hi / (hi - lo) if has_neg else H

    slot = W / n
    bar_w = round(slot * 0.36, 1)  # two bars + gap fit in one slot
    gap = round(slot * 0.04, 1)

    bars = []
    for i, year in enumerate(all_years):
        x_gdp = round(i * slot + slot * 0.08, 1)
        x_cpi = round(x_gdp + bar_w + gap, 1)

        for kind, val_map, x in [("gdp", gdp_map, x_gdp), ("cpi", cpi_map, x_cpi)]:
            if year not in val_map:
                continue
            v = val_map[year]
            if has_neg:
                bar_h = round(abs(v) / (hi - lo) * H, 1)
                y = round(baseline_y - max(v, 0) / (hi - lo) * H, 1)
            else:
                bar_h = round(max((v - lo) / vrange * H * 0.88 + H * 0.06, 1.5), 1)
                y = round(H - bar_h, 1)
            bars.append({
                "x": x, "y": max(y, 0),
                "w": bar_w, "h": max(bar_h, 1.0),
                "kind": kind,       # "gdp" or "cpi"
                "pos": v >= 0,
                "year": year,
            })

    baseline_svg = round(baseline_y, 1) if has_neg else None
    return bars, baseline_svg


_FX_INDICATORS = [
    "US Dollar Index (DXY)",
    "EUR/USD",
    "GBP/USD",
    "USD/CAD",
    "USD/CHF",
    "USD/SEK",
    "USD/CNY",
    "USD/JPY",
    "USD/INR",
]

# indicator key → (currency code, full name, value label)
# EUR/USD and GBP/USD are quoted as foreign currency per USD
_FX_META = {
    "US Dollar Index (DXY)": ("DXY",  "USD Index",        "Index"),
    "EUR/USD":                ("EUR",  "Euro",             "1 EUR = x USD"),
    "GBP/USD":                ("GBP",  "British Pound",    "1 GBP = x USD"),
    "USD/CAD":                ("CAD",  "Canadian Dollar",  "1 USD = x CAD"),
    "USD/CHF":                ("CHF",  "Swiss Franc",      "1 USD = x CHF"),
    "USD/SEK":                ("SEK",  "Swedish Krona",    "1 USD = x SEK"),
    "USD/CNY":                ("CNY",  "Chinese Yuan",     "1 USD = x CNY"),
    "USD/JPY":                ("JPY",  "Japanese Yen",     "1 USD = x JPY"),
    "USD/INR":                ("INR",  "Indian Rupee",     "1 USD = x INR"),
}


def build_fx_table(macro_df, macro_latest, n_points=30):
    """Build FX table rows with 30d sparklines for the macro page."""
    rows = []
    for name in _FX_INDICATORS:
        hist = macro_df[macro_df["indicator"] == name].sort_values("date").tail(n_points)
        hist = hist.dropna(subset=["value"])
        if hist.empty:
            continue
        values = hist["value"].tolist()
        dates = [str(d)[:10] for d in hist["date"].tolist()]
        latest_val = values[-1] if values else None
        change_1d = (macro_latest.get(name) or {}).get("change_pct")
        # Fallback: compute 1d change from last 2 history rows
        if change_1d is None and len(values) >= 2 and values[-2]:
            change_1d = (values[-1] - values[-2]) / values[-2]
        change_30d = None
        if len(values) >= 2 and values[0]:
            change_30d = (values[-1] - values[0]) / values[0]
        code, full_name, value_label = _FX_META.get(name, (name, name, ""))
        rows.append({
            "code": code,
            "name": full_name,
            "value_label": value_label,
            "value": round(latest_val, 4) if latest_val is not None else None,
            "change_1d": change_1d,
            "change_30d": change_30d,
            "spark_dates": dates,
            "spark_values": [round(v, 4) for v in values],
            "sparkline": _sparkline(values),
        })
    return rows


def build_macro_sparklines(macro_latest, macro_df, n_points=30):
    """Enrich macro_latest dict with sparkline SVG path and range data."""
    if macro_df.empty or not macro_latest:
        return macro_latest

    enriched = {}
    for name, data in macro_latest.items():
        entry = dict(data)
        hist = macro_df[macro_df["indicator"] == name].sort_values("date").tail(n_points)
        hist = hist.dropna(subset=["value"])
        values = hist["value"].tolist()
        dates = hist["date"].astype(str).tolist()
        if len(values) >= 2:
            lo, hi = min(values), max(values)
            rng = hi - lo or 1
            w, h = 80, 24
            pts = []
            for i, v in enumerate(values):
                x = round(i / (len(values) - 1) * w, 1)
                y = round((1 - (v - lo) / rng) * h, 1)
                pts.append(f"{x},{y}")
            entry["sparkline"] = " ".join(pts)
            entry["spark_lo"] = round(lo, 2)
            entry["spark_hi"] = round(hi, 2)
            entry["spark_dates"] = dates
            entry["spark_values"] = [round(v, 4) for v in values]
            cur = values[-1]
            entry["range_pct"] = round((cur - lo) / rng * 100)
            # Dynamic range label based on actual date span
            try:
                d0, d1 = pd.to_datetime(dates[0]), pd.to_datetime(dates[-1])
                span_days = (d1 - d0).days
                if span_days > 365 * 2:
                    entry["range_label"] = f"{d0.year}–{d1.year}"
                elif span_days > 60:
                    entry["range_label"] = f"{span_days // 30}m range"
                else:
                    entry["range_label"] = f"{span_days}d range"
            except Exception:
                entry["range_label"] = "range"
        else:
            entry["sparkline"] = None
            entry["spark_lo"] = None
            entry["spark_hi"] = None
            entry["spark_dates"] = []
            entry["spark_values"] = []
            entry["range_pct"] = None
            entry["range_label"] = "range"
        enriched[name] = entry
    return enriched


_COMBINED_CHART_COLORS = [
    "#38bdf8",  # sky blue   — Australia
    "#4ade80",  # green      — Brazil
    "#f87171",  # red        — China
    "#fb923c",  # orange     — France
    "#facc15",  # yellow     — Germany
    "#e879f9",  # fuchsia    — India
    "#22d3ee",  # cyan       — Japan
    "#f472b6",  # pink       — Korea, Rep.
    "#a3e635",  # lime       — Saudi Arabia
    "#818cf8",  # indigo     — United Kingdom
    "#34d399",  # emerald    — United States
    "#94a3b8",  # slate      — extra
    "#ff6b6b",  # coral      — extra
]

# Maps World Bank country name → (policy_rate_indicator, yield_10y_indicator, yield_2y_indicator)
# ST Rate = OECD short-term interest rate (3-month money market), close proxy for policy rate
_COUNTRY_RATES = {
    # (policy_rate_ind, yield_10y_ind, yield_2y_or_st_ind)
    "United States":  ("Fed Funds Rate",  "10Y Treasury Yield",   "2Y Treasury Yield"),
    "Germany":        ("ECB Policy Rate", "Germany 10Y Yield",    "Germany ST Rate"),
    "France":         ("ECB Policy Rate", "France 10Y Yield",     "France ST Rate"),
    # For countries below we only have OECD 3-month ST rates — show in 2Y/ST column
    "United Kingdom": (None,             "UK 10Y Yield",          "UK ST Rate"),
    "Japan":          (None,             "Japan 10Y Yield",       "Japan ST Rate"),
    "Australia":      (None,             "Australia 10Y Yield",   "Australia ST Rate"),
    "Korea, Rep.":    (None,             "Korea 10Y Yield",       "Korea ST Rate"),
    "Brazil":         (None,             None,                    "Brazil ST Rate"),
    "India":          (None,             None,                    "India ST Rate"),
    "China":          (None,             None,                    None),
    "Saudi Arabia":   (None,             None,                    None),
}

_COUNTRY_TO_MARKET = {
    "United States": "US",
    "China": "CN",
    "Japan": "JP",
    "India": "IN",
    "United Kingdom": "UK",
    "Germany": "DE",
    "France": "FR",
    "Korea, Rep.": "KR",
    "Brazil": "BR",
    "Saudi Arabia": "SA",
    "Australia": "AU",
}


_MARKET_INDEX_SYMBOL = {
    "US": "^GSPC", "CN": "000001.SS", "HK": "^HSI", "JP": "^N225",
    "IN": "^NSEI", "UK": "^FTSE", "DE": "^GDAXI", "FR": "^FCHI",
    "KR": "^KS11", "TW": "^TWII", "AU": "^AXJO", "BR": "^BVSP", "SA": "^TASI.SR",
}


def _get_stock_1y_return(market_code):
    """Compute 1Y return for the primary index of a market.

    Uses local indices.parquet if it has >= 300 days of history,
    otherwise fetches from yfinance directly.
    """
    try:
        path = get_data_path("markets", market_code, "indices.parquet")
        df = pd.DataFrame()
        if os.path.exists(path):
            df = load_parquet(path)

        if not df.empty and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "close"])
            if "symbol" in df.columns:
                primary = df.groupby("symbol")["close"].count().idxmax()
                df = df[df["symbol"] == primary]
            df = df.sort_values("date")
            date_range = (df.iloc[-1]["date"] - df.iloc[0]["date"]).days if len(df) >= 2 else 0
            if date_range >= 300:
                latest_date = df.iloc[-1]["date"]
                latest_val = df.iloc[-1]["close"]
                cutoff_hi = latest_date - pd.Timedelta(days=335)
                cutoff_lo = latest_date - pd.Timedelta(days=395)
                past = df[(df["date"] >= cutoff_lo) & (df["date"] <= cutoff_hi)]
                if past.empty:
                    past = df[df["date"] <= latest_date - pd.Timedelta(days=335)]
                if not past.empty and past.iloc[-1]["close"]:
                    return round((latest_val / past.iloc[-1]["close"] - 1) * 100, 1)

        # Fall back to yfinance for 1Y history
        symbol = _MARKET_INDEX_SYMBOL.get(market_code)
        if not symbol:
            return None
        import yfinance as yf
        hist = yf.download(symbol, period="13mo", interval="1mo", progress=False, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 12:
            return None
        latest_val = float(hist["Close"].iloc[-1].iloc[0] if hasattr(hist["Close"].iloc[-1], "iloc") else hist["Close"].iloc[-1])
        base_row = hist["Close"].iloc[-13] if len(hist) >= 13 else hist["Close"].iloc[0]
        base_val = float(base_row.iloc[0] if hasattr(base_row, "iloc") else base_row)
        if not base_val:
            return None
        return round((latest_val / base_val - 1) * 100, 1)
    except Exception:
        return None


def _build_detail_charts(country, macro_df, gdp, cpi, unemp, ca,
                          policy_ind, yield_ind, yield2y_ind):
    """Build list of chart dicts for expanded country detail row."""
    charts = []
    now = pd.Timestamp.now()
    cutoff_24m = now - pd.Timedelta(days=750)
    cutoff_36m = now - pd.Timedelta(days=1100)

    def _series(indicator, lookback_days=750):
        sub = macro_df[macro_df["indicator"] == indicator].dropna(subset=["value"]).sort_values("date")
        if sub.empty:
            return None
        sub = sub[sub["date"] >= now - pd.Timedelta(days=lookback_days)]
        return sub if len(sub) >= 2 else None

    def _annual(d, name, color, ref_line=None):
        vals = d["values"][-10:] if d["values"] else []
        years = d["years"][-10:] if d["years"] else []
        if len(vals) < 2:
            return None
        return {"name": name, "type": "bar", "dates": years,
                "values": vals, "unit": "%", "ref_line": ref_line, "color": color}

    def _line(sub, name, color, unit="%", ref_line=None):
        if sub is None:
            return None
        dates = sub["date"].dt.strftime("%Y-%m").tolist()
        values = [round(float(v), 3) for v in sub["value"]]
        return {"name": name, "type": "line", "dates": dates,
                "values": values, "unit": unit, "ref_line": ref_line, "color": color}

    def _yoy_series(indicator, lookback_days=1800):
        raw = macro_df[macro_df["indicator"] == indicator].dropna(subset=["value"]).sort_values("date")
        if len(raw) < 13:
            return None
        raw = raw.set_index("date")["value"]
        yoy = ((raw / raw.shift(12) - 1) * 100).dropna().reset_index()
        yoy.columns = ["date", "value"]
        yoy = yoy[yoy["date"] >= cutoff_24m]
        return yoy if len(yoy) >= 2 else None

    # Annual World Bank indicators (all countries)
    c = _annual(gdp, "GDP Growth", "#38bdf8", ref_line=0)
    if c: charts.append(c)
    c = _annual(cpi, "CPI Inflation", "#fb923c", ref_line=2)
    if c: charts.append(c)
    c = _annual(unemp, "Unemployment", "#58a6ff")
    if c: charts.append(c)
    c = _annual(ca, "Current Account", "#3fb950", ref_line=0)
    if c: charts.append(c)

    # Monthly FRED rate history
    if yield_ind:
        c = _line(_series(yield_ind, 1100), "10Y Yield", "#818cf8")
        if c: charts.append(c)
    rate_ind = policy_ind or yield2y_ind
    if rate_ind and rate_ind != yield_ind:
        label = "Policy Rate" if policy_ind else "ST Rate"
        c = _line(_series(rate_ind, 1100), label, "#38bdf8")
        if c: charts.append(c)

    # China-specific monthly (akshare)
    if country == "China":
        for ind, name, color, ref in [
            ("China PMI",            "Manufacturing PMI", "#3fb950", 50),
            ("China PPI",            "PPI YoY",           "#58a6ff",  0),
            ("China CPI",            "CPI MoM",           "#fb923c",  0),
            ("China House Price YoY","House Price YoY",   "#ff7b72",  0),
        ]:
            unit = "index" if "PMI" in ind else "%"
            c = _line(_series(ind, 750), name, color, unit=unit, ref_line=ref)
            if c: charts.append(c)

        # Per-city house price table
        try:
            city_path = get_data_path("markets", "CN", "house_price_cities.parquet")
            if os.path.exists(city_path):
                cities_df = pd.read_parquet(city_path)
                if not cities_df.empty:
                    cities_df["date"] = pd.to_datetime(cities_df["date"])
                    latest_cities = (
                        cities_df.sort_values("date")
                        .groupby("city_zh")
                        .tail(1)
                        .sort_values("yoy_pct", ascending=False)
                    )
                    cutoff_city = now - pd.Timedelta(days=400)
                    city_items = []
                    for _, crow in latest_cities.iterrows():
                        city_hist = cities_df[
                            (cities_df["city_zh"] == crow["city_zh"]) &
                            (cities_df["date"] >= cutoff_city)
                        ].sort_values("date")
                        spark_vals = [round(float(v), 2) for v in city_hist["yoy_pct"].tolist()]
                        spark_dates = city_hist["date"].dt.strftime("%Y-%m").tolist()
                        yoy = round(float(crow["yoy_pct"]), 2) if pd.notna(crow["yoy_pct"]) else None
                        city_items.append({
                            "city_en": crow["city_en"],
                            "city_zh": crow["city_zh"],
                            "price_idx": round(yoy + 100, 2) if yoy is not None else None,
                            "yoy_pct": yoy,
                            "mom_pct": round(float(crow["mom_pct"]), 2) if pd.notna(crow.get("mom_pct")) else None,
                            "spark_vals": spark_vals,
                            "spark_dates": spark_dates,
                        })
                    if city_items:
                        as_of = latest_cities.iloc[0]["date"].strftime("%Y-%m") if len(latest_cities) > 0 else ""
                        charts.append({
                            "name": "CN City House Prices (YoY%)",
                            "type": "city_table",
                            "cities": city_items,
                            "as_of": as_of,
                        })
        except Exception as e:
            log.warning(f"Failed to build CN city house price chart: {e}")

    # US-specific monthly (FRED)
    if country == "United States":
        c = _line(_yoy_series("US CPI"),        "CPI YoY",         "#fb923c", ref_line=2)
        if c: charts.append(c)
        c = _line(_series("US Unemployment Rate", 750), "Unemployment", "#58a6ff")
        if c: charts.append(c)
        c = _line(_yoy_series("US PCE"),        "PCE YoY",         "#3fb950", ref_line=2)
        if c: charts.append(c)
        c = _line(_yoy_series("US Case-Shiller HPI"), "Case-Shiller YoY", "#ff7b72", ref_line=0)
        if c: charts.append(c)

    return charts


def build_country_table(macro_df):
    """Build combined country economic table with one row per country.

    Columns: GDP (latest + YoY Δ), CPI (latest + YoY Δ), Unemployment,
    Stock 1Y return, Current Account Balance (% GDP), Real Estate (US only),
    plus a combined GDP+CPI histogram.
    """
    if macro_df.empty:
        return []

    gdp_df = macro_df[macro_df["indicator"].str.contains(
        "GDP growth (annual %)", na=False, regex=False)].copy()
    cpi_df = macro_df[macro_df["indicator"].str.contains(
        "Inflation, consumer prices", na=False, regex=False)].copy()
    unemp_df = macro_df[macro_df["indicator"].str.contains(
        "Unemployment, total", na=False, regex=False)].copy()
    ca_df = macro_df[macro_df["indicator"].str.contains(
        "Current account balance", na=False, regex=False)].copy()
    re_us_df = macro_df[macro_df["indicator"] == "US Case-Shiller HPI"].copy()
    re_cn_df = macro_df[macro_df["indicator"] == "China House Price YoY"].copy()

    if gdp_df.empty and cpi_df.empty:
        return []

    def _extract(df):
        df = df.copy()
        df["country"] = df["indicator"].astype(str).str.split(": ").str[-1]
        result = {}
        for country, grp in df.groupby("country"):
            grp = grp.sort_values("date").dropna(subset=["value"])
            if grp.empty:
                continue
            vals  = [round(float(v), 2) for v in grp["value"].tolist()]
            years = grp["date"].dt.strftime("%Y").tolist()
            result[country] = {"values": vals, "years": years}
        return result

    gdp_by_country = _extract(gdp_df)
    cpi_by_country = _extract(cpi_df)
    unemp_by_country = _extract(unemp_df) if not unemp_df.empty else {}
    ca_by_country = _extract(ca_df) if not ca_df.empty else {}
    _country_market = {c: _COUNTRY_TO_MARKET.get(c, "ZZ") for c in set(gdp_by_country) | set(cpi_by_country)}
    all_countries = sorted(_country_market, key=lambda c: _market_sort_key(_country_market[c]))

    # Build latest-value lookup and sparklines for rate indicators
    rate_indicators = set()
    for policy_ind, yield_ind, yield2y_ind in _COUNTRY_RATES.values():
        if policy_ind:
            rate_indicators.add(policy_ind)
        if yield_ind:
            rate_indicators.add(yield_ind)
        if yield2y_ind:
            rate_indicators.add(yield2y_ind)
    rate_latest = {}
    rate_sparks = {}   # indicator → {sparkline, spark_dates, spark_values, spark_lo, spark_hi}
    for ind in rate_indicators:
        sub = macro_df[macro_df["indicator"] == ind].dropna(subset=["value"]).sort_values("date")
        if not sub.empty:
            val = round(float(sub.iloc[-1]["value"]), 2)
            # Use 1-year delta so flat daily series (Fed Funds, ECB) still show change
            latest_date = sub.iloc[-1]["date"]
            year_ago = sub[sub["date"] <= latest_date - pd.Timedelta(days=335)]
            if not year_ago.empty:
                delta = round(val - float(year_ago.iloc[-1]["value"]), 3)
            elif len(sub) >= 2:
                delta = round(val - float(sub.iloc[-2]["value"]), 3)
            else:
                delta = None
            rate_latest[ind] = {"value": val, "delta": delta}
            # Sparkline (last 60 points)
            hist = sub.tail(60)
            vals = hist["value"].tolist()
            dates = hist["date"].astype(str).str[:10].tolist()
            if len(vals) >= 2:
                lo, hi = min(vals), max(vals)
                rng = hi - lo or 1
                pts = []
                for i, v in enumerate(vals):
                    x = round(i / (len(vals) - 1) * 80, 1)
                    y = round((1 - (v - lo) / rng) * 24, 1)
                    pts.append(f"{x},{y}")
                rate_sparks[ind] = {
                    "sparkline": " ".join(pts),
                    "spark_dates": dates,
                    "spark_values": [round(v, 3) for v in vals],
                    "spark_lo": round(lo, 3),
                    "spark_hi": round(hi, 3),
                }

    # CN house price: current YoY % and MoM delta in YoY rate
    re_cn_val = re_cn_chg = None
    if not re_cn_df.empty:
        re_cn_df = re_cn_df.sort_values("date").dropna(subset=["value"])
        if not re_cn_df.empty:
            re_cn_val = round(float(re_cn_df.iloc[-1]["value"]), 2)
            if len(re_cn_df) >= 2:
                re_cn_chg = round(float(re_cn_df.iloc[-1]["value"]) - float(re_cn_df.iloc[-2]["value"]), 2)

    # Case-Shiller: current index level and 1Y YoY %
    re_us_val = re_us_chg = None
    if not re_us_df.empty:
        re_us_df = re_us_df.sort_values("date").dropna(subset=["value"])
        if len(re_us_df) >= 13:
            latest_re = re_us_df.iloc[-1]["value"]
            re_us_val = round(float(latest_re), 1)
            latest_re_date = re_us_df.iloc[-1]["date"]
            cutoff_hi = pd.to_datetime(latest_re_date) - pd.Timedelta(days=335)
            cutoff_lo = pd.to_datetime(latest_re_date) - pd.Timedelta(days=395)
            past_re = re_us_df[
                (re_us_df["date"] >= cutoff_lo) & (re_us_df["date"] <= cutoff_hi)
            ]
            if not past_re.empty and past_re.iloc[-1]["value"]:
                re_us_chg = round((latest_re / past_re.iloc[-1]["value"] - 1) * 100, 1)

    rows = []
    for country in all_countries:
        gdp = gdp_by_country.get(country, {"values": [], "years": []})
        cpi = cpi_by_country.get(country, {"values": [], "years": []})
        unemp = unemp_by_country.get(country, {"values": [], "years": []})
        ca = ca_by_country.get(country, {"values": [], "years": []})

        def _latest_and_change(d):
            vals = d["values"]
            years = d["years"]
            if not vals:
                return None, None, None
            latest = vals[-1]
            year   = years[-1]
            change = round(vals[-1] - vals[-2], 2) if len(vals) >= 2 else None
            return latest, year, change

        gdp_val, gdp_year, gdp_chg = _latest_and_change(gdp)
        cpi_val, cpi_year, cpi_chg = _latest_and_change(cpi)
        unemp_val, unemp_year, _ = _latest_and_change(unemp)
        ca_val, ca_year, _ = _latest_and_change(ca)

        market_code = _COUNTRY_TO_MARKET.get(country)
        stock_1y = _get_stock_1y_return(market_code) if market_code else None
        if country == "United States":
            re_val, re_chg = re_us_val, re_us_chg
            re_val_label = "idx"
        elif country == "China":
            re_cn_index = round(re_cn_val + 100, 2) if re_cn_val is not None else None
            re_val, re_chg = re_cn_index, re_cn_val
            re_val_label = "idx"
        else:
            re_val, re_chg, re_val_label = None, None, None

        policy_ind, yield_ind, yield2y_ind = _COUNTRY_RATES.get(country, (None, None, None))
        def _rate(ind):
            d = rate_latest.get(ind) if ind else None
            return (d["value"], d["delta"]) if d else (None, None)

        policy_rate, policy_rate_delta = _rate(policy_ind)
        yield_10y,   yield_10y_delta   = _rate(yield_ind)
        yield_2y,    yield_2y_delta    = _rate(yield2y_ind)

        bars, baseline = _combined_histbars(
            gdp["values"], gdp["years"],
            cpi["values"], cpi["years"],
        )
        rows.append({
            "country":    country,
            "gdp_value":  gdp_val,
            "gdp_year":   gdp_year,
            "gdp_chg":    gdp_chg,
            "cpi_value":  cpi_val,
            "cpi_year":   cpi_year,
            "cpi_chg":    cpi_chg,
            "unemp_value": unemp_val,
            "unemp_year":  unemp_year,
            "stock_1y":   stock_1y,
            "ca_value":   ca_val,
            "ca_year":    ca_year,
            "re_value":     re_val,
            "re_chg":       re_chg,
            "re_val_label": re_val_label,
            "policy_rate":       policy_rate,
            "policy_rate_delta": policy_rate_delta,
            "yield_10y":         yield_10y,
            "yield_10y_delta":   yield_10y_delta,
            "yield_2y":          yield_2y,
            "yield_2y_delta":    yield_2y_delta,
            "rate_spark_10y":    rate_sparks.get(yield_ind),
            "rate_spark_policy": rate_sparks.get(policy_ind) or rate_sparks.get(yield2y_ind),
            "detail_charts": _build_detail_charts(
                country, macro_df, gdp, cpi, unemp, ca,
                policy_ind, yield_ind, yield2y_ind
            ),
            "gdp_years":  gdp["years"],
            "gdp_values": gdp["values"],
            "cpi_years":  cpi["years"],
            "cpi_values": cpi["values"],
            "bars":       bars,
            "baseline":   baseline,
        })
    return rows


# Keep old name as alias for backward compatibility
def build_gdp_cpi_table(macro_df):
    return build_country_table(macro_df)


def build_capital_flow_cards(markets_cfg):
    """Load capital flow data for all markets and build sparkline cards."""
    cards = {}
    colors = _COMBINED_CHART_COLORS
    color_idx = 0
    for mkt in sorted(markets_cfg):
        path = get_data_path("markets", mkt, "capital_flow.parquet")
        if not os.path.exists(path):
            continue
        df = load_parquet(path)
        if df.empty:
            continue

        # Prefer net_flow_proxy (ETF-based), fall back to net_flow (direct)
        flow_col = "net_flow_proxy" if "net_flow_proxy" in df.columns else "net_flow" if "net_flow" in df.columns else None
        if not flow_col:
            continue
        df = df.sort_values("date").dropna(subset=[flow_col])
        if df.empty:
            continue

        vals = df[flow_col].tolist()
        dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

        # Last 30 points for sparkline
        spark_vals = vals[-30:]
        spark_dates = dates[-30:]
        latest = vals[-1]
        lo, hi = min(spark_vals), max(spark_vals)
        rng = (hi - lo) or 1

        # Sparkline points
        pts = []
        for i, v in enumerate(spark_vals):
            x = round(i / max(len(spark_vals) - 1, 1) * 80, 1)
            y = round((1 - (v - lo) / rng) * 24, 1)
            pts.append(f"{x},{y}")

        # Direction based on 5-day net sum (more stable than single day)
        net_5d = sum(vals[-5:]) if len(vals) >= 5 else sum(vals)

        color = colors[color_idx % len(colors)]
        color_idx += 1
        flow_type = df["flow_type"].iloc[-1] if "flow_type" in df.columns else "etf_proxy"
        label = "Northbound" if "northbound" in str(flow_type).lower() else f"ETF proxy ({flow_type.replace('etf_proxy_', '')})"
        cards[mkt] = {
            "market": mkt,
            "label": label,
            "color": color,
            "latest": round(latest, 0),
            "net_5d": round(net_5d, 0),
            "sparkline": " ".join(pts),
            "spark_dates": spark_dates,
            "spark_values": [round(v, 2) for v in spark_vals],
            "inflow": net_5d >= 0,
        }
    return cards


def get_markets_config():
    """Load markets configuration."""
    try:
        return load_yaml("markets.yaml")["markets"]
    except Exception:
        return {}


def get_latest_snapshot(market=None):
    """Find the most recent market snapshot file.

    If market is specified, looks in data/markets/{market}/.
    Otherwise looks in legacy data/snapshots/ location.
    """
    if market:
        pattern = os.path.join(
            get_data_path("markets", market, ""),
            "market_daily_*.parquet"
        )
    else:
        pattern = os.path.join(get_data_path("snapshots", ""), "market_daily_*.parquet")
    files = sorted(glob.glob(pattern))
    if files:
        return files[-1]
    return None


def _enrich_snapshot(df):
    """Recompute return_1d/5d/20d and volume_ratio from close history.

    Always recomputes rather than trusting stored values, because files written
    by different agent runs may lack the column, and concat fills those rows with
    NaN while leaving other tickers intact — the old 'isna().all()' guard then
    skips recomputation entirely, leaving affected tickers with missing returns.
    """
    if df.empty or "close" not in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ticker", "date"])
    grp = df.groupby("ticker", group_keys=False)
    df["return_1d"]  = grp["close"].transform(lambda s: s.pct_change(1, fill_method=None))
    df["return_5d"]  = grp["close"].transform(lambda s: s.pct_change(5, fill_method=None))
    df["return_20d"] = grp["close"].transform(lambda s: s.pct_change(20, fill_method=None))
    if "volume" in df.columns:
        df["volume_ratio"] = grp["volume"].transform(
            lambda s: s / s.rolling(20, min_periods=3).mean()
        )
    return df


def get_all_market_snapshots():
    """Load snapshots from all markets, merging recent files so a partial
    today-run doesn't lose tickers that were fetched yesterday."""
    markets_cfg = get_markets_config()
    all_data = []

    for market_code in markets_cfg:
        # Load up to last 3 snapshot files and keep the most recent row per ticker
        pattern = os.path.join(
            get_data_path("markets", market_code, ""),
            "market_daily_*.parquet"
        )
        files = sorted(glob.glob(pattern))[-3:]  # last 3 files
        if not files:
            continue
        frames = []
        for f in files:
            df = load_parquet(f)
            if not df.empty:
                frames.append(df)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        # Deduplicate same (date, ticker) that appears across overlapping snapshot files,
        # but keep all dates so callers can build sparkline history.
        combined = combined.sort_values("date").drop_duplicates(
            subset=["date", "ticker"], keep="last"
        )
        all_data.append(_enrich_snapshot(combined))

    # Fall back to legacy location
    if not all_data:
        snap = get_latest_snapshot()
        if snap:
            df = load_parquet(snap)
            if not df.empty:
                all_data.append(_enrich_snapshot(df))

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def get_latest_alerts():
    """Load alerts from global alerts or legacy location."""
    # Try new location first
    path = get_data_path("global", "alerts.json")
    if os.path.exists(path):
        return load_json(path)

    # Fall back to legacy snapshots
    pattern = os.path.join(get_data_path("snapshots", ""), "alerts_*.json")
    files = sorted(glob.glob(pattern))
    if files:
        return load_json(files[-1])
    return []


def get_master_universe():
    """Load master universe from new or legacy location."""
    path = get_data_path("global", "universe_master.parquet")
    if os.path.exists(path):
        return load_parquet(path)
    return load_parquet(get_data_path("processed", "tech_universe_master.parquet"))


def _get_yf_symbol(ticker, market, universe_df=None):
    """Resolve the yfinance symbol for a ticker.

    Checks the universe parquet for a yf_symbol column first,
    then falls back to market suffix from markets.yaml config.
    For CN stocks without a yf_symbol, constructs one based on
    ticker prefix (SZ for 0xx/3xx, SS for 6xx).
    """
    # Try universe yf_symbol column first
    if universe_df is not None and not universe_df.empty and "yf_symbol" in universe_df.columns:
        match = universe_df[universe_df["ticker"] == ticker]
        if not match.empty:
            yf_sym = match.iloc[0].get("yf_symbol")
            if yf_sym and str(yf_sym) not in ("", "nan"):
                return str(yf_sym)

    # CN stocks need special suffix logic
    if market == "CN":
        if ticker.startswith("6"):
            return f"{ticker}.SS"
        else:
            return f"{ticker}.SZ"

    # Use ticker_suffix from markets.yaml
    markets_cfg = get_markets_config()
    suffix = markets_cfg.get(market, {}).get("ticker_suffix", "")
    if suffix:
        return f"{ticker}{suffix}"
    return ticker


def _format_employees(count):
    """Format employee count with comma separators."""
    if not count:
        return None
    try:
        return f"{int(count):,}"
    except (ValueError, TypeError):
        return None


def _format_yf_market_cap(value):
    """Format market cap from yfinance (in raw number) to readable string."""
    if not value:
        return None
    try:
        v = float(value)
        if v >= 1e12:
            return f"${v / 1e12:.2f}T"
        elif v >= 1e9:
            return f"${v / 1e9:.2f}B"
        elif v >= 1e6:
            return f"${v / 1e6:.1f}M"
        else:
            return f"${v:,.0f}"
    except (ValueError, TypeError):
        return None


CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600  # 7 days


def fetch_company_info(ticker, market, universe_df=None):
    """Fetch company info from yfinance with file-based caching.

    Caches results to data/markets/{MARKET}/company_info/{ticker}.json
    for 7 days. Returns a dict with company profile fields, or an empty
    dict if yfinance fails.
    """
    cache_path = get_data_path("markets", market, "company_info", f"{ticker}.json")

    # Check cache
    if os.path.exists(cache_path):
        mtime = os.path.getmtime(cache_path)
        if (time.time() - mtime) < CACHE_MAX_AGE_SECONDS:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    # Fetch from yfinance
    try:
        import yfinance as yf
        yf_sym = _get_yf_symbol(ticker, market, universe_df)
        info = yf.Ticker(yf_sym).info or {}

        # Extract the fields we care about
        result = {
            "longBusinessSummary": info.get("longBusinessSummary"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "fullTimeEmployees": info.get("fullTimeEmployees"),
            "country": info.get("country"),
            "city": info.get("city"),
            "marketCap": info.get("marketCap"),
        }

        # Write cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return result
    except Exception:
        return {}


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    markets_cfg = get_markets_config()
    market_codes = list(markets_cfg.keys())

    # Alerts (threshold-based, excluding volume_spike which is shown separately)
    all_alerts = enrich_alerts(get_latest_alerts())
    _raw = [a for a in all_alerts if a.get("alert_type") != "volume_spike"]
    # Cap at 10 per market so every market is represented in the dashboard
    from collections import defaultdict
    _mkt_count = defaultdict(int)
    alerts = []
    for a in _raw:
        mkt = a.get("market", "")
        if _mkt_count[mkt] < 10:
            alerts.append(a)
            _mkt_count[mkt] += 1

    # Top Volume Ratio: top 10 per market ranked
    full = get_all_market_snapshots()
    market_df = pd.DataFrame()
    if not full.empty:
        name_lookup = load_name_lookup()
        latest = full.sort_values("date").groupby("ticker").tail(1)
        top_per_market = (
            latest[latest["volume_ratio"].notna()]
            .sort_values("volume_ratio", ascending=False)
            .groupby("market")
            .head(10)
            .copy()
        )
        top_per_market["vol_rank"] = (
            top_per_market.groupby("market")["volume_ratio"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        top_per_market["name"] = top_per_market.apply(
            lambda r: name_lookup.get(str(r["ticker"]), str(r["ticker"])), axis=1
        )
        market_df = top_per_market.sort_values(["market", "vol_rank"])

    # Peer mapping (try global first, fall back to legacy)
    peer_df = load_parquet(get_data_path("global", "peer_mapping.parquet"))
    if peer_df.empty:
        peer_df = load_parquet(get_data_path("processed", "peer_mapping.parquet"))

    # Universe counts
    universe = get_master_universe()
    market_counts = {}
    if not universe.empty:
        for m in universe["market"].unique():
            market_counts[m] = len(universe[universe["market"] == m])

    # Market stats — breadth, top movers, alert summary
    market_stats = {}
    if not full.empty:
        snap = full.sort_values("date").groupby("ticker").tail(1)
        # Filter to rows with valid return_1d
        r = snap[snap["return_1d"].notna() & snap["return_1d"].apply(lambda x: x == x)]
        gainers = r[r["return_1d"] > 0]
        losers  = r[r["return_1d"] < 0]
        n_up    = len(gainers)
        n_down  = len(losers)
        n_flat  = len(r) - n_up - n_down
        total_w_ret = n_up + n_down + n_flat
        adv_pct = round(n_up / total_w_ret * 100) if total_w_ret else 0

        # Top gainer / loser
        top_gainer = gainers.loc[gainers["return_1d"].idxmax()] if not gainers.empty else None
        top_loser  = losers.loc[losers["return_1d"].idxmin()]  if not losers.empty else None
        name_lookup = load_name_lookup()

        def _mover(row):
            if row is None:
                return None
            return {
                "ticker": row["ticker"],
                "market": row.get("market", ""),
                "return_1d": row["return_1d"],
                "name": name_lookup.get(row["ticker"], row["ticker"]),
            }

        # Per-market breadth
        breadth_by_market = {}
        for mkt, grp in r.groupby("market"):
            mu = int((grp["return_1d"] > 0).sum())
            md = int((grp["return_1d"] < 0).sum())
            tot = mu + md + int((grp["return_1d"] == 0).sum())
            breadth_by_market[mkt] = {
                "up": mu, "down": md,
                "pct": round(mu / tot * 100) if tot else 0,
            }

        # Alert type counts
        alert_type_counts = {}
        for a in all_alerts:
            t = a.get("alert_type", "other")
            alert_type_counts[t] = alert_type_counts.get(t, 0) + 1

        market_stats = {
            "n_up": n_up,
            "n_down": n_down,
            "n_flat": n_flat,
            "adv_pct": adv_pct,
            "top_gainer": _mover(top_gainer),
            "top_loser": _mover(top_loser),
            "breadth_by_market": breadth_by_market,
            "alert_type_counts": alert_type_counts,
            "total_alerts": len(all_alerts),
        }

    # Geopolitical news
    geo = load_json(get_data_path("global", "geopolitical_context.json"))
    if not geo:
        geo = load_json(get_data_path("processed", "geopolitical_context.json"))

    # News feed — collect from all markets + legacy location
    # Load all articles (not just hit_count > 0) so every market tab has content.
    # Sort by hit_count desc so keyword-matched articles surface first.
    news_items = []
    for market_code in market_codes:
        npath = get_data_path("markets", market_code, "news.parquet")
        if os.path.exists(npath):
            ndf = load_parquet(npath)
            if not ndf.empty:
                news_items.append(ndf)
    # Always try legacy location too (covers pre-agent markets)
    legacy_news = load_parquet(get_data_path("processed", "news_feed.parquet"))
    if not legacy_news.empty:
        news_items.append(legacy_news)

    if news_items:
        combined = pd.concat(news_items, ignore_index=True)
        if "hit_count" not in combined.columns:
            combined["hit_count"] = 0
        # Deduplicate by title, keep highest hit_count
        if "title" in combined.columns:
            combined = combined.sort_values("hit_count", ascending=False).drop_duplicates("title")
        # Per-market cap: at most 15 articles per market so no single market dominates
        combined = (
            combined.sort_values("hit_count", ascending=False)
            .groupby("market", group_keys=False)
            .head(15)
            .sort_values("hit_count", ascending=False)
        )
        news_keyword = combined.head(60)
    else:
        news_keyword = pd.DataFrame()

    # Macro latest (enriched with sparklines)
    macro_df = load_parquet(get_data_path("global", "macro_indicators.parquet"))
    macro_latest = build_macro_sparklines(
        load_json(get_data_path("global", "macro_latest.json")), macro_df
    )

    # Polymarket signals (top 5 by volume for dashboard)
    polymarket_signals = _get_polymarket_signals()[:5]

    # Sector heatmap data (meta-sectors for dashboard, full detail for /sectors)
    sector_perf = normalize_sector_df(load_parquet(get_data_path("global", "sector_performance.parquet")))
    heatmap_df = meta_sector_heatmap(sector_perf) if not sector_perf.empty else sector_perf
    # Only keep sectors present in ≥3 markets (single-market sectors add noise)
    if not heatmap_df.empty:
        market_count = heatmap_df.groupby("sector")["market"].nunique()
        keep = market_count[market_count >= 3].index
        heatmap_df = heatmap_df[heatmap_df["sector"].isin(keep)]
    heatmap_markets = sorted(heatmap_df["market"].unique().tolist(), key=_market_sort_key) if not heatmap_df.empty else []
    heatmap_sectors = sorted(heatmap_df["sector"].unique().tolist()) if not heatmap_df.empty else []
    heatmap = {}
    for row in heatmap_df.to_dict("records"):
        m, s, v = row["market"], row["sector"], row.get("avg_return_1d")
        cls = "up" if v and v > 0 else "down" if v and v < 0 else "flat"
        heatmap.setdefault(s, {})[m] = {"val": round(v * 100, 1) if v is not None else None, "cls": cls}

    # Market indices (from cached parquet files — live fetch via /api/indices)
    indices = load_cached_indices()

    snapshot_date = ""
    snap = get_latest_snapshot(market_codes[0] if market_codes else None)
    if not snap:
        snap = get_latest_snapshot()
    if snap:
        snapshot_date = os.path.basename(snap).replace("market_daily_", "").replace(".parquet", "")

    return render_template(
        "index.html",
        alerts=alerts,
        market=market_df.to_dict("records") if not market_df.empty else [],
        peers=peer_df.to_dict("records") if not peer_df.empty else [],
        market_counts=market_counts,
        market_stats=market_stats,
        market_codes=market_codes,
        markets_cfg=markets_cfg,
        geo=geo,
        geo_feeds=GEO_FEEDS,
        polymarket=polymarket_signals,
        news=news_keyword.to_dict("records") if not news_keyword.empty else [],
        macro_latest=macro_latest,
        sector_perf=sector_perf.to_dict("records") if not sector_perf.empty else [],
        heatmap_markets=heatmap_markets,
        heatmap_sectors=heatmap_sectors,
        heatmap=heatmap,
        indices=indices,
        snapshot_date=snapshot_date,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@app.route("/peers")
def peers_page():
    # Load global peer groups
    peer_groups = load_json(get_data_path("global", "peer_groups.json"))
    if isinstance(peer_groups, dict):
        peer_groups = []  # empty dict fallback

    # Enrich peer groups with latest return data
    full = get_all_market_snapshots()
    latest_returns = {}
    if not full.empty:
        last_day = full.sort_values("date").groupby("ticker").tail(1)
        for _, row in last_day.iterrows():
            latest_returns[str(row["ticker"])] = row.get("return_1d")

    all_markets = set()
    for group in peer_groups:
        for company in group.get("companies", []):
            all_markets.add(company.get("market", ""))
            company["return_1d"] = latest_returns.get(str(company.get("ticker")))

    # Also load legacy CN->US mapping for backward compat display
    legacy_groups = {}
    legacy_df = load_parquet(get_data_path("processed", "peer_mapping.parquet"))
    if not legacy_df.empty and "cn_name" in legacy_df.columns:
        for cn_name in legacy_df["cn_name"].unique():
            rows = legacy_df[legacy_df["cn_name"] == cn_name].sort_values("rank")
            legacy_groups[cn_name] = rows.to_dict("records")

    return render_template(
        "peers.html",
        peer_groups=peer_groups,
        all_markets=sorted(all_markets, key=_market_sort_key),
        legacy_groups=legacy_groups,
    )


@app.route("/market")
@app.route("/market/<market_code>")
def market_page(market_code=None):
    markets_cfg = get_markets_config()
    market_codes = list(markets_cfg.keys())

    if market_code and market_code not in markets_cfg:
        market_code = None

    # Default to first market if none selected
    active_market = market_code or (market_codes[0] if market_codes else None)

    # Load universe for sector/name info
    universe = get_master_universe()
    tags = {}
    try:
        tags_raw = load_yaml("company_tags.yaml")
        for cname, info in tags_raw.items():
            tags[info["ticker"]] = info
    except Exception:
        pass

    name_lookup = load_name_lookup()
    cap_lookup = load_market_caps()

    # Load market snapshot data
    full = get_all_market_snapshots()
    latest = {}
    ticker_history = {}
    if not full.empty:
        # Prefer rows with a valid close price so a partial today-run (NaN close)
        # doesn't shadow yesterday's complete data.
        full_valid = full[full["close"].notna()] if "close" in full.columns else full
        if full_valid.empty:
            full_valid = full
        last_day = full_valid.sort_values("date").groupby("ticker").tail(1)
        for _, row in last_day.iterrows():
            latest[row["ticker"]] = row.to_dict()
        for ticker, grp in full.groupby("ticker"):
            grp_sorted = grp.sort_values("date").tail(30)
            ticker_history[ticker] = {
                "sparkline_closes": grp_sorted["close"].dropna().tolist(),
                "daily": [
                    {
                        "date": str(r["date"])[:10] if r.get("date") is not None else "",
                        "close": round(float(r["close"]), 2) if pd.notna(r.get("close")) else None,
                        "return_1d": round(float(r["return_1d"]) * 100, 2) if pd.notna(r.get("return_1d")) else None,
                        "volume": int(r["volume"]) if pd.notna(r.get("volume")) else None,
                        "volume_ratio": round(float(r["volume_ratio"]), 2) if pd.notna(r.get("volume_ratio")) else None,
                    }
                    for _, r in grp_sorted.iterrows()
                ],
            }

    # Load financials snapshots per market (latest annual + latest quarterly per ticker)
    fin_by_ticker = {}  # ticker → {"annual": {...}, "quarter": {...}}
    _data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "markets")
    for mc in market_codes:
        fin_path = os.path.join(_data_root, mc, "financials.parquet")
        if not os.path.exists(fin_path):
            continue
        try:
            fin_df = pd.read_parquet(fin_path)
            if fin_df.empty or "ticker" not in fin_df.columns or "period_end" not in fin_df.columns:
                continue
            fin_df["period_end"] = pd.to_datetime(fin_df["period_end"], errors="coerce")
            fin_df = fin_df.dropna(subset=["period_end"])
            for tkr, grp in fin_df.groupby("ticker"):
                ann = grp[grp["period_type"] == "annual"].sort_values("period_end")
                qtr = grp[grp["period_type"] == "quarterly"].sort_values("period_end")
                fin_by_ticker[str(tkr)] = {
                    "annual":  ann.iloc[-1].to_dict() if not ann.empty else None,
                    "quarter": qtr.iloc[-1].to_dict() if not qtr.empty else None,
                }
        except Exception:
            pass

    # Build combined stocks list per market
    stocks_by_market = {}
    for mc in market_codes:
        stocks_by_market[mc] = []

    # Merge universe info with market data
    if not universe.empty:
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            market = row["market"]
            if market not in stocks_by_market:
                stocks_by_market[market] = []

            name = row["name"]
            if market == "US":
                name = US_STOCK_NAMES.get(ticker, name)
            elif name == ticker and ticker in name_lookup:
                # Only use the flat name_lookup as a fallback for placeholder names.
                # Unconditional override causes cross-market ticker collisions
                # (e.g. HK:2382 "Sunny Optical" being overwritten by TW:2382 "廣達").
                name = name_lookup[ticker]

            tag = tags.get(ticker, {})
            mkt = latest.get(ticker, {})
            mc_val = cap_lookup.get(ticker) or mkt.get("market_cap")

            fin = fin_by_ticker.get(str(ticker), {})
            fin_ann = fin.get("annual") or {}
            fin_qtr = fin.get("quarter") or {}
            stock = {
                "ticker": ticker,
                "name": name,
                "market": market,
                "sector": _SECTOR_MAP.get(str(row.get("sector", "") or "").strip(), str(row.get("sector", "") or "").strip()),
                "subsector": str(row.get("subsector", "") or ""),
                "sub_sector": tag.get("sub_sector", []),
                "board": get_board(ticker, market),
                "close": mkt.get("close"),
                "return_1d": mkt.get("return_1d"),
                "return_5d": mkt.get("return_5d"),
                "return_20d": mkt.get("return_20d"),
                "volume": mkt.get("volume"),
                "volume_ratio": mkt.get("volume_ratio"),
                "market_cap": mc_val,
                "sparkline": _sparkline(ticker_history.get(ticker, {}).get("sparkline_closes", []) if isinstance(ticker_history.get(ticker), dict) else ticker_history.get(ticker, [])),
                "history": ticker_history.get(ticker, {}).get("daily", []) if isinstance(ticker_history.get(ticker), dict) else [],
                # Financials
                "fin_ann_fy":      fin_ann.get("fiscal_year"),
                "fin_ann_rev":     fin_ann.get("revenue"),
                "fin_ann_rev_yoy": fin_ann.get("revenue_yoy"),
                "fin_ann_np":      fin_ann.get("net_profit"),
                "fin_ann_np_yoy":  fin_ann.get("net_profit_yoy"),
                "fin_qtr_end":     str(fin_qtr.get("period_end", ""))[:10] if fin_qtr else None,
                "fin_qtr_rev":     fin_qtr.get("revenue"),
                "fin_qtr_rev_yoy": fin_qtr.get("revenue_yoy"),
                "fin_qtr_np":      fin_qtr.get("net_profit"),
                "fin_qtr_np_yoy":  fin_qtr.get("net_profit_yoy"),
            }
            stocks_by_market[market].append(stock)
    else:
        # Fallback: use snapshot data only
        for ticker, mkt in latest.items():
            close_val = mkt.get("close")
            if close_val is None or pd.isna(close_val):
                continue
            market = mkt.get("market", "")
            if market not in stocks_by_market:
                stocks_by_market[market] = []
            name = name_lookup.get(ticker, ticker)
            if market == "US":
                name = US_STOCK_NAMES.get(ticker, name)
            mc_val = cap_lookup.get(ticker) or mkt.get("market_cap")
            stock = {
                "ticker": ticker,
                "name": name,
                "market": market,
                "sector": "",
                "sub_sector": [],
                "board": get_board(ticker, market),
                "close": mkt.get("close"),
                "return_1d": mkt.get("return_1d"),
                "return_5d": mkt.get("return_5d"),
                "return_20d": mkt.get("return_20d"),
                "volume": mkt.get("volume"),
                "volume_ratio": mkt.get("volume_ratio"),
                "market_cap": mc_val,
            }
            stocks_by_market[market].append(stock)

    # Compute market stats and top gainers/losers per market
    market_stats = {}
    top_gainers = {}
    top_losers = {}
    for mc in stocks_by_market:
        stocks = stocks_by_market[mc]
        total = len(stocks)
        total_cap = sum(s["market_cap"] for s in stocks if s.get("market_cap"))
        gainers_count = sum(1 for s in stocks if s.get("return_1d") and s["return_1d"] > 0)
        losers_count = sum(1 for s in stocks if s.get("return_1d") and s["return_1d"] < 0)

        market_stats[mc] = {
            "total_stocks": total,
            "total_market_cap": total_cap,
            "gainers_count": gainers_count,
            "losers_count": losers_count,
        }

        # Sort by return_1d for gainers/losers, attach sparkline
        with_returns = [s for s in stocks if s.get("return_1d") is not None
                        and s["return_1d"] == s["return_1d"]]
        for s in with_returns:
            th = ticker_history.get(s["ticker"], [])
            prices = th.get("sparkline_closes", []) if isinstance(th, dict) else th
            s["sparkline"] = _sparkline(prices)

        sorted_by_ret = sorted(with_returns, key=lambda s: s["return_1d"], reverse=True)
        top_gainers[mc] = sorted_by_ret[:5]
        sorted_asc = sorted(with_returns, key=lambda s: s["return_1d"])
        top_losers[mc] = sorted_asc[:5]

    # Market indices (from cached parquet files)
    indices = load_cached_indices()

    return render_template(
        "market.html",
        stocks_by_market=stocks_by_market,
        market_stats=market_stats,
        top_gainers=top_gainers,
        top_losers=top_losers,
        indices=indices,
        market_codes=market_codes,
        markets_cfg=markets_cfg,
        active_market=active_market,
    )


@app.route("/news")
def news_page():
    markets_cfg = get_markets_config()

    # Collect news from all markets + always include legacy
    all_news = []
    for market_code in markets_cfg:
        npath = get_data_path("markets", market_code, "news.parquet")
        if os.path.exists(npath):
            ndf = load_parquet(npath)
            if not ndf.empty:
                all_news.append(ndf)
    legacy_ndf = load_parquet(get_data_path("processed", "news_feed.parquet"))
    if not legacy_ndf.empty:
        all_news.append(legacy_ndf)

    if all_news:
        news_df = pd.concat(all_news, ignore_index=True)
        if "hit_count" not in news_df.columns:
            news_df["hit_count"] = 0
        if "title" in news_df.columns:
            news_df = news_df.sort_values("hit_count", ascending=False).drop_duplicates("title")
        news_df = news_df.sort_values("hit_count", ascending=False)
        # Enrich with company name from master universe
        universe = get_master_universe()
        if not universe.empty and "name" in universe.columns:
            name_map = dict(zip(universe["ticker"], universe["name"]))
            news_df["company_name"] = news_df["ticker"].map(name_map).fillna("")
        else:
            news_df["company_name"] = ""
    else:
        news_df = pd.DataFrame()

    geo = load_json(get_data_path("global", "geopolitical_context.json"))
    if not geo:
        geo = load_json(get_data_path("processed", "geopolitical_context.json"))

    return render_template(
        "news.html",
        news=news_df.to_dict("records") if not news_df.empty else [],
        geo=geo,
        market_codes=list(get_markets_config().keys()),
    )


@app.route("/companies")
def companies_page():
    """Redirect old companies page to unified market page."""
    return redirect(url_for("market_page"))


@app.route("/companies/<ticker>")
def company_detail(ticker):
    universe = get_master_universe()
    name_lookup = load_name_lookup()

    stock_row = None
    if not universe.empty:
        match = universe[universe["ticker"] == ticker]
        if not match.empty:
            stock_row = match.iloc[0]

    if stock_row is None:
        return render_template("company_detail.html", company=None, ticker=ticker)

    market = stock_row["market"]
    is_us = market == "US"
    display_name = US_STOCK_NAMES.get(ticker, ticker) if is_us else stock_row["name"]

    tags = load_yaml("company_tags.yaml")
    tag_info = {}
    for cname, info in tags.items():
        if info["ticker"] == ticker:
            tag_info = info
            break

    # Market data from market-specific snapshot
    snap = get_latest_snapshot(market)
    if not snap:
        snap = get_latest_snapshot()
    mkt_data = {}
    if snap:
        full = load_parquet(snap)
        if not full.empty:
            stock = full[full["ticker"] == ticker].sort_values("date")
            if not stock.empty:
                mkt_data = stock.iloc[-1].to_dict()

    cap_lookup = load_market_caps()
    mc = cap_lookup.get(ticker) or mkt_data.get("market_cap")

    # Try global peer mapping first, fall back to legacy
    peer_df = load_parquet(get_data_path("global", "peer_mapping.parquet"))
    peers = []
    if not peer_df.empty and "ticker" in peer_df.columns:
        peer_rows = peer_df[peer_df["ticker"] == ticker].sort_values("rank")
        peers = peer_rows.to_dict("records")
    else:
        peer_df = load_parquet(get_data_path("processed", "peer_mapping.parquet"))
        if not peer_df.empty:
            if not is_us:
                peer_rows = peer_df[peer_df["cn_ticker"] == ticker].sort_values("rank")
            else:
                peer_rows = peer_df[peer_df["us_ticker"] == ticker].sort_values("rank")
            peers = peer_rows.to_dict("records")

    # News
    company_news = []
    news_path = get_data_path("markets", market, "news.parquet")
    if os.path.exists(news_path):
        news_df = load_parquet(news_path)
    else:
        news_df = load_parquet(get_data_path("processed", "news_feed.parquet"))

    if not news_df.empty:
        ticker_news = news_df[news_df["ticker"] == ticker].sort_values(
            "hit_count" if "hit_count" in news_df.columns else "ticker",
            ascending=False
        ).head(20)
        company_news = ticker_news.to_dict("records")

    alerts = [a for a in enrich_alerts(get_latest_alerts()) if a.get("ticker") == ticker]

    # Fetch yfinance company profile info (cached)
    yf_info = fetch_company_info(ticker, market, universe)

    company = {
        "ticker": ticker,
        "name": display_name,
        "cn_name": stock_row["name"] if not is_us else None,
        "market": market,
        "sector": stock_row.get("sector", ""),
        "board": get_board(ticker, market),
        "sub_sector": tag_info.get("sub_sector", []),
        "customer_type": tag_info.get("customer_type", []),
        "revenue_model": tag_info.get("revenue_model", []),
        "close": mkt_data.get("close"),
        "return_1d": mkt_data.get("return_1d"),
        "return_5d": mkt_data.get("return_5d"),
        "return_20d": mkt_data.get("return_20d"),
        "volume": mkt_data.get("volume"),
        "volume_ratio": mkt_data.get("volume_ratio"),
        "market_cap": mc,
        "peers": peers,
    }

    # Build company profile from yfinance info
    company_profile = {}
    if yf_info:
        company_profile = {
            "description": yf_info.get("longBusinessSummary"),
            "sector": yf_info.get("sector"),
            "industry": yf_info.get("industry"),
            "website": yf_info.get("website"),
            "employees": _format_employees(yf_info.get("fullTimeEmployees")),
            "country": yf_info.get("country"),
            "city": yf_info.get("city"),
            "market_cap": _format_yf_market_cap(yf_info.get("marketCap")),
        }

    return render_template(
        "company_detail.html",
        company=company,
        company_profile=company_profile,
        news=company_news,
        alerts=alerts,
        ticker=ticker,
    )


@app.route("/macro")
def macro_page():
    macro_df = load_parquet(get_data_path("global", "macro_indicators.parquet"))

    # If yfinance indicators (VIX, Gold, Oil, etc.) are missing from the parquet
    # (e.g. the global agent's yfinance fetch failed last run), fall back to a
    # fresh live fetch so the macro page always shows commodities/sentiment.
    _yf_indicators = {"VIX", "Gold", "Oil WTI", "Copper", "Bitcoin"} | set(_FX_INDICATORS)
    _present = set(macro_df["indicator"].unique()) if not macro_df.empty else set()
    if not _yf_indicators.issubset(_present):
        try:
            from src.macro.commodities import fetch_all_yf_macro, get_macro_latest as _gml
            _yf_df = fetch_all_yf_macro(days=60)
            if not _yf_df.empty:
                macro_df = pd.concat([macro_df, _yf_df], ignore_index=True) if not macro_df.empty else _yf_df
        except Exception:
            pass

    macro_latest = build_macro_sparklines(
        load_json(get_data_path("global", "macro_latest.json")), macro_df
    )

    # Supplement macro_latest with any yfinance indicators still missing
    # (they won't be in macro_latest.json if the global agent failed to fetch them)
    _yf_missing = _yf_indicators - set(macro_latest.keys())
    if _yf_missing and not macro_df.empty:
        for ind in _yf_missing:
            sub = macro_df[macro_df["indicator"] == ind].sort_values("date")
            if sub.empty:
                continue
            row = sub.iloc[-1]
            macro_latest[ind] = {
                "value": round(float(row["value"]), 2) if pd.notna(row["value"]) else None,
                "change_pct": round(float(row["change_pct"]), 4) if "change_pct" in row and pd.notna(row.get("change_pct")) else None,
                "date": str(row["date"])[:10],
                "symbol": str(row.get("symbol", "")),
            }
        macro_latest = build_macro_sparklines(macro_latest, macro_df)

    # Combined country economic table (one row per country)
    country_table = build_country_table(macro_df)

    # Indicators excluded from individual cards:
    # - World Bank multi-country indicators (shown in the country table)
    # - FRED "US GDP" / "US CPI" / "US PCE" / "US Unemployment Rate" — shown in table
    # - FRED "US Case-Shiller HPI" — level index, shown as % change in country table
    _exclude_keys = {
        "US GDP", "US CPI", "US PCE", "US Unemployment Rate", "US Case-Shiller HPI",
        "Fed Funds Rate", "10Y Treasury Yield", "2Y Treasury Yield",
        "ECB Policy Rate", "UK Policy Rate",
        "Germany 10Y Yield", "UK 10Y Yield", "Japan 10Y Yield", "France 10Y Yield",
        "Australia 10Y Yield", "Korea 10Y Yield", "Brazil 10Y Yield",
        "Germany ST Rate", "UK ST Rate", "Japan ST Rate", "France ST Rate",
        "Australia ST Rate", "Korea ST Rate", "Brazil ST Rate", "India ST Rate",
    } | set(_FX_INDICATORS)
    _chart_keys = {"VIX", "Gold", "Oil WTI", "Copper", "Bitcoin"}
    macro_latest_filtered = {
        k: v for k, v in macro_latest.items()
        if "GDP growth (annual %)" not in k
        and "Inflation, consumer prices" not in k
        and "Unemployment, total" not in k
        and "Current account balance" not in k
        and k not in _exclude_keys
        and k not in _chart_keys
    }
    # 5 market indicators displayed as a row of bigger line charts
    macro_charts = {k: macro_latest[k] for k in _chart_keys if k in macro_latest}

    fx_table = build_fx_table(macro_df, macro_latest)
    markets_cfg = get_markets_config()
    capital_flows = build_capital_flow_cards(markets_cfg)
    polymarket = _get_polymarket_signals()
    return render_template(
        "macro.html",
        macro_latest=macro_latest_filtered,
        macro_charts=macro_charts,
        gdp_cpi_table=country_table,
        fx_table=fx_table,
        capital_flows=capital_flows,
        macro_data=macro_df.to_dict("records") if not macro_df.empty else [],
        polymarket=polymarket,
    )


@app.route("/sectors")
def sectors_page():
    sector_perf = normalize_sector_df(load_parquet(get_data_path("global", "sector_performance.parquet")))
    if not sector_perf.empty:
        sector_perf["_morder"] = sector_perf["market"].apply(_market_sort_key)
        sector_perf = sector_perf.sort_values(["_morder", "sector"]).drop(columns=["_morder"])
    correlations = load_json(get_data_path("global", "correlations.json"))
    markets_cfg = get_markets_config()
    return render_template(
        "sectors.html",
        sector_perf=sector_perf.to_dict("records") if not sector_perf.empty else [],
        correlations=correlations,
        market_codes=list(markets_cfg.keys()),
    )


@app.route("/alerts")
def alerts_page():
    all_alerts = enrich_alerts(get_latest_alerts())
    markets_cfg = get_markets_config()

    market_filter = request.args.get("market")
    if market_filter:
        all_alerts = [a for a in all_alerts if a.get("market") == market_filter]

    _market_order = list(markets_cfg.keys())
    _seen = set(a.get("market", "") for a in all_alerts if a.get("market"))
    alert_markets = [m for m in _market_order if m in _seen]

    return render_template(
        "alerts.html",
        alerts=all_alerts,
        alert_markets=alert_markets,
        market_codes=list(markets_cfg.keys()),
        active_market=market_filter,
    )


@app.route("/agents")
def agents_page():
    markets_cfg = get_markets_config()
    return render_template(
        "agents.html",
        market_codes=list(markets_cfg.keys()),
        markets_cfg=markets_cfg,
    )


@app.route("/status")
def status_page():
    status = load_json(get_data_path("agent_status.json"))
    return render_template("status.html", status=status)


# ── API Endpoints ───────────────────────────────────────────────────────


@app.route("/api/agent-status")
def api_agent_status():
    """JSON API for agent status — used by pixel agents UI.

    Returns sensible defaults when no pipeline has run yet, so the
    pixel agents page always shows meaningful content.
    """
    status = load_json(get_data_path("agent_status.json"))

    # If status file is empty or missing, build defaults from config
    if not status or "agents" not in status:
        markets_cfg = get_markets_config()
        market_codes = list(markets_cfg.keys())
        agents = {}
        for code in market_codes:
            agents[f"{code}_data"] = {
                "state": "idle",
                "role": "data",
                "market": code,
                "progress": f"Data agent for {code}",
            }
            agents[f"{code}_news"] = {
                "state": "idle",
                "role": "news",
                "market": code,
                "progress": f"News agent for {code}",
            }
            agents[f"{code}_signal"] = {
                "state": "idle",
                "role": "signal",
                "market": code,
                "progress": f"Signal agent for {code}",
            }
        agents["global"] = {
            "state": "idle",
            "role": "global",
            "market": "ALL",
            "progress": "Global strategist",
        }
        status = {
            "agents": agents,
            "pipeline": {
                "state": "idle",
                "completed_agents": 0,
                "total_agents": len(agents),
            },
        }
    else:
        # Ensure every known market agent has an entry even if status
        # file only has partial data (e.g. mid-run).
        markets_cfg = get_markets_config()
        agents = status.get("agents", {})
        pipeline_state = status.get("pipeline", {}).get("state", "")

        # If pipeline is not running, any "running"/"waiting" agents are stale
        if pipeline_state != "running":
            for agent in agents.values():
                if agent.get("state") in ("running", "waiting", "scheduled"):
                    agent["state"] = "idle"

        for code in markets_cfg:
            for role in ("data", "news", "signal"):
                key = f"{code}_{role}"
                if key not in agents:
                    agents[key] = {
                        "state": "idle",
                        "role": role,
                        "market": code,
                        "progress": "",
                    }
        if "global" not in agents:
            agents["global"] = {
                "state": "idle",
                "role": "global",
                "market": "ALL",
                "progress": "",
            }
        status["agents"] = agents

    return jsonify(status)


def _pid_is_alive(pid):
    """Return True if a process with this PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _clear_pipeline_status():
    """Force the status file pipeline state to idle."""
    status = load_json(get_data_path("agent_status.json")) or {}
    status.setdefault("pipeline", {})["state"] = "idle"
    try:
        with open(get_data_path("agent_status.json"), "w") as f:
            json.dump(status, f)
    except Exception:
        pass
    pid_path = get_data_path("pipeline.pid")
    if os.path.exists(pid_path):
        try:
            os.remove(pid_path)
        except Exception:
            pass


@app.route("/api/run-pipeline", methods=["POST"])
def api_run_pipeline():
    """Start the pipeline by running ./run.sh in a background subprocess."""
    import subprocess

    data = request.get_json() or {}
    scope = data.get("scope", "all")
    force = data.get("force", False)
    agent = data.get("agent", "")  # single agent name e.g. "global"

    # Check if actually running (verify process is alive, not just status file)
    pid_path = get_data_path("pipeline.pid")
    status = load_json(get_data_path("agent_status.json"))
    if status and status.get("pipeline", {}).get("state") == "running":
        # Double-check: is the process actually still alive?
        try:
            pid = int(open(pid_path).read().strip()) if os.path.exists(pid_path) else None
        except Exception:
            pid = None
        if pid and _pid_is_alive(pid):
            return jsonify({"error": "Pipeline is already running"})
        # Process is dead but status is stale — clear it and continue
        _clear_pipeline_status()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_script = os.path.join(project_dir, "run.sh")

    cmd = [run_script]
    if agent:
        cmd += ["--agent", agent]
    elif scope != "all":
        cmd += ["--market", scope]
    if force:
        cmd += ["--force"]

    proc = subprocess.Popen(
        cmd,
        cwd=project_dir,
        start_new_session=True,
        stdout=open(get_data_path("logs", f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"), "a"),
        stderr=subprocess.STDOUT,
    )
    # Save PID so stop endpoint can kill the process group
    pid_path = get_data_path("pipeline.pid")
    with open(pid_path, "w") as f:
        f.write(str(proc.pid))
    return jsonify({"status": "started", "scope": scope, "force": force})


@app.route("/api/stop-pipeline", methods=["POST"])
def api_stop_pipeline():
    """Kill the running pipeline process group."""
    import signal as _signal
    pid_path = get_data_path("pipeline.pid")
    kill_error = None
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.killpg(os.getpgid(pid), _signal.SIGTERM)
    except Exception as e:
        kill_error = str(e)
    # Always clear status so the next run isn't blocked
    _clear_pipeline_status()
    return jsonify({"status": "stopped", "kill_error": kill_error})


@app.route("/api/indices")
def api_indices():
    """JSON API for market indices."""
    market = request.args.get("market")
    markets = [market] if market else None
    return jsonify(fetch_indices(markets))


@app.route("/api/macro-latest")
def api_macro_latest():
    """JSON API for latest macro indicators."""
    return jsonify(load_json(get_data_path("global", "macro_latest.json")))


@app.route("/api/deep-analysis", methods=["POST"])
def api_deep_analysis():
    """Run TradingAgents multi-agent analysis on a ticker.

    Accepts {"ticker": "NVDA"} or {"ticker": "002049", "suffix": ".SZ"}
    Returns {"analysis": "...", "status": "ok"} or error.
    """
    from web.agent_llm import trading_agents_analyze

    body = request.get_json(force=True)
    ticker = body.get("ticker", "").strip().upper()
    suffix = body.get("suffix", "")

    if not ticker:
        return jsonify({"status": "error", "analysis": "No ticker provided."}), 400

    symbol = f"{ticker}{suffix}" if suffix else ticker
    analysis = trading_agents_analyze(symbol)

    if analysis:
        return jsonify({"status": "ok", "analysis": analysis})
    # analysis is None only when _get_trading_agents returns None after key check — shouldn't happen now
    return jsonify({"status": "ok", "analysis": (
        f"⚙ TradingAgents could not start for {symbol}.\n\n"
        "Make sure `tradingagents` is installed and your API key is configured in ⚙ settings."
    )})


_LLM_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm_config.json")

def _load_llm_config():
    try:
        with open(_LLM_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {"api_key": "", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"}

def _save_llm_config(cfg):
    os.makedirs(os.path.dirname(_LLM_CONFIG_PATH), exist_ok=True)
    with open(_LLM_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

@app.route("/api/llm-config", methods=["GET"])
def api_llm_config_get():
    cfg = _load_llm_config()
    # Mask key for display
    key = cfg.get("api_key", "")
    masked = (key[:8] + "..." + key[-4:]) if len(key) > 12 else ("*" * len(key) if key else "")
    return jsonify({"api_key_masked": masked, "base_url": cfg.get("base_url", ""), "model": cfg.get("model", "")})

@app.route("/api/llm-config", methods=["POST"])
def api_llm_config_save():
    body = request.get_json(force=True)
    cfg = _load_llm_config()
    if body.get("api_key"):
        cfg["api_key"] = body["api_key"]
    if body.get("base_url"):
        cfg["base_url"] = body["base_url"]
    if body.get("model"):
        cfg["model"] = body["model"]
    _save_llm_config(cfg)
    # Reset cached TradingAgents instance so it picks up the new key
    try:
        import web.agent_llm as _allm
        _allm._ta_instance = None
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/agent-chat", methods=["POST"])
def api_agent_chat():
    """Chat endpoint for pixel agent conversations.

    Uses LLM (OpenAI-compatible API) when available, falls back to keyword matching.
    Accepts {"agent": "CN_data", "message": "top movers", "history": [...]}
    """
    from web.agent_llm import agent_chat

    body = request.get_json(force=True)
    agent_name = body.get("agent", "")
    message = (body.get("message", "") or "").strip()
    history = body.get("history", [])
    language = body.get("language", "English")
    context = body.get("context", {})  # carries resolved ticker across turns

    if not agent_name:
        return jsonify({"response": "No agent specified."}), 400

    # Parse agent type and market from the name
    if agent_name == "global":
        agent_type = "global"
        market = None
    else:
        parts = agent_name.rsplit("_", 1)
        if len(parts) == 2:
            market, agent_type = parts[0], parts[1]
        else:
            return jsonify({"response": "Unknown agent format."}), 400

    # Try LLM-backed response first
    try:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_dir, "data")
        result = agent_chat(agent_type, market, message, data_dir, history,
                            language=language, context=context)
        # agent_chat returns (response, context) tuple
        if isinstance(result, tuple):
            llm_response, new_context = result
        else:
            llm_response, new_context = result, context
        if llm_response:
            return jsonify({"response": llm_response, "context": new_context})
    except Exception as e:
        app.logger.warning(f"LLM chat failed: {e}")

    # Fallback to keyword matching
    try:
        response = _build_agent_response(agent_type, market, message.lower())
    except Exception as e:
        response = f"Sorry, I encountered an error: {str(e)}"

    return jsonify({"response": response or "Sorry, no data available for this query."})


# ---------------------------------------------------------------------------
# Watchlist / Price-watch API
# ---------------------------------------------------------------------------

_WATCHLIST_PATH = None

def _get_watchlist_path():
    global _WATCHLIST_PATH
    if _WATCHLIST_PATH is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _WATCHLIST_PATH = os.path.join(project_dir, "data", "watchlist.json")
    return _WATCHLIST_PATH


def _load_watchlist():
    path = _get_watchlist_path()
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_watchlist(items):
    path = _get_watchlist_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


@app.route("/api/top-movers")
def api_top_movers():
    """Return top gainers and losers for a market (or all markets)."""
    market = request.args.get("market", "").upper()
    n = int(request.args.get("n", 5))

    markets_cfg = get_markets_config()
    all_markets = list(markets_cfg.keys()) if markets_cfg else _MARKET_ORDER
    if market and market in all_markets:
        target_markets = [market]
    else:
        target_markets = _MARKET_ORDER

    name_map = {}
    gainers = []
    losers = []

    for mc in target_markets:
        snap_path = get_latest_snapshot(mc)
        if not snap_path:
            continue
        try:
            df = pd.read_parquet(snap_path)
        except Exception:
            continue
        if "return_1d" not in df.columns:
            continue
        df = df[df["return_1d"].notna()]
        # Build name map from universe
        try:
            uni = pd.read_parquet(get_data_path("markets", mc, "universe.parquet"))
            for _, row in uni.iterrows():
                name_map[str(row["ticker"])] = row.get("name", "")
        except Exception:
            pass

        df = df.copy()
        df["market"] = mc
        df["name"] = df["ticker"].astype(str).map(name_map).fillna("")

        g = df[df["return_1d"] > 0].nlargest(n, "return_1d")
        l = df[df["return_1d"] < 0].nsmallest(n, "return_1d")

        for _, row in g.iterrows():
            gainers.append({
                "ticker": str(row["ticker"]),
                "name": str(row.get("name", "")),
                "market": mc,
                "return_1d": round(float(row["return_1d"]), 2),
                "close": round(float(row["close"]), 4) if pd.notna(row.get("close")) else None,
            })
        for _, row in l.iterrows():
            losers.append({
                "ticker": str(row["ticker"]),
                "name": str(row.get("name", "")),
                "market": mc,
                "return_1d": round(float(row["return_1d"]), 2),
                "close": round(float(row["close"]), 4) if pd.notna(row.get("close")) else None,
            })

    gainers.sort(key=lambda x: x["return_1d"], reverse=True)
    losers.sort(key=lambda x: x["return_1d"])
    return jsonify({"gainers": gainers[:n], "losers": losers[:n]})


@app.route("/api/watchlist", methods=["GET"])
def api_watchlist_get():
    return jsonify(_load_watchlist())


@app.route("/api/watchlist", methods=["POST"])
def api_watchlist_add():
    body = request.get_json(force=True)
    items = _load_watchlist()
    next_id = max((x.get("id", 0) for x in items), default=0) + 1
    item = {
        "id": next_id,
        "ticker": body.get("ticker", ""),
        "name": body.get("name", ""),
        "market": body.get("market", ""),
        "alert_price": float(body.get("alert_price", 0)),
        "direction": body.get("direction", "above"),  # "above" or "below"
        "ref_price": float(body.get("ref_price", 0)),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "triggered": False,
    }
    items.append(item)
    _save_watchlist(items)
    return jsonify({"ok": True, "id": next_id})


@app.route("/api/watchlist/<int:item_id>", methods=["DELETE"])
def api_watchlist_delete(item_id):
    items = _load_watchlist()
    items = [x for x in items if x.get("id") != item_id]
    _save_watchlist(items)
    return jsonify({"ok": True})


@app.route("/api/check-prices", methods=["POST"])
def api_check_prices():
    """Check current prices for watchlist items. Returns triggered alerts."""
    body = request.get_json(force=True)
    tickers = body.get("tickers", [])  # list of {ticker, market}
    if not tickers:
        return jsonify({"prices": {}})

    prices = {}

    # Group by market
    by_market = {}
    for t in tickers:
        mk = t.get("market", "US")
        by_market.setdefault(mk, []).append(t["ticker"])

    for mk, tkrs in by_market.items():
        markets_cfg = get_markets_config()
        suffix = markets_cfg.get(mk, {}).get("ticker_suffix", "")

        # Try yfinance first for non-CN
        if mk != "CN":
            try:
                import yfinance as yf
                syms = [t + suffix if not t.endswith(suffix) else t for t in tkrs]
                data = yf.download(syms, period="1d", progress=False, auto_adjust=True)
                close = data["Close"] if "Close" in data else data
                if hasattr(close, "columns"):
                    for sym, orig in zip(syms, tkrs):
                        if sym in close.columns:
                            val = close[sym].dropna()
                            if not val.empty:
                                prices[orig] = round(float(val.iloc[-1]), 4)
                else:
                    val = close.dropna()
                    if not val.empty and tkrs:
                        prices[tkrs[0]] = round(float(val.iloc[-1]), 4)
            except Exception as e:
                app.logger.warning(f"yfinance price check failed for {mk}: {e}")
        else:
            # CN: use akshare spot
            try:
                import akshare as ak
                spot = ak.stock_zh_a_spot_em()
                code_col = "代码"
                price_col = "最新价"
                for t in tkrs:
                    row = spot[spot[code_col] == t]
                    if not row.empty:
                        prices[t] = round(float(row.iloc[0][price_col]), 4)
            except Exception as e:
                app.logger.warning(f"akshare price check failed: {e}")

    return jsonify({"prices": prices})


@app.route("/api/email-config", methods=["GET"])
def api_email_config_get():
    """Return current email config (password masked)."""
    cfg = get_settings().get("email", {})
    return jsonify({
        "enabled":   cfg.get("enabled", False),
        "smtp_host": cfg.get("smtp_host", "smtp.gmail.com"),
        "smtp_port": cfg.get("smtp_port", 587),
        "smtp_user": cfg.get("smtp_user", ""),
        "smtp_password": "***" if cfg.get("smtp_password") else "",
        "recipient": cfg.get("recipient", ""),
    })


@app.route("/api/email-config", methods=["POST"])
def api_email_config_save():
    """Save email config to settings.yaml."""
    import yaml
    body = request.get_json(force=True)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings_path = os.path.join(project_dir, "config", "settings.yaml")
    with open(settings_path) as f:
        settings = yaml.safe_load(f) or {}
    if "email" not in settings:
        settings["email"] = {}
    settings["email"]["enabled"]       = bool(body.get("enabled", False))
    settings["email"]["smtp_host"]     = body.get("smtp_host", "smtp.gmail.com")
    settings["email"]["smtp_port"]     = int(body.get("smtp_port", 587))
    settings["email"]["smtp_user"]     = body.get("smtp_user", "")
    settings["email"]["recipient"]     = body.get("recipient", "")
    # Only update password if a real value was sent (not the masked placeholder)
    pwd = body.get("smtp_password", "")
    if pwd and pwd != "***":
        settings["email"]["smtp_password"] = pwd
    with open(settings_path, "w") as f:
        yaml.dump(settings, f, allow_unicode=True, default_flow_style=False)
    return jsonify({"ok": True})


@app.route("/api/notify-alert", methods=["POST"])
def api_notify_alert():
    """Send an email alert for a triggered price watch."""
    import smtplib
    from email.mime.text import MIMEText
    body = request.get_json(force=True)
    ticker    = body.get("ticker", "")
    market    = body.get("market", "")
    price     = body.get("price", "")
    direction = body.get("direction", "above")
    alert_price = body.get("alert_price", "")

    cfg = get_settings().get("email", {})
    if not cfg.get("enabled") or not cfg.get("smtp_user") or not cfg.get("smtp_password"):
        return jsonify({"ok": False, "reason": "email not configured"})

    dirStr = "≥" if direction == "above" else "≤"
    subject = f"[Price Alert] {ticker} ({market}) = {price}"
    text = (
        f"Your price alert was triggered:\n\n"
        f"  Ticker:  {ticker} ({market})\n"
        f"  Price:   {price}\n"
        f"  Alert:   {dirStr} {alert_price}\n\n"
        f"— Global Market Monitor"
    )
    msg = MIMEText(text)
    msg["Subject"] = subject
    msg["From"]    = cfg["smtp_user"]
    msg["To"]      = cfg.get("recipient") or cfg["smtp_user"]

    try:
        with smtplib.SMTP(cfg.get("smtp_host", "smtp.gmail.com"),
                          int(cfg.get("smtp_port", 587))) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.sendmail(cfg["smtp_user"], msg["To"], msg.as_string())
        return jsonify({"ok": True})
    except Exception as e:
        app.logger.warning(f"Email alert failed: {e}")
        return jsonify({"ok": False, "reason": str(e)})


# ---------------------------------------------------------------------------
# Backtest routes
# ---------------------------------------------------------------------------

@app.route("/backtest")
def backtest_page():
    markets_cfg  = get_markets_config()
    market_codes = sorted(markets_cfg.keys(), key=_market_sort_key) if markets_cfg else _MARKET_ORDER
    return render_template(
        "backtest.html",
        market_codes=market_codes,
        markets_cfg=markets_cfg,
    )


@app.route("/api/backtest/run", methods=["POST"])
def api_backtest_run():
    from src.backtest.engine import run_backtest

    body     = request.get_json(force=True)
    market   = body.get("market", "US").upper()
    timeframe = body.get("timeframe", "daily").lower()
    start_date = body.get("start_date", "2022-01-01")
    end_date   = body.get("end_date", "2024-12-31")
    force      = bool(body.get("force", False))

    universe_source = body.get("universe_source", {"type": "index", "key": "SP500"})
    signal          = body.get("signal", {"type": "builtin", "name": "volume_breakout"})

    # Legacy compatibility: if old-style "strategy" param is passed
    if "strategy" in body and "signal" not in body:
        signal = {"type": "builtin", "name": body["strategy"]}

    if timeframe not in ("daily", "weekly", "monthly", "yearly"):
        return jsonify({"error": f"Unknown timeframe {timeframe!r}"}), 400

    # Validate date format
    from datetime import datetime as _dt
    for _label, _val in (("start_date", start_date), ("end_date", end_date)):
        try:
            _dt.strptime(_val, "%Y-%m-%d")
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid {_label} format, expected YYYY-MM-DD"}), 400
    if start_date >= end_date:
        return jsonify({"error": "start_date must be before end_date"}), 400

    try:
        result = run_backtest(
            market=market,
            timeframe=timeframe,
            universe_source=universe_source,
            signal=signal,
            start_date=start_date,
            end_date=end_date,
            force=force,
        )
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Backtest failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/status/<market>/<timeframe>")
def api_backtest_status(market, timeframe):
    """
    Return cached result metadata (metrics only, no full equity curve/trades).
    Returns 404 if no cached result exists.
    """
    from src.backtest.engine import load_cached_result, _make_cache_key
    from src.backtest.strategies import DEFAULTS

    # Build cache key from query params
    universe_source = {"type": "index", "key": request.args.get("index_key", "SP500")}
    strategy_name   = request.args.get("strategy") or DEFAULTS.get(timeframe.lower())
    signal          = {"type": "builtin", "name": strategy_name}
    start_date      = request.args.get("start_date", "2022-01-01")
    end_date        = request.args.get("end_date", "2024-12-31")
    cache_key       = _make_cache_key(market.upper(), timeframe.lower(), universe_source, signal, start_date, end_date)

    result = load_cached_result(market.upper(), timeframe.lower(), cache_key, end_date)
    if result is None:
        return jsonify({"cached": False}), 404
    return jsonify({
        "cached":       True,
        "market":       result.get("market"),
        "timeframe":    result.get("timeframe"),
        "strategy":     result.get("strategy"),
        "metrics":      result.get("metrics"),
        "generated_at": result.get("generated_at"),
        "duration_s":   result.get("duration_s"),
    })


@app.route("/api/backtest/strategies")
def api_backtest_strategies():
    """Return available strategies per timeframe."""
    from src.backtest.strategies import strategies_for_timeframe, DEFAULTS
    return jsonify({
        tf: {
            "default":    DEFAULTS[tf],
            "strategies": strategies_for_timeframe(tf),
        }
        for tf in ("daily", "weekly", "monthly", "yearly")
    })


@app.route("/api/backtest/preview", methods=["POST"])
def api_backtest_preview():
    """
    Preview selected stocks from all active signal sources combined.
    Body: {"market": str, "signals": [{type, ...label}, ...]}
      signal types:
        builtin:        {type:"builtin", name: str|list, label: str}
        code:           {type:"code", code: str, label: str}
        custom_tickers: {type:"custom_tickers", tickers: [str,...], label: str}
    Returns: {"tickers": [{ticker, name, sector, strategies, signal_count}], "count": int}
    """
    from src.backtest.engine import _load_universe_df, _build_signal_fn
    from src.backtest.strategies import get_strategy
    from src.common.config import get_data_path
    import pandas as pd

    body    = request.get_json(force=True)
    market  = body.get("market", "US").upper()
    signals = body.get("signals", [])

    if not signals:
        return jsonify({"tickers": [], "count": 0, "error": "No signals provided"})

    try:
        universe_source = {"type": "market"}
        universe_df = _load_universe_df(universe_source, market)
        if universe_df.empty:
            return jsonify({"tickers": [], "count": 0,
                            "error": f"No universe data for {market}. Run the data pipeline first."})

        # Load history from market_daily snapshots — fast single parquet read, no network calls
        import glob as _glob
        daily_files = sorted(_glob.glob(
            os.path.join(get_data_path("markets", market), "market_daily_*.parquet")
        ))
        if not daily_files:
            return jsonify({"tickers": [], "count": 0,
                            "error": f"No market data for {market}. Run the data pipeline first."})

        market_df = pd.read_parquet(daily_files[-1])  # latest snapshot contains full rolling history
        history   = {tk: grp.reset_index(drop=True) for tk, grp in market_df.groupby("ticker")}
        signal_date = market_df["date"].max()

        # Run each signal source, collect {ticker -> [label,...]}
        ticker_strategies: dict = {}

        for sig in signals:
            stype = sig.get("type", "builtin")
            label = sig.get("label", stype)
            try:
                if stype == "builtin":
                    names = sig.get("name", [])
                    if isinstance(names, str):
                        names = [names]
                    for name in names:
                        s = get_strategy(name)
                        slabel = s.description or name
                        for tk in s.select(universe_df, history, signal_date):
                            ticker_strategies.setdefault(tk, []).append(slabel)

                elif stype == "code":
                    code = sig.get("code", "")
                    if code.strip():
                        signal_fn = _build_signal_fn({"type": "code", "code": code}, "daily")
                        for tk in signal_fn(universe_df, history, signal_date):
                            ticker_strategies.setdefault(tk, []).append(label)

                elif stype == "custom_tickers":
                    for tk in sig.get("tickers", []):
                        ticker_strategies.setdefault(tk.upper(), []).append(label)

            except Exception as e:
                app.logger.warning(f"Preview signal '{label}' failed: {e}")

        # Enrich with universe metadata
        meta = universe_df.set_index("ticker")
        result = []
        for tk, strats in ticker_strategies.items():
            row = meta.loc[tk] if tk in meta.index else {}
            result.append({
                "ticker":       tk,
                "name":         str(row.get("name",      "")) if hasattr(row, "get") else "",
                "sector":       str(row.get("sector",    "")) if hasattr(row, "get") else "",
                "subsector":    str(row.get("subsector", "")) if hasattr(row, "get") else "",
                "strategies":   strats,
                "signal_count": len(strats),
            })

        result.sort(key=lambda x: (-x["signal_count"], x["ticker"]))
        return jsonify({"tickers": result, "count": len(result),
                        "signal_date": signal_date.strftime("%Y-%m-%d")})

    except Exception as e:
        app.logger.warning(f"Preview failed: {e}")
        return jsonify({"tickers": [], "count": 0, "error": str(e)})


@app.route("/api/backtest/generate-signal", methods=["POST"])
def api_backtest_generate_signal():
    """
    Generate select() code from natural language description.
    Body: {"description": "select top 10 stocks by 20-day momentum"}
    Returns: {"code": str, "explanation": str, "success": bool, "error": str}
    """
    from src.backtest.signal_builder import generate_signal_code

    body = request.get_json(force=True)
    description = body.get("description", "").strip()
    if not description:
        return jsonify({"success": False, "error": "No description provided"}), 400

    llm_config = _load_llm_config()
    if not llm_config.get("api_key"):
        return jsonify({"success": False, "error": "LLM API key not configured. Please set it in Settings."}), 400

    result = generate_signal_code(description, llm_config)
    return jsonify(result)


@app.route("/api/backtest/validate-signal", methods=["POST"])
def api_backtest_validate_signal():
    """
    Validate a select() function by running it on a small test dataset.
    Body: {"code": str, "market": str}
    Returns: {"valid": bool, "tickers": [...], "error": str}
    """
    from src.backtest.signal_builder import validate_select_fn
    from src.backtest.data_loader import load_universe
    from src.common.config import get_data_path
    import glob as _glob

    body   = request.get_json(force=True)
    code   = body.get("code", "")
    market = body.get("market", "US").upper()

    try:
        universe_df = load_universe(market)
        if universe_df.empty:
            return jsonify({"valid": False, "error": f"No universe data for {market}"})

        # Use market_daily snapshot — fast, no network calls
        daily_files = sorted(_glob.glob(
            os.path.join(get_data_path("markets", market), "market_daily_*.parquet")
        ))
        if not daily_files:
            return jsonify({"valid": False, "error": f"No market data for {market}. Run the data pipeline first."})

        market_df = pd.read_parquet(daily_files[-1])
        history   = {tk: grp.reset_index(drop=True) for tk, grp in market_df.groupby("ticker")}
        test_date = market_df["date"].max()

        result = validate_select_fn(code, universe_df, history, test_date)
        return jsonify(result)
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)})


def _build_agent_response(agent_type, market, message):
    """Route to the correct agent response builder."""
    if agent_type == "market":
        try:
            return _signal_agent_response(market, message)
        except Exception:
            return _data_agent_response(market, message)
    elif agent_type == "data":
        return _data_agent_response(market, message)
    elif agent_type == "news":
        return _news_agent_response(market, message)
    elif agent_type == "signal":
        return _signal_agent_response(market, message)
    elif agent_type == "global":
        return _global_agent_response(message)
    else:
        return f"Unknown agent type: {agent_type}"


def _get_name_lookup(market):
    """Load ticker→name mapping from universe file."""
    try:
        uni = pd.read_parquet(get_data_path("markets", market, "universe.parquet"))
        if "name" in uni.columns:
            return dict(zip(uni["ticker"], uni["name"]))
    except Exception:
        pass
    return {}


def _data_agent_response(market, message):
    """Data Agent: precise, numbers-focused, professional."""
    base_path = get_data_path("markets", market, "")
    names = _get_name_lookup(market)

    # Status / greeting
    if any(kw in message for kw in ["status", "how are you"]):
        status = load_json(get_data_path("agent_status.json"))
        agent_info = status.get("agents", {}).get(f"{market}_data", {})
        state = agent_info.get("state", "unknown")
        progress = agent_info.get("progress", "-")
        return (
            f"[{market} Data Agent — Status Report]\n"
            f"State: {state}\n"
            f"Progress: {progress}\n"
            f"All systems nominal. Standing by for data queries."
        )

    # Top movers / gainers / losers
    if any(kw in message for kw in ["top movers", "gainers", "losers", "movers"]):
        snap = get_latest_snapshot(market)
        if not snap:
            return f"No market snapshot available for {market} yet."
        df = load_parquet(snap)
        if df.empty:
            return f"No data in latest snapshot for {market}."
        latest = df.sort_values("date").groupby("ticker").tail(1)
        if "return_1d" not in latest.columns:
            return "Return data not available."
        if "losers" in message:
            sorted_df = latest.sort_values("return_1d", ascending=True).head(10)
            label = "Bottom Losers"
        else:
            sorted_df = latest.sort_values("return_1d", ascending=False).head(10)
            label = "Top Gainers"
        lines = [f"[{market} — {label} Today]"]
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            ticker = row["ticker"]
            name = names.get(ticker, "")
            ret = row.get("return_1d", 0) or 0
            vr = row.get("volume_ratio", 0) or 0
            close = row.get("close", 0) or 0
            name_str = f" {name}" if name else ""
            lines.append(
                f"{i}. {ticker}{name_str}  {ret:+.2%}  "
                f"close {close:.2f}  vol {vr:.1f}x"
            )
        return "\n".join(lines)

    # Volume / most active
    if any(kw in message for kw in ["volume", "active"]):
        snap = get_latest_snapshot(market)
        if not snap:
            return f"No market snapshot available for {market}."
        df = load_parquet(snap)
        if df.empty:
            return "No data available."
        latest = df.sort_values("date").groupby("ticker").tail(1)
        if "volume_ratio" not in latest.columns:
            return "Volume ratio data not available."
        sorted_df = latest.sort_values("volume_ratio", ascending=False).head(10)
        lines = [f"[{market} — Most Active by Volume Ratio]"]
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            ticker = row["ticker"]
            name = names.get(ticker, "")
            vr = row.get("volume_ratio", 0) or 0
            ret = row.get("return_1d", 0) or 0
            name_str = f" {name}" if name else ""
            lines.append(
                f"{i}. {ticker}{name_str}  vol {vr:.1f}x avg  {ret:+.2%}"
            )
        return "\n".join(lines)

    # Market cap / biggest
    if any(kw in message for kw in ["market cap", "biggest", "largest"]):
        cap_path = os.path.join(base_path, "market_cap.parquet")
        df = load_parquet(cap_path)
        if df.empty:
            return f"No market cap data for {market}."
        if "market_cap" not in df.columns:
            return "Market cap column not found."
        sorted_df = df.dropna(subset=["market_cap"]).sort_values(
            "market_cap", ascending=False
        ).head(10)
        lines = [f"[{market} — Largest by Market Cap]"]
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            ticker = row["ticker"]
            name = names.get(ticker, "")
            mc = row["market_cap"]
            mc_str = format_market_cap(mc) if mc else "N/A"
            name_str = f" {name}" if name else ""
            lines.append(f"{i}. {ticker}{name_str}  {mc_str}")
        return "\n".join(lines)

    # Indices
    if any(kw in message for kw in ["indices", "index"]):
        idx_path = os.path.join(base_path, "indices.parquet")
        df = load_parquet(idx_path)
        if df.empty:
            return f"No index data for {market}."
        latest = df.sort_values("date").groupby("name").tail(1)
        lines = [f"[{market} — Market Indices]"]
        for _, row in latest.iterrows():
            name = row.get("name", row.get("symbol", "?"))
            close = row.get("close", 0) or 0
            chg = row.get("change_pct", 0) or 0
            lines.append(f"  {name}: {close:,.2f}  ({chg:+.2%})")
        return "\n".join(lines)

    # Universe / stocks / how many
    if any(kw in message for kw in ["universe", "stocks", "how many"]):
        uni_path = os.path.join(base_path, "universe.parquet")
        df = load_parquet(uni_path)
        if df.empty:
            return f"No universe data for {market}."
        total = len(df)
        lines = [f"[{market} — Universe: {total} stocks]"]
        if "sector" in df.columns:
            sector_counts = df["sector"].value_counts()
            for sector, count in sector_counts.items():
                if sector and str(sector) != "nan":
                    lines.append(f"  {sector}: {count}")
        return "\n".join(lines)

    # Check if user is asking about a specific ticker
    # Try to match any word in the message to a known ticker
    words = message.upper().replace(",", " ").split()
    for w in words:
        if w in names or w in {t.upper() for t in names}:
            ticker = w
            # Find this ticker's data
            snap = get_latest_snapshot(market)
            if snap:
                df = load_parquet(snap)
                if not df.empty:
                    latest = df.sort_values("date").groupby("ticker").tail(1)
                    match = latest[latest["ticker"].str.upper() == ticker.upper()]
                    if not match.empty:
                        row = match.iloc[0]
                        name = names.get(row["ticker"], "")
                        ret = row.get("return_1d", 0) or 0
                        vr = row.get("volume_ratio", 0) or 0
                        close = row.get("close", 0) or 0
                        vol = row.get("volume", 0) or 0
                        lines = [f"[{market} — {row['ticker']} {name}]"]
                        lines.append(f"  Close: {close:.2f}")
                        lines.append(f"  Return 1d: {ret:+.2%}")
                        lines.append(f"  Volume ratio: {vr:.1f}x")
                        if vol:
                            lines.append(f"  Volume: {vol:,.0f}")
                        r5 = row.get("return_5d")
                        r20 = row.get("return_20d")
                        if r5 is not None:
                            lines.append(f"  Return 5d: {r5:+.2%}")
                        if r20 is not None:
                            lines.append(f"  Return 20d: {r20:+.2%}")
                        return "\n".join(lines)

    # Default: show a quick summary with names
    snap = get_latest_snapshot(market)
    if not snap:
        return (
            f"[{market} Data Agent] No data yet — run the pipeline first.\n"
            f"Try: top movers, losers, volume, market cap, indices"
        )
    df = load_parquet(snap)
    if df.empty:
        return f"[{market} Data Agent] No data available."
    latest = df.sort_values("date").groupby("ticker").tail(1)
    n = latest["ticker"].nunique()
    lines = [f"[{market} Data Agent — {n} stocks tracked]"]
    if "return_1d" in latest.columns:
        valid = latest.dropna(subset=["return_1d"])
        if not valid.empty:
            avg = valid["return_1d"].mean()
            up = (valid["return_1d"] > 0).sum()
            down = (valid["return_1d"] < 0).sum()
            lines.append(f"  Market avg: {avg:+.2%} | {up} up / {down} down")
            top3 = valid.nlargest(3, "return_1d")
            bot3 = valid.nsmallest(3, "return_1d")
            top_str = ", ".join(
                f"{r['ticker']} {names.get(r['ticker'], '')} {r['return_1d']:+.1%}"
                for _, r in top3.iterrows()
            )
            bot_str = ", ".join(
                f"{r['ticker']} {names.get(r['ticker'], '')} {r['return_1d']:+.1%}"
                for _, r in bot3.iterrows()
            )
            lines.append(f"  Top: {top_str}")
            lines.append(f"  Bottom: {bot_str}")
    lines.append("")
    lines.append("Ask: top movers, losers, volume, market cap, indices, or any ticker code")
    return "\n".join(lines)


def _news_agent_response(market, message):
    """News Agent: informative, headline-style, journalist-like."""
    news_path = get_data_path("markets", market, "news.parquet")
    df = load_parquet(news_path)

    # Search keyword
    if message.startswith("search "):
        keyword = message[7:].strip()
        if df.empty:
            return f"No news archive for {market} to search."
        if "title" in df.columns:
            matches = df[df["title"].str.contains(keyword, case=False, na=False)]
        elif "headline" in df.columns:
            matches = df[df["headline"].str.contains(keyword, case=False, na=False)]
        else:
            return "News data format not recognized."
        if matches.empty:
            return f'BREAKING: No results found for "{keyword}" in {market} news.'
        matches = matches.head(10)
        title_col = "title" if "title" in matches.columns else "headline"
        lines = [f'[{market} NEWS — Search: "{keyword}" — {len(matches)} results]']
        for _, row in matches.iterrows():
            ticker = row.get("ticker", "")
            lines.append(f"  >> {row[title_col]}  [{ticker}]")
        return "\n".join(lines)

    # Trending / hot
    if any(kw in message for kw in ["trending", "hot"]):
        if df.empty:
            return f"No news data for {market} yet. Check back later."
        if "hit_count" not in df.columns:
            return "Keyword hit data not available."
        sorted_df = df.sort_values("hit_count", ascending=False).head(10)
        title_col = "title" if "title" in sorted_df.columns else "headline"
        lines = [f"[{market} — TRENDING NOW]"]
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            title = row.get(title_col, "untitled")
            hits = row.get("hit_count", 0)
            ticker = row.get("ticker", "")
            lines.append(f"{i}. {title}  [{ticker}] ({hits} keyword hits)")
        return "\n".join(lines)

    # Latest news / headlines
    if any(kw in message for kw in ["news", "headlines", "latest"]):
        if df.empty:
            return f"No news for {market} yet. The wire is quiet."
        title_col = "title" if "title" in df.columns else "headline"
        recent = df.head(10) if "date" not in df.columns else df.sort_values(
            "date", ascending=False
        ).head(10)
        lines = [f"[{market} — Latest Headlines]"]
        for i, (_, row) in enumerate(recent.iterrows(), 1):
            title = row.get(title_col, "untitled")
            ticker = row.get("ticker", "")
            lines.append(f"{i}. {title}  [{ticker}]")
        return "\n".join(lines)

    # Default: auto-show latest headlines summary
    if df.empty:
        return f"[{market} News Agent] No news data yet — run the pipeline first.\nTry: latest news, trending, search <keyword>"
    total = len(df)
    hits = df[df["hit_count"] > 0] if "hit_count" in df.columns else df.head(0)
    title_col = "title" if "title" in df.columns else "headline"
    lines = [f"[{market} News Agent — {total} articles, {len(hits)} with keyword matches]"]
    # Show top 5 headlines
    recent = df.sort_values("date", ascending=False).head(5) if "date" in df.columns else df.head(5)
    for _, row in recent.iterrows():
        ticker = row.get("ticker", "")
        title = str(row.get(title_col, ""))[:70]
        lines.append(f"  {ticker}: {title}")
    lines.append("")
    lines.append("Ask: latest news, trending, or search <keyword>")
    return "\n".join(lines)


def _signal_agent_response(market, message):
    """Signal Agent: synthesizes data + news + alerts into briefings."""
    base_path = get_data_path("markets", market, "")

    # Market summary / briefing — synthesize data + news + alerts
    if any(kw in message for kw in ["summary", "briefing", "brief", "overview", "report"]):
        lines = [f"[{market} — MARKET BRIEFING]", ""]

        # 1. Price data summary
        today = datetime.now().strftime("%Y%m%d")
        snap = load_parquet(
            get_data_path("markets", market, f"market_daily_{today}.parquet")
        )
        if not snap.empty and "return_1d" in snap.columns:
            # Deduplicate to latest row per ticker (parquet may contain multi-day rows)
            if "ticker" in snap.columns and "date" in snap.columns:
                snap = snap.sort_values("date").drop_duplicates("ticker", keep="last")
            valid = snap.dropna(subset=["return_1d"])
            if not valid.empty:
                avg_ret = valid["return_1d"].mean()
                up = (valid["return_1d"] > 0).sum()
                down = (valid["return_1d"] < 0).sum()
                top = valid.nlargest(3, "return_1d")
                bot = valid.nsmallest(3, "return_1d")
                lines.append(f"PRICES: {len(valid)} stocks | avg {avg_ret:+.2%} | {up} up / {down} down")
                top_str = ", ".join(
                    f"{r['ticker']} {r['return_1d']:+.1%}" for _, r in top.iterrows()
                )
                bot_str = ", ".join(
                    f"{r['ticker']} {r['return_1d']:+.1%}" for _, r in bot.iterrows()
                )
                lines.append(f"  Top: {top_str}")
                lines.append(f"  Bottom: {bot_str}")
        else:
            lines.append("PRICES: No data yet — run pipeline first")

        lines.append("")

        # 2. News summary
        news_df = load_parquet(get_data_path("markets", market, "news.parquet"))
        if not news_df.empty:
            total = len(news_df)
            hits = (
                news_df[news_df["hit_count"] > 0]
                if "hit_count" in news_df.columns
                else news_df.head(0)
            )
            lines.append(f"NEWS: {total} articles, {len(hits)} keyword matches")
            if not hits.empty:
                top_news = hits.nlargest(3, "hit_count")
                for _, row in top_news.iterrows():
                    t = row.get("ticker", "")
                    title = str(row.get("title", ""))[:60]
                    kw = row.get("keywords_matched", "")
                    lines.append(f"  {t}: {title}")
                    if kw:
                        lines.append(f"       keywords: {kw}")
        else:
            lines.append("NEWS: No news data yet")

        lines.append("")

        # 3. Alerts summary
        alerts = load_json(os.path.join(base_path, "alerts.json"))
        if alerts and isinstance(alerts, list):
            by_type = {}
            for a in alerts:
                t = a.get("alert_type", "unknown")
                by_type.setdefault(t, []).append(a)
            lines.append(f"ALERTS: {len(alerts)} active")
            for atype, items in by_type.items():
                lines.append(f"  {atype}: {len(items)}")
                for a in items[:3]:
                    ticker = a.get("ticker", "")
                    signal = a.get("signal", "")
                    if signal:
                        lines.append(f"    {signal}")
                    else:
                        ret = a.get("return_1d")
                        vr = a.get("volume_ratio")
                        d = ""
                        if ret is not None:
                            d += f" {ret:+.2%}"
                        if vr is not None:
                            d += f" vol {vr:.1f}x"
                        lines.append(f"    {ticker}{d}")
        else:
            lines.append("ALERTS: All clear, no active alerts")

        # 4. Capital flow
        flow_df = load_parquet(os.path.join(base_path, "capital_flow.parquet"))
        if not flow_df.empty and "date" in flow_df.columns:
            latest = flow_df.sort_values("date").iloc[-1]
            net = latest.get("net_flow", latest.get("value", "N/A"))
            lines.append("")
            lines.append(f"CAPITAL FLOW: latest net {net}")

        return "\n".join(lines)

    # Alerts / signals
    if any(kw in message for kw in ["alerts", "signals"]):
        alerts_path = os.path.join(base_path, "alerts.json")
        alerts = load_json(alerts_path)
        if not alerts or not isinstance(alerts, list):
            return f"[{market} SIGNAL] All clear. No active alerts detected."
        lines = [f"[{market} — ACTIVE ALERTS: {len(alerts)}]"]
        for a in alerts[:15]:
            atype = a.get("alert_type", "unknown")
            ticker = a.get("ticker", "?")
            signal = a.get("signal", "")
            if signal:
                lines.append(f"  !! {atype.upper()}: {signal}")
            else:
                ret = a.get("return_1d")
                vr = a.get("volume_ratio")
                detail = ""
                if ret is not None:
                    detail += f" ret {ret:+.2%}"
                if vr is not None:
                    detail += f" vol {vr:.1f}x"
                lines.append(f"  !! {atype.upper()}: {ticker}{detail}")
        return "\n".join(lines)

    # Volume spikes specifically
    if any(kw in message for kw in ["volume spike", "volume spikes"]):
        alerts_path = os.path.join(base_path, "alerts.json")
        alerts = load_json(alerts_path)
        if not alerts or not isinstance(alerts, list):
            return f"[{market} SIGNAL] No volume spike alerts active."
        vol_alerts = [a for a in alerts if "volume" in a.get("alert_type", "")]
        if not vol_alerts:
            return f"[{market} SIGNAL] No volume spikes detected. Market is calm."
        lines = [f"[{market} — VOLUME SPIKE ALERTS: {len(vol_alerts)}]"]
        for a in vol_alerts[:10]:
            ticker = a.get("ticker", "?")
            vr = a.get("volume_ratio", 0)
            ret = a.get("return_1d", 0) or 0
            lines.append(
                f"  !! {ticker}  vol {vr:.1f}x average  {ret:+.2%}"
            )
        return "\n".join(lines)

    # Capital flow
    if any(kw in message for kw in ["flow", "capital"]):
        flow_path = os.path.join(base_path, "capital_flow.parquet")
        df = load_parquet(flow_path)
        if df.empty:
            return f"[{market} SIGNAL] No capital flow data available."
        latest = df.sort_values("date").tail(5) if "date" in df.columns else df.tail(5)
        lines = [f"[{market} — Capital Flow (Latest)]"]
        for _, row in latest.iterrows():
            date = row.get("date", "")
            net = row.get("net_flow", row.get("value", "N/A"))
            flow_type = row.get("flow_type", "")
            lines.append(f"  {date}  net: {net}  {flow_type}")
        return "\n".join(lines)

    # Default: auto-show a quick briefing
    return _signal_agent_response(market, "summary")


def _global_agent_response(message):
    """Global Agent: big-picture, strategic, analytical."""

    # Macro / VIX / rates
    if any(kw in message for kw in ["macro", "vix", "rates", "economy"]):
        macro = load_json(get_data_path("global", "macro_latest.json"))
        if not macro:
            return "[Global Strategist] Macro data not yet available."
        lines = ["[Global — Macro Snapshot]"]
        for key, val in macro.items():
            if isinstance(val, dict):
                name = val.get("name", key)
                value = val.get("value", "N/A")
                change = val.get("change_pct")
                chg_str = f"  ({change:+.2%})" if change else ""
                lines.append(f"  {name}: {value}{chg_str}")
            else:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    # Sectors
    if any(kw in message for kw in ["sectors", "sector"]):
        df = load_parquet(get_data_path("global", "sector_performance.parquet"))
        if df.empty:
            return "[Global Strategist] Sector data not yet computed."
        lines = ["[Global — Sector Performance Overview]"]
        display = df.head(15)
        for _, row in display.iterrows():
            sector = row.get("sector", "?")
            mkt = row.get("market", "")
            ret = row.get("avg_return_1d", row.get("return_1d", 0)) or 0
            count = row.get("count", "")
            count_str = f"  ({count} stocks)" if count else ""
            lines.append(f"  {sector} [{mkt}]: {ret:+.2%}{count_str}")
        return "\n".join(lines)

    # Correlations
    if "correlation" in message:
        corr = load_json(get_data_path("global", "correlations.json"))
        if not corr:
            return "[Global Strategist] Correlation data not available."
        lines = ["[Global — Cross-Market Correlations]"]
        if isinstance(corr, dict):
            for key, val in list(corr.items())[:10]:
                if isinstance(val, (int, float)):
                    lines.append(f"  {key}: {val:.3f}")
                else:
                    lines.append(f"  {key}: {val}")
        elif isinstance(corr, list):
            for item in corr[:10]:
                lines.append(f"  {item}")
        return "\n".join(lines)

    # Alerts
    if any(kw in message for kw in ["alerts", "alert"]):
        alerts = load_json(get_data_path("global", "alerts.json"))
        if not alerts or not isinstance(alerts, list):
            return "[Global Strategist] No global alerts active. Markets are stable."
        lines = [f"[Global — Active Alerts: {len(alerts)}]"]
        for a in alerts[:15]:
            atype = a.get("alert_type", "unknown")
            signal = a.get("signal", "")
            ticker = a.get("ticker", "")
            if signal:
                lines.append(f"  >> {atype}: {signal}")
            else:
                lines.append(f"  >> {atype}: {ticker}")
        return "\n".join(lines)

    # Peers
    if "peer" in message:
        peers = load_json(get_data_path("global", "peer_groups.json"))
        if not peers:
            return "[Global Strategist] Peer group data not available."
        if isinstance(peers, list):
            lines = [f"[Global — Peer Groups: {len(peers)} groups]"]
            for g in peers[:8]:
                name = g.get("group_name", g.get("name", "?"))
                companies = g.get("companies", [])
                tickers = [c.get("ticker", "?") for c in companies[:5]]
                lines.append(f"  {name}: {', '.join(tickers)}")
        else:
            lines = ["[Global — Peer Groups]", f"  {peers}"]
        return "\n".join(lines)

    # Default
    return (
        "[Global Strategist] Here's a quick overview:\n"
    )
    # Auto-show macro + alerts summary
    parts = [result]
    macro = load_json(get_data_path("global", "macro_latest.json"))
    if macro:
        items = []
        for key, val in list(macro.items())[:6]:
            if isinstance(val, dict):
                name = val.get("name", key)
                value = val.get("value", "N/A")
                change = val.get("change_pct")
                chg = f" ({change:+.2%})" if change else ""
                items.append(f"  {name}: {value}{chg}")
            else:
                items.append(f"  {key}: {val}")
        parts.append("Macro: " + "\n".join(items))
    alerts = load_json(get_data_path("global", "alerts.json"))
    if alerts and isinstance(alerts, list):
        parts.append(f"Alerts: {len(alerts)} active")
        for a in alerts[:3]:
            signal = a.get("signal", a.get("alert_type", ""))
            parts.append(f"  >> {signal}")
    parts.append("\nAsk: macro, sectors, correlations, alerts, peers")
    return "\n".join(parts)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
