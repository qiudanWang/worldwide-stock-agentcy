"""Global Agent: cross-market macro, sectors, correlations, global alerts."""

import glob
import json
import os
import pandas as pd
from datetime import datetime

from src.agents.base import BaseAgent, AgentResult
from src.common.config import get_data_path, load_yaml
from src.common.logger import get_logger

log = get_logger("agent.global")


class GlobalAgent(BaseAgent):
    """Runs after all market agents. Produces cross-market analysis."""

    def __init__(self, name="global", depends_on=None):
        super().__init__(name, agent_type="global", market=None,
                         depends_on=depends_on or [])

    def run(self) -> AgentResult:
        errors = []
        records = 0
        today = datetime.now().strftime("%Y%m%d")

        # Step 1: Merge universes
        self.update_status(progress="Merging universes...")
        master = self._merge_universes()
        if not master.empty:
            path = get_data_path("global", "universe_master.parquet")
            master.to_parquet(path, index=False)
            records += len(master)

        # Also save to legacy location for backward compatibility
        legacy_path = get_data_path("processed", "tech_universe_master.parquet")
        if not master.empty:
            master.to_parquet(legacy_path, index=False)

        # Step 2: Macro indicators
        self.update_status(progress="Fetching macro indicators...")
        macro_df, macro_latest = self._fetch_macro()
        if not macro_df.empty:
            path = get_data_path("global", "macro_indicators.parquet")
            # Merge with existing parquet: if a category failed this run (e.g. yfinance
            # commodities fetch error), preserve rows from the last successful run for
            # that category instead of wiping them.
            if os.path.exists(path):
                try:
                    existing = pd.read_parquet(path)
                    existing["date"] = pd.to_datetime(existing["date"])
                    macro_df["date"] = pd.to_datetime(macro_df["date"])
                    new_cats = set(macro_df["category"].unique()) if "category" in macro_df.columns else set()
                    if "category" in existing.columns and new_cats:
                        old_only = existing[~existing["category"].isin(new_cats)]
                        macro_df = pd.concat([old_only, macro_df], ignore_index=True)
                except Exception:
                    pass
            macro_df.to_parquet(path, index=False)
            records += len(macro_df)

        if macro_latest:
            path = get_data_path("global", "macro_latest.json")
            with open(path, "w") as f:
                json.dump(macro_latest, f, indent=2, default=str)

        # Step 3: Sector performance
        self.update_status(progress="Computing sector performance...")
        sector_perf = self._compute_sector_performance()
        if not sector_perf.empty:
            path = get_data_path("global", "sector_performance.parquet")
            sector_perf.to_parquet(path, index=False)
            records += len(sector_perf)

        # Step 4: Cross-market correlations
        self.update_status(progress="Computing correlations...")
        correlations = self._compute_correlations()
        if correlations:
            path = get_data_path("global", "correlations.json")
            with open(path, "w") as f:
                json.dump(correlations, f, indent=2, default=str)

        # Step 5: Geopolitical news
        self.update_status(progress="Fetching geopolitical news...")
        geo = self._fetch_geopolitical()
        if geo:
            path = get_data_path("global", "geopolitical_context.json")
            with open(path, "w") as f:
                json.dump(geo, f, indent=2, default=str)
            # Also save to legacy location
            legacy_geo = get_data_path("processed", "geopolitical_context.json")
            with open(legacy_geo, "w") as f:
                json.dump(geo, f, indent=2, default=str)

        # Step 5b: Build global peer groups
        self.update_status(progress="Building peer groups...")
        try:
            from src.peers.peer_matcher import build_global_peer_groups
            groups = build_global_peer_groups()
            log.info(f"Built {len(groups)} peer groups")
        except Exception as e:
            log.warning(f"Failed to build peer groups: {e}")

        # Step 6: Peer-relative alerts (CN vs US only, existing logic)
        self.update_status(progress="Checking peer alerts...")
        peer_alerts = self._check_peer_alerts()

        # Step 7: Global alerts (macro, sector, cross-market, breakout)
        self.update_status(progress="Generating global alerts...")
        all_alerts = self._generate_global_alerts(
            macro_latest, sector_perf, correlations
        )
        all_alerts.extend(peer_alerts)

        # Step 8: Cross-market price and gap alerts
        self.update_status(progress="Checking price & gap alerts across markets...")
        price_gap_alerts = self._check_cross_market_price_gap_alerts()
        all_alerts.extend(price_gap_alerts)

        # Collect all market-level alerts too
        market_alerts = self._collect_market_alerts()
        all_alerts.extend(market_alerts)

        # Save all alerts
        if all_alerts:
            path = get_data_path("global", "alerts.json")
            with open(path, "w") as f:
                json.dump(all_alerts, f, indent=2, default=str)

            # Also save to legacy location
            from src.alerts.formatter import save_alerts
            from src.common.stock_info import enrich_alerts
            enriched = enrich_alerts(all_alerts)
            save_alerts(enriched)

        self.update_status(
            progress=f"{records} records, {len(all_alerts)} alerts"
        )

        return AgentResult(
            success=True,
            records_written=records,
            errors=errors,
        )

    def _merge_universes(self):
        """Merge all market universes into a master universe."""
        markets_cfg = load_yaml("markets.yaml")["markets"]
        all_dfs = []

        for market_code in markets_cfg:
            try:
                path = get_data_path("markets", market_code, "universe.parquet")
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    if "market" not in df.columns:
                        df["market"] = market_code
                    all_dfs.append(df)
            except Exception as e:
                log.warning(f"Failed to load {market_code} universe: {e}")

        if not all_dfs:
            return pd.DataFrame()

        master = pd.concat(all_dfs, ignore_index=True)
        log.info(f"Master universe: {len(master)} stocks from {len(all_dfs)} markets")
        return master

    def _fetch_macro(self):
        """Fetch all macro indicators."""
        all_data = []

        # yfinance-based macro (commodities, currencies, sentiment)
        try:
            from src.macro.commodities import fetch_all_yf_macro, get_macro_latest
            yf_macro = fetch_all_yf_macro(days=60)
            if not yf_macro.empty:
                all_data.append(yf_macro)
                macro_latest = get_macro_latest(yf_macro)
            else:
                macro_latest = {}
        except Exception as e:
            log.warning(f"Failed to fetch yfinance macro: {e}")
            macro_latest = {}

        # FRED data (US rates, economy)
        try:
            from src.macro.fred import fetch_all_fred_series
            fred_df = fetch_all_fred_series()
            if not fred_df.empty:
                # Convert to same schema
                fred_named = fred_df.rename(columns={"name": "indicator"}) if "name" in fred_df.columns else fred_df.copy()
                if "category" not in fred_named.columns:
                    fred_named["category"] = "fred"
                all_data.append(fred_named[["date", "indicator", "value"]].assign(
                    category="fred", symbol=fred_named.get("series_id", ""),
                    change_pct=None
                ))
                # Add FRED latest values to macro_latest
                for _, row in fred_named.sort_values("date").groupby("indicator").tail(1).iterrows():
                    ind = row.get("indicator", row.get("name", ""))
                    if not ind:
                        continue
                    macro_latest[ind] = {
                        "value": round(float(row["value"]), 4) if pd.notna(row["value"]) else None,
                        "change_pct": None,
                        "date": row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
                        "symbol": str(row.get("series_id", "")),
                    }
                log.info(f"Added {fred_named['indicator'].nunique()} FRED series to macro_latest")
        except Exception as e:
            log.warning(f"Failed to fetch FRED data: {e}")

        # World Bank data (global GDP growth, CPI by country)
        try:
            from src.macro.world_bank import fetch_all_wb_indicators
            wb_df = fetch_all_wb_indicators()
            if not wb_df.empty:
                wb_for_concat = wb_df.copy()
                wb_for_concat["date"] = pd.to_datetime(wb_for_concat["year"].astype(str) + "-01-01")
                wb_for_concat["indicator"] = wb_for_concat["indicator_name"] + ": " + wb_for_concat["country_name"]
                all_data.append(wb_for_concat[["date", "indicator", "value"]].assign(
                    category="world_bank", symbol="", change_pct=None
                ))
                # Add latest per indicator + country to macro_latest
                for _, row in wb_df.sort_values("year").groupby(["indicator", "country"]).tail(1).iterrows():
                    key = f"{row['indicator_name']}: {row['country_name']}"
                    macro_latest[key] = {
                        "value": round(float(row["value"]), 2) if pd.notna(row["value"]) else None,
                        "change_pct": None,
                        "date": str(int(row["year"])),
                        "symbol": "",
                    }
                log.info(f"Added {len(wb_df['indicator'].unique())} World Bank indicators to macro_latest")
        except Exception as e:
            log.warning(f"Failed to fetch World Bank data: {e}")

        # CN macro data
        try:
            from src.macro.cn_macro import fetch_all_cn_macro
            cn_macro = fetch_all_cn_macro()
            if not cn_macro.empty:
                all_data.append(
                    cn_macro[["date", "value"]].assign(
                        indicator=cn_macro["name"],
                        category="cn_macro", symbol="", change_pct=None
                    )
                )
        except Exception as e:
            log.warning(f"Failed to fetch CN macro: {e}")

        # CN house price per-city data (saved separately, not in macro_indicators)
        try:
            from src.macro.cn_macro import fetch_cn_house_price_cities
            cities_df = fetch_cn_house_price_cities()
            if not cities_df.empty:
                city_path = get_data_path("markets", "CN", "house_price_cities.parquet")
                cities_df.to_parquet(city_path, index=False)
                log.info(f"CN house price cities: {cities_df['city_zh'].nunique()} cities, {len(cities_df)} rows saved")
        except Exception as e:
            log.warning(f"Failed to fetch CN house price cities: {e}")

        if not all_data:
            return pd.DataFrame(), {}

        combined = pd.concat(all_data, ignore_index=True)
        log.info(f"Macro data: {len(combined)} data points")
        return combined, macro_latest

    def _compute_sector_performance(self):
        """Compute sector performance across all markets."""
        from src.analysis.sector_performance import compute_sector_performance
        markets_cfg = load_yaml("markets.yaml")["markets"]
        today = datetime.now().strftime("%Y%m%d")

        all_latest = []
        for market_code in markets_cfg:
            try:
                path = get_data_path("markets", market_code,
                                     f"market_daily_{today}.parquet")
                if not os.path.exists(path):
                    # Fall back to most recent available snapshot
                    candidates = sorted(glob.glob(
                        get_data_path("markets", market_code, "market_daily_*.parquet")
                    ))
                    if not candidates:
                        continue
                    path = candidates[-1]
                df = pd.read_parquet(path)
                if df.empty:
                    continue

                # Always recompute returns from OHLCV history.
                # Checking column existence is not enough — files from older agent
                # runs may have the column but with NaN values for some tickers.
                df = df.sort_values(["ticker", "date"])
                grp = df.groupby("ticker")["close"]
                df["return_1d"]  = grp.transform(lambda s: s.pct_change(1, fill_method=None))
                df["return_5d"]  = grp.transform(lambda s: s.pct_change(5, fill_method=None))
                df["return_20d"] = grp.transform(lambda s: s.pct_change(20, fill_method=None))

                # Get latest day per ticker
                latest = df.groupby("ticker").tail(1)

                # Enrich with sector: prefer watchlist group names (granular)
                # over yfinance sector field (generic "Technology").
                latest = latest.copy()
                sector_map = {}

                # 1. Base: yfinance sector from universe.parquet
                uni_path = get_data_path("markets", market_code, "universe.parquet")
                if os.path.exists(uni_path):
                    uni = pd.read_parquet(uni_path)
                    if "sector" in uni.columns:
                        sector_map = dict(zip(uni["ticker"], uni["sector"]))

                # 2. Override with watchlist group names where available
                wl_file = f"{market_code.lower()}_watchlist.yaml"
                try:
                    wl_cfg = load_yaml(wl_file)
                    if isinstance(wl_cfg, dict):
                        for group_name, tickers in wl_cfg.items():
                            if isinstance(tickers, list):
                                for t in tickers:
                                    sector_map[str(t)] = str(group_name)
                except Exception:
                    pass

                if sector_map:
                    latest["sector"] = latest["ticker"].map(sector_map)

                # 3. Supplement with watchlist tickers missing from market_daily
                present_tickers = set(latest["ticker"].tolist())
                wl_extra = {}
                wl_file2 = f"{market_code.lower()}_watchlist.yaml"
                try:
                    wl_cfg2 = load_yaml(wl_file2)
                    if isinstance(wl_cfg2, dict):
                        for group_name, tickers in wl_cfg2.items():
                            if isinstance(tickers, list):
                                for t in tickers:
                                    if str(t) not in present_tickers:
                                        wl_extra[str(t)] = str(group_name)
                except Exception:
                    pass

                if wl_extra:
                    try:
                        import yfinance as yf
                        mkt_cfg = markets_cfg.get(market_code, {})
                        suffix = mkt_cfg.get("ticker_suffix", "")
                        yf_tickers = [t + suffix for t in wl_extra]
                        raw = yf.download(
                            yf_tickers, period="30d", auto_adjust=True,
                            progress=False, threads=True
                        )
                        if not raw.empty:
                            close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0) if isinstance(raw.columns, pd.MultiIndex) else None
                            if close is not None:
                                extra_rows = []
                                for yf_sym, base_t in zip(yf_tickers, list(wl_extra.keys())):
                                    col = yf_sym if yf_sym in close.columns else (close.columns[0] if len(yf_tickers) == 1 else None)
                                    if col is None:
                                        continue
                                    s = close[col].dropna()
                                    if len(s) < 2:
                                        continue
                                    r1d = s.pct_change().iloc[-1]
                                    r5d = s.pct_change(5).iloc[-1] if len(s) >= 5 else None
                                    r20d = s.pct_change(20).iloc[-1] if len(s) >= 20 else None
                                    extra_rows.append({
                                        "ticker": base_t,
                                        "market": market_code,
                                        "close": float(s.iloc[-1]),
                                        "return_1d": float(r1d) if pd.notna(r1d) else None,
                                        "return_5d": float(r5d) if r5d is not None and pd.notna(r5d) else None,
                                        "return_20d": float(r20d) if r20d is not None and pd.notna(r20d) else None,
                                        "sector": wl_extra[base_t],
                                    })
                                if extra_rows:
                                    extra_df = pd.DataFrame(extra_rows)
                                    latest = pd.concat([latest, extra_df], ignore_index=True)
                                    log.info(f"{market_code}: supplemented {len(extra_rows)} watchlist tickers")
                    except Exception as e:
                        log.warning(f"{market_code}: failed to supplement watchlist tickers: {e}")

                all_latest.append(latest)
            except Exception as e:
                log.warning(f"Failed to load {market_code} data: {e}")

        if not all_latest:
            return pd.DataFrame()

        combined = pd.concat(all_latest, ignore_index=True)
        return compute_sector_performance(combined)

    def _compute_correlations(self):
        """Compute cross-market index correlations."""
        from src.analysis.correlations import compute_index_correlations
        markets_cfg = load_yaml("markets.yaml")["markets"]

        all_indices = []
        for market_code in markets_cfg:
            try:
                idx_path = get_data_path("markets", market_code, "indices.parquet")
                if os.path.exists(idx_path):
                    df = pd.read_parquet(idx_path)
                    all_indices.append(df)
            except Exception:
                continue

        if not all_indices:
            return {}

        combined = pd.concat(all_indices, ignore_index=True)
        return compute_index_correlations(combined)

    def _fetch_geopolitical(self):
        """Fetch geopolitical news context."""
        try:
            from src.news.world_monitor import fetch_geopolitical_context
            return fetch_geopolitical_context()
        except Exception as e:
            log.warning(f"Failed to fetch geopolitical context: {e}")
            return {}

    def _check_peer_alerts(self):
        """Check peer-relative alerts (CN vs US)."""
        today = datetime.now().strftime("%Y%m%d")
        alerts = []

        try:
            cn_path = get_data_path("markets", "CN",
                                     f"market_daily_{today}.parquet")
            us_path = get_data_path("markets", "US",
                                     f"market_daily_{today}.parquet")

            if not os.path.exists(cn_path) or not os.path.exists(us_path):
                return []

            cn_data = pd.read_parquet(cn_path)
            us_data = pd.read_parquet(us_path)
            combined = pd.concat([cn_data, us_data], ignore_index=True)

            from src.market_data.volume_signals import get_latest_signals
            latest = get_latest_signals(combined)

            from src.alerts.peer_relative_alerts import check_peer_relative_alerts
            peer_alerts = check_peer_relative_alerts(latest)

            if not peer_alerts.empty:
                for _, row in peer_alerts.iterrows():
                    alerts.append({
                        "date": today,
                        "ticker": row["cn_ticker"],
                        "market": "CN",
                        "alert_type": "peer_relative",
                        "signal": row["signal"],
                    })
        except Exception as e:
            log.warning(f"Failed to check peer alerts: {e}")

        return alerts

    def _generate_global_alerts(self, macro_latest, sector_perf, correlations):
        """Generate all global-level alerts."""
        today = datetime.now().strftime("%Y%m%d")
        alerts = []

        # Macro alerts (includes vix_spike, vix_elevated, dxy, oil, gold, btc, rates)
        try:
            from src.alerts.macro_alerts import check_macro_alerts
            macro_alerts = check_macro_alerts(macro_latest or {})
            for a in macro_alerts:
                a["date"] = today
            alerts.extend(macro_alerts)
        except Exception as e:
            log.warning(f"Macro alerts failed: {e}")

        # Sector rotation alerts
        if not sector_perf.empty:
            try:
                from src.analysis.sector_performance import detect_sector_rotation
                from src.alerts.sector_alerts import check_sector_rotation_alerts
                rotations = detect_sector_rotation(sector_perf, sector_perf)
                sector_alerts = check_sector_rotation_alerts(rotations)
                for a in sector_alerts:
                    a["date"] = today
                alerts.extend(sector_alerts)
            except Exception as e:
                log.warning(f"Sector alerts failed: {e}")

        # Cross-market divergence
        if not sector_perf.empty:
            try:
                from src.analysis.cross_market import detect_cross_market_divergence
                from src.alerts.sector_alerts import check_cross_market_alerts
                divergences = detect_cross_market_divergence(sector_perf)
                cross_alerts = check_cross_market_alerts(divergences)
                for a in cross_alerts:
                    a["date"] = today
                alerts.extend(cross_alerts)
            except Exception as e:
                log.warning(f"Cross-market alerts failed: {e}")

        # Index breakout alerts
        try:
            from src.analysis.cross_market import detect_index_breakout
            from src.alerts.sector_alerts import check_index_breakout_alerts
            markets_cfg = load_yaml("markets.yaml")["markets"]
            all_indices = []
            for mc in markets_cfg:
                idx_path = get_data_path("markets", mc, "indices.parquet")
                if os.path.exists(idx_path):
                    all_indices.append(pd.read_parquet(idx_path))
            if all_indices:
                idx_combined = pd.concat(all_indices, ignore_index=True)
                breakouts = detect_index_breakout(idx_combined)
                breakout_alerts = check_index_breakout_alerts(breakouts)
                for a in breakout_alerts:
                    a["date"] = today
                alerts.extend(breakout_alerts)
        except Exception as e:
            log.warning(f"Index breakout alerts failed: {e}")

        log.info(f"Global alerts: {len(alerts)} total")
        return alerts

    def _check_cross_market_price_gap_alerts(self):
        """Run price_alert and gap_alert checks across all markets.

        This catches any big movers that individual SignalAgents may have
        already found, plus ensures we have a global view.
        """
        markets_cfg = load_yaml("markets.yaml")["markets"]
        today = datetime.now().strftime("%Y%m%d")
        rules = load_yaml("alert_rules.yaml")
        daily_threshold = rules.get("price_alert", {}).get("daily_return_threshold", 0.05)
        gap_threshold = rules.get("gap_alert", {}).get("gap_threshold", 0.03)

        alerts = []

        for market_code in markets_cfg:
            try:
                path = get_data_path("markets", market_code,
                                     f"market_daily_{today}.parquet")
                if not os.path.exists(path):
                    continue
                df = pd.read_parquet(path)
                if df.empty:
                    continue

                # Price alerts: daily return >= threshold
                latest = df.sort_values("date").groupby("ticker").tail(1)
                if "return_1d" in latest.columns:
                    big_movers = latest[latest["return_1d"].abs() >= daily_threshold]
                    for _, row in big_movers.iterrows():
                        ret = row["return_1d"]
                        direction = "up" if ret > 0 else "down"
                        alerts.append({
                            "date": today,
                            "alert_type": "price_alert",
                            "ticker": row["ticker"],
                            "market": market_code,
                            "return_1d": round(ret, 4),
                            "close": round(row.get("close", 0), 2),
                            "signal": (
                                f"{row['ticker']} {direction} {abs(ret):.1%} "
                                f"to {row.get('close', 0):.2f}"
                            ),
                        })

                # Gap alerts: open vs previous close
                for ticker, group in df.sort_values("date").groupby("ticker"):
                    if len(group) < 2:
                        continue
                    if "open" not in group.columns or "close" not in group.columns:
                        continue

                    prev_close = group.iloc[-2]["close"]
                    today_open = group.iloc[-1]["open"]

                    if prev_close == 0 or pd.isna(prev_close) or pd.isna(today_open):
                        continue

                    gap_pct = (today_open - prev_close) / prev_close

                    if abs(gap_pct) >= gap_threshold:
                        direction = "up" if gap_pct > 0 else "down"
                        alerts.append({
                            "date": today,
                            "alert_type": "gap_alert",
                            "ticker": ticker,
                            "market": market_code,
                            "gap_pct": round(gap_pct, 4),
                            "prev_close": round(prev_close, 2),
                            "open": round(today_open, 2),
                            "signal": (
                                f"{ticker} gap {direction} {abs(gap_pct):.1%}: "
                                f"prev close {prev_close:.2f} -> open {today_open:.2f}"
                            ),
                        })

            except Exception as e:
                log.warning(f"Price/gap alerts for {market_code} failed: {e}")

        if alerts:
            log.info(
                f"Cross-market price/gap alerts: {len(alerts)} "
                f"across {len(markets_cfg)} markets"
            )
        return alerts

    def _collect_market_alerts(self):
        """Collect all per-market alerts into one list."""
        markets_cfg = load_yaml("markets.yaml")["markets"]
        all_alerts = []

        for market_code in markets_cfg:
            try:
                path = get_data_path("markets", market_code, "alerts.json")
                if os.path.exists(path):
                    with open(path, "r") as f:
                        market_alerts = json.load(f)
                    all_alerts.extend(market_alerts)
            except Exception:
                continue

        return all_alerts
