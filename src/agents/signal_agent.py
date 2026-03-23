"""Signal Agent: per-market capital flow + local alerts."""

import json
import os
import pandas as pd
from datetime import datetime

from src.agents.base import BaseAgent, AgentResult
from src.common.config import get_data_path, load_yaml
from src.common.logger import get_logger

log = get_logger("agent.signal")


class SignalAgent(BaseAgent):
    """Monitors capital flows and generates local alerts for a single market."""

    def __init__(self, name, market, market_config, depends_on=None):
        super().__init__(name, agent_type="signal", market=market,
                         depends_on=depends_on)
        self.market_config = market_config

    def run(self) -> AgentResult:
        market = self.market
        today = datetime.now().strftime("%Y%m%d")
        all_alerts = []
        errors = []
        records = 0

        # Step 1: Capital flow
        self.update_status(progress="Fetching capital flow...")
        capital_flow = self._fetch_capital_flow()
        if not capital_flow.empty:
            flow_path = get_data_path("markets", market, "capital_flow.parquet")
            # Append to existing history (dedup by date) so single-day sources
            # like CN northbound accumulate over time rather than being overwritten
            if os.path.exists(flow_path):
                try:
                    existing = pd.read_parquet(flow_path)
                    combined = pd.concat([existing, capital_flow], ignore_index=True)
                    combined["date"] = pd.to_datetime(combined["date"])
                    combined = combined.drop_duplicates(subset=["date"], keep="last")
                    combined = combined.sort_values("date").reset_index(drop=True)
                    capital_flow = combined
                except Exception:
                    pass
            capital_flow.to_parquet(flow_path, index=False)
            records += len(capital_flow)

        # Step 2: Volume spike alerts
        self.update_status(progress="Checking volume alerts...")
        vol_alerts = self._check_volume_alerts()
        all_alerts.extend(vol_alerts)

        # Step 3: News spike alerts
        self.update_status(progress="Checking news alerts...")
        news_alerts = self._check_news_alerts()
        all_alerts.extend(news_alerts)

        # Step 4: Capital flow alerts
        self.update_status(progress="Checking capital flow alerts...")
        cap_alerts = self._check_capital_flow_alerts(capital_flow)
        all_alerts.extend(cap_alerts)

        # Step 5: Price alerts (new)
        self.update_status(progress="Checking price alerts...")
        price_alerts = self._check_price_alerts()
        all_alerts.extend(price_alerts)

        # Step 6: Gap alerts (new)
        self.update_status(progress="Checking gap alerts...")
        gap_alerts = self._check_gap_alerts()
        all_alerts.extend(gap_alerts)

        # Save alerts
        if all_alerts:
            for a in all_alerts:
                a["date"] = today
                a["market"] = market

            alerts_path = get_data_path("markets", market, "alerts.json")
            with open(alerts_path, "w") as f:
                json.dump(all_alerts, f, indent=2, default=str)

        self.update_status(
            progress=f"{len(all_alerts)} alerts"
        )

        return AgentResult(
            success=True,
            records_written=records + len(all_alerts),
            errors=errors,
        )

    def _fetch_capital_flow(self):
        """Fetch capital flow data for this market."""
        flow_source = self.market_config.get("capital_flow_source")

        if flow_source is None:
            return pd.DataFrame()

        if flow_source == "akshare_northbound" and self.market == "CN":
            from src.capital_flow.northbound import fetch_northbound_flow
            df = fetch_northbound_flow()
            if not df.empty:
                df["flow_type"] = "northbound"
                df["market"] = "CN"
            return df

        elif flow_source == "akshare_southbound" and self.market == "HK":
            # Southbound uses same API with different filter
            try:
                from src.capital_flow.northbound import fetch_northbound_flow
                # Reuse northbound; in practice southbound needs separate call
                return pd.DataFrame()
            except Exception:
                return pd.DataFrame()

        elif flow_source == "etf_proxy":
            etf = self.market_config.get("capital_flow_etf")
            if etf:
                from src.capital_flow.etf_flows import fetch_etf_flow_proxy
                return fetch_etf_flow_proxy(etf, self.market)
            return pd.DataFrame()

        return pd.DataFrame()

    def _check_volume_alerts(self):
        """Check for volume spikes in today's market data."""
        today = datetime.now().strftime("%Y%m%d")
        market = self.market

        try:
            path = get_data_path("markets", market,
                                 f"market_daily_{today}.parquet")
            df = pd.read_parquet(path)
        except (FileNotFoundError, Exception):
            return []

        if df.empty:
            return []

        from src.market_data.volume_signals import compute_volume_signals, get_latest_signals
        if "volume_ratio" not in df.columns:
            df = compute_volume_signals(df)
        latest = get_latest_signals(df)

        rules = load_yaml("alert_rules.yaml")
        threshold = rules["volume_alert"]["volume_ratio_threshold"]
        surge_threshold = rules["volume_alert"].get("volume_surge_threshold", 5.0)

        alerts = []
        spikes = latest[latest["volume_ratio"] >= threshold]
        for _, row in spikes.iterrows():
            vr = row["volume_ratio"]
            tier = "volume_surge" if vr >= surge_threshold else "volume_spike"
            alerts.append({
                "alert_type": tier,
                "ticker": row["ticker"],
                "volume_ratio": round(vr, 2),
                "return_1d": round(row.get("return_1d", 0) or 0, 4),
                "close": round(row.get("close", 0), 2),
            })

        return alerts

    def _check_news_alerts(self):
        """Check for news spikes in this market's news data."""
        market = self.market

        try:
            news_path = get_data_path("markets", market, "news.parquet")
            news_df = pd.read_parquet(news_path)
        except (FileNotFoundError, Exception):
            return []

        if news_df.empty:
            return []

        from src.news.keyword_filter import compute_news_counts
        from src.alerts.news_alerts import check_news_alerts

        news_counts = compute_news_counts(news_df)
        alerts_df = check_news_alerts(news_counts)

        alerts = []
        if not alerts_df.empty:
            for _, row in alerts_df.iterrows():
                alerts.append({
                    "alert_type": "news_spike",
                    "ticker": row["ticker"],
                    "news_count_keyword": int(row.get("news_count_keyword", 0)),
                })

        return alerts

    def _check_capital_flow_alerts(self, capital_flow):
        """Check for significant capital flow events."""
        if capital_flow.empty:
            return []

        market = self.market
        alerts = []

        if market == "CN" and "net_flow" in capital_flow.columns:
            from src.alerts.capital_flow_alerts import check_capital_flow_alerts
            cap_alerts = check_capital_flow_alerts(capital_flow)
            if not cap_alerts.empty:
                flow = cap_alerts["net_flow"].iloc[0]
                alerts.append({
                    "alert_type": "capital_flow",
                    "ticker": "NORTHBOUND",
                    "net_flow": float(flow),
                })

        elif "volume_ratio" in capital_flow.columns:
            # ETF proxy: flag if ETF volume spike
            rules = load_yaml("alert_rules.yaml")
            threshold = rules.get("etf_flow_alert", {}).get(
                "volume_spike_threshold", 1.5
            )
            latest = capital_flow.sort_values("date").tail(1)
            if not latest.empty:
                vr = latest["volume_ratio"].iloc[0]
                if pd.notna(vr) and vr >= threshold:
                    etf = self.market_config.get("capital_flow_etf", "")
                    alerts.append({
                        "alert_type": "capital_flow",
                        "ticker": etf,
                        "volume_ratio": round(vr, 2),
                        "signal": f"ETF {etf} volume {vr:.1f}x average",
                    })

        return alerts

    def _check_price_alerts(self):
        """Check for stocks with large daily price moves (>= 5% up or down)."""
        today = datetime.now().strftime("%Y%m%d")
        market = self.market

        try:
            path = get_data_path("markets", market,
                                 f"market_daily_{today}.parquet")
            df = pd.read_parquet(path)
        except (FileNotFoundError, Exception):
            return []

        if df.empty:
            return []

        rules = load_yaml("alert_rules.yaml")
        daily_threshold = rules.get("price_alert", {}).get("daily_return_threshold", 0.05)

        # Get latest day per ticker
        latest = df.sort_values("date").groupby("ticker").tail(1)

        if "return_1d" not in latest.columns:
            return []

        alerts = []
        big_movers = latest[latest["return_1d"].abs() >= daily_threshold]
        for _, row in big_movers.iterrows():
            ret = row["return_1d"]
            direction = "up" if ret > 0 else "down"
            alerts.append({
                "alert_type": "price_alert",
                "ticker": row["ticker"],
                "return_1d": round(ret, 4),
                "close": round(row.get("close", 0), 2),
                "signal": f"{row['ticker']} {direction} {abs(ret):.1%} to {row.get('close', 0):.2f}",
            })

        if alerts:
            log.info(f"Price alerts ({market}): {len(alerts)} stocks with >{daily_threshold:.0%} daily move")

        return alerts

    def _check_gap_alerts(self):
        """Check for stocks with significant gaps (open vs previous close >= 3%)."""
        today = datetime.now().strftime("%Y%m%d")
        market = self.market

        try:
            path = get_data_path("markets", market,
                                 f"market_daily_{today}.parquet")
            df = pd.read_parquet(path)
        except (FileNotFoundError, Exception):
            return []

        if df.empty:
            return []

        rules = load_yaml("alert_rules.yaml")
        gap_threshold = rules.get("gap_alert", {}).get("gap_threshold", 0.03)

        alerts = []

        # For each ticker, compute the gap between today's open and yesterday's close
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
                    "alert_type": "gap_alert",
                    "ticker": ticker,
                    "gap_pct": round(gap_pct, 4),
                    "prev_close": round(prev_close, 2),
                    "open": round(today_open, 2),
                    "close": round(group.iloc[-1].get("close", 0), 2),
                    "signal": (
                        f"{ticker} gap {direction} {abs(gap_pct):.1%}: "
                        f"prev close {prev_close:.2f} -> open {today_open:.2f}"
                    ),
                })

        if alerts:
            log.info(f"Gap alerts ({market}): {len(alerts)} stocks with >{gap_threshold:.0%} gap")

        return alerts
