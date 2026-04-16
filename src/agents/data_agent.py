"""Data Agent: per-market universe + OHLCV + market cap + indices."""

import os
import pandas as pd
from datetime import datetime, timedelta

from src.agents.base import BaseAgent, AgentResult
from src.common.config import get_data_path, load_yaml, get_settings
from src.common.logger import get_logger
from src.common.rate_limiter import yf_limiter

log = get_logger("agent.data")


def _enrich_subsector_cn(df: pd.DataFrame) -> pd.DataFrame:
    """Add subsector (申万 Level-2 industry) for CN A-share universe."""
    try:
        import akshare as ak
        sw = ak.stock_board_industry_name_em()  # cols: 板块名称, 板块代码
        # Get constituent stocks for each SW industry board
        industry_map = {}
        for _, row in sw.iterrows():
            board = row.get("板块名称", "")
            try:
                stocks = ak.stock_board_industry_cons_em(symbol=board)
                if stocks is not None and not stocks.empty:
                    code_col = next((c for c in ["代码", "股票代码", "code"] if c in stocks.columns), None)
                    if code_col:
                        for code in stocks[code_col].astype(str):
                            if code not in industry_map:
                                industry_map[code] = board
            except Exception:
                continue
        if industry_map:
            df = df.copy()
            df["subsector"] = df["ticker"].map(industry_map).fillna("")
            filled = (df["subsector"] != "").sum()
            log.info(f"[CN] Subsector enriched: {filled}/{len(df)} stocks")
    except Exception as e:
        log.warning(f"[CN] Subsector enrichment failed: {e}")
        if "subsector" not in df.columns:
            df = df.copy()
            df["subsector"] = ""
    return df


class DataAgent(BaseAgent):
    """Fetches universe, OHLCV, market cap, and indices for a single market."""

    def __init__(self, name, market, market_config, depends_on=None):
        super().__init__(name, agent_type="data", market=market,
                         depends_on=depends_on)
        self.market_config = market_config

    def should_run(self):
        """Skip if today's market data snapshot already exists."""
        today = datetime.now().strftime("%Y%m%d")
        path = get_data_path("markets", self.market,
                             f"market_daily_{today}.parquet")
        if os.path.exists(path):
            log.info(f"[{self.name}] Today's snapshot exists, skipping")
            return False
        return True

    def run(self) -> AgentResult:
        today = datetime.now().strftime("%Y%m%d")
        total_records = 0
        errors = []

        # Step 1: Build/load universe
        self.update_status(progress="Building universe...")
        universe = self._build_universe()
        if universe.empty:
            return AgentResult(success=False, errors=["Failed to build universe"])
        total_records += len(universe)

        tickers = universe["ticker"].tolist()
        yf_symbols = dict(zip(universe["ticker"], universe.get("yf_symbol", universe["ticker"])))

        # Step 2: Fetch market data
        self.update_status(progress=f"Fetching {len(tickers)} stocks...")
        market_data = self._fetch_market_data(tickers, yf_symbols)

        if market_data.empty:
            errors.append("No market data fetched")
        else:
            # Compute volume signals
            from src.market_data.volume_signals import compute_volume_signals
            market_data = compute_volume_signals(market_data)

            # Save daily snapshot
            path = get_data_path("markets", self.market,
                                 f"market_daily_{today}.parquet")
            market_data.to_parquet(path, index=False)
            total_records += market_data["ticker"].nunique()
            self.update_status(
                progress=f"{market_data['ticker'].nunique()} stocks fetched"
            )

        # Step 3: Market cap
        self.update_status(progress="Fetching market cap...")
        cap_df = self._fetch_market_cap(tickers, yf_symbols)
        if not cap_df.empty:
            cap_path = get_data_path("markets", self.market, "market_cap.parquet")
            # Only overwrite if we have reasonable coverage (>30% of universe).
            # A partial API failure should not wipe valid cached data.
            coverage = len(cap_df) / max(len(tickers), 1)
            if coverage >= 0.30 or not os.path.exists(cap_path):
                cap_df.to_parquet(cap_path, index=False)
            else:
                log.warning(
                    f"Market cap coverage too low ({len(cap_df)}/{len(tickers)} tickers); "
                    "keeping cached file."
                )

        # Step 4: Indices
        self.update_status(progress="Fetching indices...")
        indices = self._fetch_indices()
        if not indices.empty:
            idx_path = get_data_path("markets", self.market, "indices.parquet")
            # Merge with existing history so a partial yfinance response (1 day)
            # doesn't wipe prior rows needed to compute change_pct.
            if os.path.exists(idx_path):
                try:
                    existing = pd.read_parquet(idx_path)
                    existing["date"] = pd.to_datetime(existing["date"])
                    indices["date"] = pd.to_datetime(indices["date"])
                    combined = pd.concat([existing, indices], ignore_index=True)
                    combined = combined.sort_values("date").drop_duplicates(
                        subset=["date", "symbol"], keep="last"
                    )
                    # Recompute change_pct on full series per symbol
                    combined["change_pct"] = combined.groupby("symbol")["close"].transform(
                        lambda s: s.pct_change(1)
                    )
                    indices = combined
                except Exception:
                    pass
            indices.to_parquet(idx_path, index=False)

        # Step 5: Fundamentals cache (weekly refresh — non-CN markets only)
        # CN uses akshare in local_trading_agent; HK/JP/etc. use yfinance.
        if self.market != "CN":
            fund_path = get_data_path("markets", self.market, "fundamentals.parquet")
            fund_stale = True
            if os.path.exists(fund_path):
                age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(fund_path))
                if age.days < 7:
                    fund_stale = False
                    log.info(f"[{self.name}] Fundamentals cache is {age.days}d old, skipping")
            if fund_stale:
                self.update_status(progress="Fetching fundamentals (weekly)...")
                try:
                    from src.financials.yf_fundamentals import fetch_yf_fundamentals_batch
                    fund_df = fetch_yf_fundamentals_batch(tickers, self.market, limit=60)
                    if not fund_df.empty:
                        fund_df.to_parquet(fund_path, index=False)
                        log.info(f"[{self.name}] Saved fundamentals: {len(fund_df)} stocks → {fund_path}")
                    else:
                        log.warning(f"[{self.name}] Fundamentals fetch returned empty")
                except Exception as e:
                    log.warning(f"[{self.name}] Fundamentals fetch failed: {e}")

        return AgentResult(
            success=len(errors) == 0 or not market_data.empty,
            records_written=total_records,
            errors=errors,
        )

    def _build_universe(self):
        """Build or load the universe for this market."""
        source = self.market_config.get("universe_source", "watchlist")
        market = self.market

        if source == "akshare_industry":
            # CN market: use existing cn_universe
            from src.universe.cn_universe import build_cn_universe
            df = build_cn_universe()
            if not df.empty:
                df["yf_symbol"] = df["ticker"]
        elif source == "nasdaq_api":
            from src.universe.us_universe import build_us_tech_universe
            df = build_us_tech_universe()
        elif source == "krx_api":
            from src.universe.kr_universe import build_kr_tech_universe
            df = build_kr_tech_universe()
        elif source == "twse_api":
            from src.universe.tw_universe import build_tw_tech_universe
            df = build_tw_tech_universe()
        elif source == "asx_api":
            from src.universe.au_universe import build_au_tech_universe
            df = build_au_tech_universe()
        elif source == "hkex_api":
            from src.universe.hk_universe import build_hk_tech_universe
            df = build_hk_tech_universe()
        elif source == "jpx_api":
            from src.universe.jp_universe import build_jp_tech_universe
            df = build_jp_tech_universe()
        elif source == "nse_api":
            from src.universe.in_universe import build_in_tech_universe
            df = build_in_tech_universe()
        elif source == "lse_api":
            from src.universe.uk_universe import build_uk_tech_universe
            df = build_uk_tech_universe()
        elif source == "xetra_api":
            from src.universe.de_universe import build_de_tech_universe
            df = build_de_tech_universe()
        elif source == "euronext_api":
            from src.universe.fr_universe import build_fr_tech_universe
            df = build_fr_tech_universe()
        elif source == "b3_api":
            from src.universe.br_universe import build_br_tech_universe
            df = build_br_tech_universe()
        elif source == "tadawul_api":
            from src.universe.sa_universe import build_sa_tech_universe
            df = build_sa_tech_universe()
        elif source == "watchlist":
            from src.universe.yf_universe import build_yf_universe
            config_file = self.market_config.get("universe_config", "")
            suffix = self.market_config.get("ticker_suffix", "")
            df = build_yf_universe(market, config_file, suffix)
        else:
            log.error(f"[{self.name}] Unknown universe source: {source}")
            return pd.DataFrame()

        # Fallback to cached universe if live fetch failed
        if df.empty:
            cached = get_data_path("markets", market, "universe.parquet")
            if os.path.exists(cached):
                log.warning(f"[{self.name}] Live fetch failed, using cached universe")
                df = pd.read_parquet(cached)
        # Second fallback: legacy processed file (CN)
        if df.empty and market == "CN":
            legacy = get_data_path("processed", "cn_tech_universe.parquet")
            if os.path.exists(legacy):
                log.warning(f"[{self.name}] Using legacy CN universe")
                df = pd.read_parquet(legacy)
                if "yf_symbol" not in df.columns:
                    df["yf_symbol"] = df["ticker"]

        # Enrich subsector if missing
        if not df.empty:
            has_industry = "industry" in df.columns and (df["industry"].fillna("") != "").any()
            has_subsector = "subsector" in df.columns and (df["subsector"].fillna("") != "").any()

            # Step 1: Seed subsector from previously cached universe (preserves backfilled data)
            cached_path = get_data_path("markets", market, "universe.parquet")
            if not has_subsector and os.path.exists(cached_path):
                try:
                    cached = pd.read_parquet(cached_path)
                    if "subsector" in cached.columns and (cached["subsector"].fillna("") != "").any():
                        sub_map = cached.dropna(subset=["ticker"]).set_index("ticker")["subsector"].to_dict()
                        df = df.copy()
                        if "subsector" not in df.columns:
                            df["subsector"] = ""
                        df["subsector"] = df["ticker"].map(sub_map).fillna(df["subsector"].fillna(""))
                        has_subsector = (df["subsector"].fillna("") != "").any()
                        log.info(f"[{self.name}] Subsector: seeded {(df['subsector'].fillna('') != '').sum()}/{len(df)} from cache")
                except Exception as e:
                    log.warning(f"[{self.name}] Could not seed subsector from cache: {e}")

            # Step 2: Enrich any remaining empty subsectors
            if has_industry and not has_subsector:
                # Markets whose universe builder already populates industry (KR, TW, BR, DE, FR, IN, UK, SA)
                df["subsector"] = df["industry"]
            elif not has_subsector:
                if market == "CN":
                    df = _enrich_subsector_cn(df)
                    # Fallback to yfinance if akshare failed
                    if not (df["subsector"].fillna("") != "").any():
                        from src.universe.yf_universe import enrich_subsector_yf
                        df = df.copy()
                        _suffix_map = {"6": ".SS"}  # 6xxxxx → Shanghai, others → Shenzhen
                        df["_cn_yf"] = df["ticker"].apply(
                            lambda t: f"{t}.SS" if str(t).startswith("6") else f"{t}.SZ"
                        )
                        df = enrich_subsector_yf(df, ticker_col="_cn_yf")
                        df = df.drop(columns=["_cn_yf"], errors="ignore")
                else:
                    from src.universe.yf_universe import enrich_subsector_yf
                    # Apply exchange suffix to yf_symbol if missing (HK → .HK, JP → .T, etc.)
                    _YF_SUFFIX = {"HK": ".HK", "JP": ".T", "AU": ".AX", "IN": ".NS",
                                  "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
                                  "UK": ".L", "BR": ".SA", "SA": ".SR"}
                    suffix = _YF_SUFFIX.get(market, "")
                    if suffix and "yf_symbol" in df.columns:
                        df = df.copy()
                        mask = ~df["yf_symbol"].astype(str).str.endswith(suffix)
                        df.loc[mask, "yf_symbol"] = df.loc[mask, "yf_symbol"].astype(str) + suffix
                    yf_col = "yf_symbol" if "yf_symbol" in df.columns else "ticker"
                    df = enrich_subsector_yf(df, ticker_col=yf_col)

        # Enrich names if still set to ticker number (fallback from build_yf_universe placeholder)
        if not df.empty and "name" in df.columns and "ticker" in df.columns:
            name_is_ticker = df["name"].astype(str) == df["ticker"].astype(str)
            if name_is_ticker.any():
                from src.universe.yf_universe import enrich_name_yf
                _YF_SUFFIX = {"HK": ".HK", "JP": ".T", "AU": ".AX", "IN": ".NS",
                              "KR": ".KS", "TW": ".TW", "DE": ".DE", "FR": ".PA",
                              "UK": ".L", "BR": ".SA", "SA": ".SR"}
                suffix = _YF_SUFFIX.get(market, "")
                if suffix and "yf_symbol" in df.columns:
                    df = df.copy()
                    mask = ~df["yf_symbol"].astype(str).str.endswith(suffix)
                    df.loc[mask, "yf_symbol"] = df.loc[mask, "yf_symbol"].astype(str) + suffix
                yf_col = "yf_symbol" if "yf_symbol" in df.columns else "ticker"
                df = enrich_name_yf(df, ticker_col=yf_col)

        # Save to market directory
        if not df.empty:
            path = get_data_path("markets", market, "universe.parquet")
            df.to_parquet(path, index=False)
            log.info(f"[{self.name}] Universe: {len(df)} stocks")

        return df

    def _fetch_market_data(self, tickers, yf_symbols):
        """Fetch OHLCV data for all tickers in this market."""
        source = self.market_config.get("source", "yfinance")
        suffix = self.market_config.get("ticker_suffix", "")
        market = self.market

        if source == "akshare" and market == "CN":
            from src.market_data.cn_market_data import fetch_cn_batch
            return fetch_cn_batch(tickers)
        else:
            from src.market_data.yf_market_data import fetch_yf_batch
            return fetch_yf_batch(tickers, market=market, ticker_suffix=suffix)

    def _fetch_market_cap(self, tickers, yf_symbols):
        """Fetch market cap data for this market."""
        # Check cache freshness — also invalidate if coverage is < 50% of current universe
        cap_path = get_data_path("markets", self.market, "market_cap.parquet")
        if not self.force and os.path.exists(cap_path):
            age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cap_path))
            cached = pd.read_parquet(cap_path)
            coverage = len(cached) / max(len(tickers), 1)
            if age < timedelta(days=1) and coverage >= 0.5:
                log.info(f"[{self.name}] Market cap cache fresh, skipping")
                return cached

        source = self.market_config.get("market_cap_source", "yfinance")
        market = self.market

        if source == "akshare" and market == "CN":
            from src.market_data.market_cap import (
                fetch_cn_market_cap_spot, fetch_cn_market_cap_batch, fetch_cn_market_cap_yf
            )
            df = fetch_cn_market_cap_spot()
            if df.empty or df["market_cap"].notna().sum() < len(tickers) * 0.5:
                # akshare spot failed or incomplete — try akshare batch
                df_batch = fetch_cn_market_cap_batch(tickers)
                if df_batch["market_cap"].notna().sum() > df["market_cap"].notna().sum():
                    df = df_batch
            if df.empty or df["market_cap"].notna().sum() < len(tickers) * 0.5:
                # Both akshare methods failed — fall back to yfinance
                log.warning("[CN_data] akshare market cap failed, falling back to yfinance")
                df = fetch_cn_market_cap_yf(tickers)
            return df
        else:
            # Use yfinance fast_info in parallel batches with per-call timeout
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
            suffix = self.market_config.get("ticker_suffix", "")
            rows = []
            PER_CALL_TIMEOUT = 15  # seconds — skip any ticker that doesn't respond
            WORKERS = 5

            def _fetch_one(ticker):
                symbol = f"{ticker}{suffix}" if suffix else ticker
                try:
                    with yf_limiter:
                        fi = yf.Ticker(symbol).fast_info
                        cap = getattr(fi, "market_cap", None)
                    return ticker, cap
                except Exception:
                    return ticker, None

            total = len(tickers)
            done = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(_fetch_one, t): t for t in tickers}
                for fut in as_completed(futures):
                    try:
                        ticker, cap = fut.result(timeout=PER_CALL_TIMEOUT)
                    except (FutureTimeout, Exception):
                        ticker = futures[fut]
                        cap = None
                    rows.append({"ticker": ticker, "market_cap": cap})
                    done += 1
                    if done % 50 == 0 or done == total:
                        ok = sum(1 for r in rows if r["market_cap"] is not None)
                        log.info(f"[{self.name}] Market cap: {done}/{total} ({ok} ok)")

            df = pd.DataFrame(rows)
            return df

    def _fetch_indices(self):
        """Fetch index data for this market."""
        import yfinance as yf
        indices_cfg = self.market_config.get("indices", [])
        results = []

        # CN indices: use akshare (yfinance returns incomplete history for .SZ indices)
        if self.market == "CN":
            return self._fetch_cn_indices_akshare(indices_cfg)

        for idx in indices_cfg:
            try:
                with yf_limiter:
                    t = yf.Ticker(idx["symbol"])
                    h = t.history(period="60d")
                if h.empty:
                    continue
                h = h.reset_index()
                h["symbol"] = idx["symbol"]
                h["name"] = idx["name"]
                h["market"] = self.market
                h = h.rename(columns={"Date": "date", "Close": "close"})
                h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None)

                # Compute change
                h["change_pct"] = h["close"].pct_change(1)

                results.append(h[["date", "symbol", "name", "market",
                                   "close", "change_pct"]])
            except Exception as e:
                log.warning(f"Failed to fetch index {idx['symbol']}: {e}")

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def _fetch_cn_indices_akshare(self, indices_cfg):
        """Fetch CN index history via akshare (reliable full history for .SS/.SZ indices)."""
        import akshare as ak

        # Map Yahoo Finance symbol → akshare code
        _AK_CODE = {
            "000001.SS": "sh000001",
            "399001.SZ": "sz399001",
            "399006.SZ": "sz399006",
        }

        results = []
        for idx in indices_cfg:
            symbol = idx["symbol"]
            ak_code = _AK_CODE.get(symbol)
            if not ak_code:
                continue
            try:
                df = ak.stock_zh_index_daily(symbol=ak_code)
                df["date"] = pd.to_datetime(df["date"])
                df = df.tail(60)  # keep last 60 trading days
                df["symbol"] = symbol
                df["name"] = idx["name"]
                df["market"] = "CN"
                df["change_pct"] = df["close"].pct_change(1)
                results.append(df[["date", "symbol", "name", "market",
                                    "close", "change_pct"]])
            except Exception as e:
                log.warning(f"Failed to fetch CN index {symbol} via akshare: {e}")

        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)
