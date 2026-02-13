"""
Macro Events & News Module via Borsapy

Provides access to:
- Economic calendar (TR, US, EU events)
- TCMB inflation data
- Central bank policy rates
- Government bond yields
- Stock-specific KAP news/announcements
- Earnings calendar

Usage:
    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Get economic calendar
    events = client.get_economic_calendar(days_ahead=7)

    # Get inflation data
    inflation = client.get_inflation_data()

    # Get stock news
    news = client.get_stock_news("THYAO", limit=10)
"""

from datetime import datetime, timedelta
import json

import pandas as pd

try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None


class MacroEventsClient:
    """
    Client for macro economic events and news data.

    Provides access to:
    - Economic calendar
    - Inflation data
    - Bond yields
    - Central bank rates
    - Stock news/announcements
    """

    def __init__(self):
        """Initialize the macro events client."""
        if not BORSAPY_AVAILABLE:
            raise ImportError(
                "borsapy is not installed. Install with: pip install borsapy"
            )

        self._ticker_cache: dict[str, bp.Ticker] = {}

    def _get_ticker(self, symbol: str) -> "bp.Ticker":
        """Get or create ticker object."""
        symbol = symbol.upper().split(".")[0]
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = bp.Ticker(symbol)
        return self._ticker_cache[symbol]

    @staticmethod
    def _period_from_days(days_ahead: int) -> str:
        """Map requested forward window to borsapy calendar period."""
        if days_ahead <= 1:
            return "1d"
        if days_ahead <= 7:
            return "1w"
        if days_ahead <= 14:
            return "2w"
        return "1mo"

    @staticmethod
    def _index_info_to_snapshot(info: dict) -> dict:
        """Normalize borsapy Index.info payload to a minimal snapshot."""
        if not isinstance(info, dict):
            return {"value": None, "change_pct": None}

        value = info.get("last")
        change_pct = info.get("change_percent")
        if change_pct is None:
            prev = info.get("close")
            if value is not None and prev not in (None, 0):
                change_pct = ((value - prev) / prev) * 100

        return {"value": value, "change_pct": change_pct}

    # -------------------------------------------------------------------------
    # Economic Calendar
    # -------------------------------------------------------------------------

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        days_back: int = 0,
        countries: list[str] = None,
        importance: str = None,
    ) -> pd.DataFrame:
        """
        Get economic calendar events.

        Args:
            days_ahead: Number of days to look ahead (for filtering)
            days_back: Number of days to look back (for filtering)
            countries: Country codes to filter (e.g., ["TR", "US", "EU"])
            importance: Filter by importance ("high", "medium", "low")

        Returns:
            DataFrame with economic events:
            - date: Event date/time
            - country: Country code
            - event: Event name
            - importance: Event importance
            - actual: Actual value (if released)
            - forecast: Forecasted value
            - previous: Previous value
        """
        try:
            importance_map = {
                "medium": "mid",
                "mid": "mid",
                "high": "high",
                "low": "low",
            }
            bp_importance = importance_map.get(importance.lower(), importance.lower()) if importance else None
            calendar = bp.economic_calendar(
                period=self._period_from_days(days_ahead),
                country=countries,
                importance=bp_importance,
            )

            if calendar is None or (isinstance(calendar, pd.DataFrame) and calendar.empty):
                return pd.DataFrame()

            # Filter by date range if provider returned wider window than requested
            date_col = next((c for c in ["date", "Date", "datetime", "event_date"] if c in calendar.columns), None)
            if date_col:
                try:
                    calendar[date_col] = pd.to_datetime(calendar[date_col], errors="coerce")
                    start = datetime.now() - timedelta(days=days_back)
                    end = datetime.now() + timedelta(days=days_ahead)
                    calendar = calendar[(calendar[date_col] >= start) & (calendar[date_col] <= end)]
                except Exception:
                    pass

            # Defensive filtering if provider-side filters were ignored
            if countries:
                country_col = next(
                    (c for c in ["country", "Country", "country_code"] if c in calendar.columns),
                    None,
                )
                if country_col:
                    wanted = {c.upper() for c in countries}
                    calendar = calendar[
                        calendar[country_col].astype(str).str.upper().isin(wanted)
                    ]

            if importance:
                importance_col = next(
                    (c for c in ["importance", "Importance", "impact"] if c in calendar.columns),
                    None,
                )
                if importance_col:
                    wanted = importance.lower()
                    calendar = calendar[
                        calendar[importance_col].astype(str).str.lower().isin({wanted, "mid" if wanted == "medium" else wanted})
                    ]

            return calendar

        except Exception as e:
            print(f"  Warning: Failed to get economic calendar: {e}")
            return pd.DataFrame()

    def get_upcoming_high_impact_events(
        self,
        days_ahead: int = 7,
        countries: list[str] = None,
    ) -> pd.DataFrame:
        """
        Get upcoming high-impact economic events.

        Args:
            days_ahead: Number of days to look ahead
            countries: Country codes to filter

        Returns:
            DataFrame with high-impact events
        """
        return self.get_economic_calendar(
            days_ahead=days_ahead,
            countries=countries,
            importance="high",
        )

    # -------------------------------------------------------------------------
    # Inflation Data
    # -------------------------------------------------------------------------

    def get_inflation_data(self, periods: int = 24) -> pd.DataFrame:
        """
        Get TCMB inflation data (TUFE - CPI).

        Args:
            periods: Number of monthly periods to retrieve

        Returns:
            DataFrame with inflation data:
            - date: Period date
            - cpi: Consumer Price Index
            - yoy_change: Year-over-year change (%)
            - mom_change: Month-over-month change (%)
        """
        try:
            inflation = bp.Inflation()
            # Use tufe() method which returns historical TUFE data
            data = inflation.tufe()

            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                return pd.DataFrame()

            # Ensure latest records are returned (provider may return descending order).
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.sort_index()
            elif "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
                data = data.sort_values("Date")

            # Limit to requested most recent periods
            if len(data) > periods:
                data = data.tail(periods)

            return data

        except Exception as e:
            print(f"  Warning: Failed to get inflation data: {e}")
            return pd.DataFrame()

    def get_latest_inflation(self) -> dict:
        """
        Get the latest inflation reading.

        Returns:
            Dict with latest inflation data
        """
        try:
            inflation = bp.Inflation()
            # Use latest() which returns the most recent reading as a dict
            latest = inflation.latest()

            if latest is None:
                return {"error": "No data available"}

            result = {
                "timestamp": datetime.now().isoformat(),
            }

            if isinstance(latest, dict):
                result.update(latest)
            else:
                result["value"] = latest

            return result

        except Exception as e:
            # Fallback to getting from historical data
            data = self.get_inflation_data(periods=2)

            if data.empty:
                return {"error": f"No data available: {e}"}

            latest = data.iloc[-1]
            result = {
                "date": str(latest.name) if hasattr(latest, "name") else None,
                "timestamp": datetime.now().isoformat(),
            }

            # Extract available fields
            for col in data.columns:
                result[col.lower().replace(" ", "_")] = latest[col]

            return result

    # -------------------------------------------------------------------------
    # Bond Yields
    # -------------------------------------------------------------------------

    def get_bond_yields(self) -> dict:
        """
        Get Turkish government bond yields via TCMB rates.

        Returns:
            Dict with available rates:
            - policy_rate: TCMB policy rate
            - timestamp: Data timestamp
        """
        try:
            tcmb = bp.TCMB()
            result = {"timestamp": datetime.now().isoformat()}

            if tcmb.policy_rate is not None:
                result["policy_rate"] = tcmb.policy_rate

            if tcmb.overnight:
                result["overnight"] = tcmb.overnight

            if tcmb.late_liquidity:
                result["late_liquidity"] = tcmb.late_liquidity

            rates_table = getattr(tcmb, "rates", None)
            if isinstance(rates_table, pd.DataFrame) and not rates_table.empty:
                result["rates_table"] = rates_table.to_dict("records")

            if len(result) == 1:
                return {"error": "No rate data available", "timestamp": result["timestamp"]}

            return result

        except Exception as e:
            print(f"  Warning: Failed to get bond yields: {e}")
            return {"error": str(e)}

    def get_yield_curve(self) -> pd.DataFrame:
        """
        Get the yield curve data.

        Returns:
            DataFrame with yield curve:
            - maturity: Bond maturity
            - yield: Current yield
        """
        yields = self.get_bond_yields()

        if "error" in yields:
            return pd.DataFrame()

        records: list[dict] = []

        if isinstance(yields.get("rates_table"), list):
            for row in yields["rates_table"]:
                if not isinstance(row, dict):
                    continue
                records.append(
                    {
                        "maturity": row.get("type"),
                        "borrowing": row.get("borrowing"),
                        "lending": row.get("lending"),
                    }
                )

        if not records and "policy_rate" in yields:
            records.append(
                {
                    "maturity": "policy",
                    "borrowing": None,
                    "lending": yields.get("policy_rate"),
                }
            )

        return pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Central Bank Rates
    # -------------------------------------------------------------------------

    def get_tcmb_rates(self) -> dict:
        """
        Get TCMB (Turkish Central Bank) policy rates.

        Returns:
            Dict with policy rates:
            - policy_rate: 1-week repo rate
            - overnight_lending: Overnight lending rate
            - overnight_borrowing: Overnight borrowing rate
            - late_liquidity: Late liquidity window rate
            - timestamp: Data timestamp
        """
        try:
            result = {
                "timestamp": datetime.now().isoformat(),
            }

            tcmb = bp.TCMB()

            if tcmb.policy_rate is not None:
                result["policy_rate"] = tcmb.policy_rate

            if tcmb.overnight:
                result["overnight"] = tcmb.overnight

            if tcmb.late_liquidity:
                result["late_liquidity"] = tcmb.late_liquidity

            rates_table = getattr(tcmb, "rates", None)
            if isinstance(rates_table, pd.DataFrame) and not rates_table.empty:
                result["rates_detail"] = rates_table.to_dict("records")

            return result

        except Exception as e:
            print(f"  Warning: Failed to get TCMB rates: {e}")
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # Stock News & Announcements
    # -------------------------------------------------------------------------

    def get_stock_news(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get KAP announcements/news for a stock.

        Args:
            symbol: Stock symbol
            limit: Maximum number of news items

        Returns:
            List of news items:
            - title: News title
            - date: Publication date
            - summary: News summary
            - url: Link to full announcement
        """
        try:
            ticker = self._get_ticker(symbol)
            news = ticker.news

            if news is None:
                return []

            # Convert to list of dicts if needed
            if isinstance(news, pd.DataFrame):
                news = news.to_dict("records")

            # Limit results
            if isinstance(news, list) and len(news) > limit:
                news = news[:limit]

            return news if news else []

        except Exception as e:
            print(f"  Warning: Failed to get news for {symbol}: {e}")
            return []

    def get_earnings_calendar(
        self,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Get earnings announcement dates for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with earnings dates
        """
        try:
            ticker = self._get_ticker(symbol)
            earnings = ticker.earnings_dates

            if earnings is None:
                return pd.DataFrame()

            return earnings

        except Exception as e:
            print(f"  Warning: Failed to get earnings calendar for {symbol}: {e}")
            return pd.DataFrame()

    def get_analyst_recommendations(
        self,
        symbol: str,
    ) -> dict:
        """
        Get analyst recommendations for a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with analyst data:
            - price_target: Consensus price target
            - recommendation: Consensus recommendation
            - num_analysts: Number of analysts
        """
        try:
            ticker = self._get_ticker(symbol)

            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
            }

            # Get price targets
            try:
                targets = ticker.analyst_price_targets
                if targets is not None:
                    if isinstance(targets, dict):
                        result["price_targets"] = targets
                    elif isinstance(targets, pd.DataFrame) and not targets.empty:
                        result["price_targets"] = targets.to_dict("records")
            except Exception:
                pass

            # Get recommendations
            try:
                recs = ticker.recommendations
                if recs is not None:
                    if isinstance(recs, dict):
                        result["recommendations"] = recs
                    elif isinstance(recs, pd.DataFrame) and not recs.empty:
                        result["recommendations"] = recs.tail(5).to_dict("records")
            except Exception:
                pass

            return result

        except Exception as e:
            print(f"  Warning: Failed to get analyst data for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    # -------------------------------------------------------------------------
    # Eurobonds
    # -------------------------------------------------------------------------

    def get_eurobonds(self) -> pd.DataFrame:
        """
        Get Turkish eurobond data (38+ securities).

        Returns:
            DataFrame with eurobond data:
            - name: Bond name
            - isin: ISIN code
            - maturity: Maturity date
            - coupon: Coupon rate
            - price: Current price
            - yield: Current yield
        """
        try:
            eurobonds = bp.eurobonds()

            if eurobonds is None:
                return pd.DataFrame()

            return eurobonds

        except Exception as e:
            print(f"  Warning: Failed to get eurobonds: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Market Sentiment
    # -------------------------------------------------------------------------

    def get_market_sentiment(self) -> dict:
        """
        Get overall market sentiment indicators.

        Returns:
            Dict with sentiment data:
            - foreign_flow: Net foreign investment flow
            - index_trend: XU100 trend
            - volatility: Market volatility
        """
        result = {
            "timestamp": datetime.now().isoformat(),
        }

        # Get XU100 data for trend
        try:
            xu100 = bp.Index("XU100")
            info = xu100.info
            if info:
                result["xu100"] = self._index_info_to_snapshot(info)
        except Exception:
            result["xu100"] = {"error": "Failed to fetch"}

        # Get USD/TRY for FX sentiment
        try:
            usdtry = bp.FX("USD")
            info = usdtry.info
            if info:
                last = info.get("last") if isinstance(info, dict) else None
                open_ = info.get("open") if isinstance(info, dict) else None
                change_pct = None
                if last is not None and open_ not in (None, 0):
                    change_pct = ((last - open_) / open_) * 100
                result["usdtry"] = {"rate": last, "change_pct": change_pct}
        except Exception:
            result["usdtry"] = {"error": "Failed to fetch"}

        return result

    # -------------------------------------------------------------------------
    # Summary Methods
    # -------------------------------------------------------------------------

    def get_macro_summary(self) -> dict:
        """
        Get comprehensive macro summary.

        Returns:
            Dict with all macro data:
            - inflation: Latest inflation
            - bond_yields: Current yields
            - sentiment: Market sentiment
            - upcoming_events: High-impact events
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
        }

        # Inflation
        summary["inflation"] = self.get_latest_inflation()

        # Bond yields
        summary["bond_yields"] = self.get_bond_yields()

        # Market sentiment
        summary["sentiment"] = self.get_market_sentiment()

        # Upcoming high-impact events
        events = self.get_upcoming_high_impact_events(days_ahead=7)
        if not events.empty:
            summary["upcoming_events"] = events.head(5).to_dict("records")
        else:
            summary["upcoming_events"] = []

        return summary

    def to_json(self, data: dict) -> str:
        """Convert data to JSON string."""
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# Convenience function
def get_macro_client() -> MacroEventsClient:
    """Get a MacroEventsClient instance."""
    return MacroEventsClient()
