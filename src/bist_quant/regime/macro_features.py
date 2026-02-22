"""
Macro Features Module
=====================
Loads and normalizes TCMB + USDTRY macro data for regime augmentation.

Usage:
    macro = MacroFeatures(data_dir='../data')
    macro.load()
    risk_index = macro.compute_risk_index()
    usdtry_mom = macro.compute_usdtry_momentum()
    cds_flag   = macro.compute_cds_stress_flag()
"""

import logging
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from bist_quant.clients.economic_calendar_provider import EconomicCalendarProvider
from bist_quant.clients.derivatives_provider import DerivativesProvider
from bist_quant.clients.fixed_income_provider import FixedIncomeProvider
from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MacroConfig:
    """All macro experiment parameters in one place."""

    # --- USDTRY Crisis Override ---
    usdtry_momentum_window: int = 20          # Days to measure USDTRY appreciation
    usdtry_crisis_threshold: float = 0.08     # 8% depreciation triggers Stress

    # --- CDS Stress Gate ---
    cds_lookback: int = 252                   # Rolling percentile window
    cds_stress_percentile: float = 0.90       # 90th percentile = stress zone
    cds_rapid_change_threshold: float = 0.20  # 20% CDS spike in 20 days

    # --- Composite Risk Index ---
    risk_index_lookback: int = 252            # Rolling percentile window
    risk_index_weights: dict = field(default_factory=lambda: {
        'cds': 0.40,
        'vix': 0.30,
        'usdtry': 0.30,
    })
    tcmb_policy_rate_weight: float = 0.15
    futures_basis_weight: float = 0.10
    use_intraday_fx: bool = False
    intraday_fx_weight: float = 0.10
    intraday_fx_currency: str = "USD"
    intraday_fx_interval: str = "1h"
    intraday_fx_period: str = "5d"
    intraday_fx_momentum_bars: int = 6
    intraday_fx_lookback: int = 5
    yield_curve_inversion_threshold: float = -0.5
    # Risk score → allocation multiplier: 1.0 - scale_factor * risk_pctile
    risk_index_scale_factor: float = 0.50     # Max reduction: 50% at risk=100

    # --- Regime allocation overrides ---
    stress_allocation: float = 0.0            # 0% stocks when macro says Stress


# ============================================================================
# MACRO FEATURES LOADER
# ============================================================================

class MacroFeatures:
    """Load, align, and compute macro indicators for regime augmentation."""

    def __init__(self, data_dir=None, config: MacroConfig = None):
        self.config = config or MacroConfig()

        if data_dir is None:
            here = Path(__file__).resolve().parent
            # Root data folder is at Market Research/data
            self.data_dir = here.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.tcmb = None
        self.usdtry = None
        self._loaded = False
        self._fixed_income_provider: FixedIncomeProvider | None = None
        self._economic_calendar_provider: EconomicCalendarProvider | None = None
        self._derivatives_provider: DerivativesProvider | None = None
        self._fx_enhanced_provider: FXEnhancedProvider | None = None
        self._intraday_fx: pd.DataFrame | None = None

    def load(self) -> None:
        """Load TCMB indicators and USDTRY data."""
        # TCMB macro indicators
        tcmb_path = self.data_dir / "tcmb_indicators.csv"
        if tcmb_path.exists():
            self.tcmb = pd.read_csv(tcmb_path, parse_dates=['Date'])
            self.tcmb = self.tcmb.set_index('Date').sort_index()
            self.tcmb = self.tcmb.ffill()  # Forward-fill macro data (published irregularly)
            logger.info(f"  TCMB: {len(self.tcmb)} days, columns: {list(self.tcmb.columns)}")
        else:
            raise FileNotFoundError(f"TCMB data not found: {tcmb_path}")

        # USDTRY exchange rate
        usdtry_path = self.data_dir / "usdtry_data.csv"
        if usdtry_path.exists():
            usd = pd.read_csv(usdtry_path, parse_dates=['Date'])
            usd = usd.set_index('Date').sort_index()
            self.usdtry = usd['USDTRY'].astype(float)
            self.usdtry = self.usdtry.ffill()
            logger.info(f"  USDTRY: {len(self.usdtry)} days")
        else:
            raise FileNotFoundError(f"USDTRY data not found: {usdtry_path}")

        self._loaded = True
        try:
            self.load_from_borsapy(self.fixed_income)
        except Exception as exc:
            logger.info("  Fixed income live supplement skipped: %s", exc)
        try:
            self.load_derivatives_from_borsapy(DerivativesProvider())
        except Exception as exc:
            logger.info("  Derivatives live supplement skipped: %s", exc)
        if self.config.use_intraday_fx:
            try:
                self.load_intraday_fx_from_borsapy(FXEnhancedProvider())
            except Exception as exc:
                logger.info("  Intraday FX live supplement skipped: %s", exc)

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first")

    @property
    def economic_calendar(self) -> EconomicCalendarProvider:
        if self._economic_calendar_provider is None:
            self._economic_calendar_provider = EconomicCalendarProvider()
        return self._economic_calendar_provider

    @property
    def fixed_income(self) -> FixedIncomeProvider:
        if self._fixed_income_provider is None:
            self._fixed_income_provider = FixedIncomeProvider()
        return self._fixed_income_provider

    @property
    def derivatives(self) -> DerivativesProvider:
        if self._derivatives_provider is None:
            self._derivatives_provider = DerivativesProvider()
        return self._derivatives_provider

    @property
    def fx_enhanced(self) -> FXEnhancedProvider:
        if self._fx_enhanced_provider is None:
            self._fx_enhanced_provider = FXEnhancedProvider()
        return self._fx_enhanced_provider

    @staticmethod
    def _rolling_percentile(series: pd.Series, lookback: int) -> pd.Series:
        return series.rolling(lookback, min_periods=lookback // 2).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

    @staticmethod
    def _to_percent_series(series: pd.Series) -> pd.Series:
        clean = pd.to_numeric(series, errors='coerce')
        if clean.dropna().empty:
            return clean
        median_abs = clean.dropna().abs().median()
        if median_abs <= 1.5:
            return clean * 100.0
        return clean

    @staticmethod
    def _latest_value(series: pd.Series) -> float | None:
        clean = pd.to_numeric(series, errors='coerce').dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])

    def _upsert_tcmb_point(self, column: str, value: float) -> None:
        if self.tcmb is None:
            return
        timestamp = pd.Timestamp(datetime.now().date())
        if timestamp not in self.tcmb.index:
            self.tcmb.loc[timestamp] = pd.NA
        self.tcmb.loc[timestamp, column] = float(value)

    def load_from_borsapy(self, provider: FixedIncomeProvider) -> None:
        """Supplement static CSVs with live bond/TCMB data."""
        self._check_loaded()

        # Policy-rate history enriches macro regime features.
        tcmb_hist = provider.get_tcmb_history(rate_type="policy", period="1y")
        if not tcmb_hist.empty:
            hist = tcmb_hist.copy()
            hist.index = pd.to_datetime(hist.index, errors='coerce')
            hist = hist[hist.index.notna()]
            if "rate" in hist.columns:
                series = pd.to_numeric(hist["rate"], errors='coerce')
            else:
                numeric_cols = hist.select_dtypes(include="number").columns.tolist()
                series = pd.to_numeric(hist[numeric_cols[0]], errors='coerce') if numeric_cols else pd.Series(dtype=float)
            series = self._to_percent_series(series).dropna()
            if not series.empty:
                self.tcmb["policy_rate"] = self.tcmb.get("policy_rate", pd.Series(index=self.tcmb.index, dtype=float))
                self.tcmb.loc[series.index, "policy_rate"] = series

        # Snapshot policy rates can still be useful even if history fetch fails.
        tcmb_rates = provider.get_tcmb_rates()
        policy_now = self._to_percent_series(pd.Series([tcmb_rates.get("policy_rate")])).dropna()
        if not policy_now.empty:
            self._upsert_tcmb_point("policy_rate", float(policy_now.iloc[0]))

        # Yield-curve slope (10Y - 2Y), in percentage points.
        yields = provider.get_bond_yields()
        y2 = yields.get("2Y")
        y10 = yields.get("10Y")
        if y2 is not None and y10 is not None:
            self._upsert_tcmb_point("yield_curve_slope", float(y10 - y2))
        else:
            curve = provider.get_yield_curve()
            if not curve.empty and {"maturity", "yield"}.issubset(set(curve.columns)):
                curve_map = {
                    str(row["maturity"]).upper(): float(row["yield"])
                    for _, row in curve.iterrows()
                    if pd.notna(row.get("maturity")) and pd.notna(row.get("yield"))
                }
                y2_alt = curve_map.get("2Y")
                y10_alt = curve_map.get("10Y")
                if y2_alt is not None and y10_alt is not None:
                    self._upsert_tcmb_point("yield_curve_slope", float(y10_alt - y2_alt))

        # Eurobond spread proxy for sovereign stress.
        spread = provider.get_spread_index()
        if spread is not None and math.isfinite(spread):
            self._upsert_tcmb_point("eurobond_spread", float(spread))

        self.tcmb = self.tcmb.sort_index().ffill()

    def _refresh_eurobond_spread_snapshot(self) -> None:
        """Best-effort refresh of latest USD eurobond spread proxy."""
        if self.tcmb is None:
            return
        try:
            spread = self.fixed_income.get_spread_index()
        except Exception as exc:
            logger.info("  Eurobond spread refresh skipped: %s", exc)
            return
        if spread is None or not math.isfinite(spread):
            return
        self._upsert_tcmb_point("eurobond_spread", float(spread))
        self.tcmb = self.tcmb.sort_index().ffill()

    def load_derivatives_from_borsapy(self, provider: DerivativesProvider) -> None:
        """Supplement static CSVs with live VIOP basis/sentiment data."""
        self._check_loaded()

        premium = provider.get_index_futures_premium()
        basis_points = pd.to_numeric(
            pd.Series([premium.get("premium_points")]),
            errors='coerce',
        ).dropna()
        basis_pct = pd.to_numeric(
            pd.Series([premium.get("premium_pct")]),
            errors='coerce',
        ).dropna()
        if not basis_points.empty:
            self._upsert_tcmb_point("futures_basis", float(basis_points.iloc[0]))
        if not basis_pct.empty:
            self._upsert_tcmb_point("futures_basis_pct", float(basis_pct.iloc[0]))

        put_call_ratio = pd.to_numeric(
            pd.Series([provider.get_put_call_ratio()]),
            errors='coerce',
        ).dropna()
        if not put_call_ratio.empty:
            self._upsert_tcmb_point("put_call_ratio", float(put_call_ratio.iloc[0]))

        self.tcmb = self.tcmb.sort_index().ffill()

    def load_intraday_fx_from_borsapy(self, provider: FXEnhancedProvider) -> None:
        """Load intraday FX bars for rapid-stress momentum features."""
        self._check_loaded()

        data = provider.get_intraday(
            currency=self.config.intraday_fx_currency,
            interval=self.config.intraday_fx_interval,
            period=self.config.intraday_fx_period,
        )
        if data.empty:
            self._intraday_fx = None
            return

        work = data.copy()
        timestamp_col = next(
            (
                col
                for col in ("timestamp", "datetime", "date", "time")
                if col in work.columns
            ),
            None,
        )
        if timestamp_col is None and isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={"index": "timestamp"})
            timestamp_col = "timestamp"

        close_col = next(
            (col for col in ("close", "Close", "last", "price", "value") if col in work.columns),
            None,
        )
        if timestamp_col is None or close_col is None:
            self._intraday_fx = None
            return

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(work[timestamp_col], errors="coerce"),
                "close": pd.to_numeric(work[close_col], errors="coerce"),
            }
        ).dropna(subset=["timestamp", "close"])

        if frame.empty:
            self._intraday_fx = None
            return

        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        self._intraday_fx = frame.reset_index(drop=True)

    def compute_intraday_fx_momentum(self) -> pd.Series:
        """
        Rapid FX momentum from intraday bars.

        Uses a bar-based pct-change on intraday close prices, then maps the
        latest value per day to the macro daily index.
        """
        self._check_loaded()

        if not self.config.use_intraday_fx:
            return pd.Series(dtype=float, name="intraday_fx_momentum")

        if self._intraday_fx is None or self._intraday_fx.empty:
            try:
                self.load_intraday_fx_from_borsapy(self.fx_enhanced)
            except Exception as exc:
                logger.info("  Intraday FX fetch skipped: %s", exc)
                return pd.Series(dtype=float, name="intraday_fx_momentum")

        if self._intraday_fx is None or self._intraday_fx.empty:
            return pd.Series(dtype=float, name="intraday_fx_momentum")

        frame = self._intraday_fx.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        if frame.empty:
            return pd.Series(dtype=float, name="intraday_fx_momentum")

        series = frame.set_index("timestamp")["close"]
        bars = max(1, int(self.config.intraday_fx_momentum_bars))
        momentum = series.pct_change(bars)
        daily = momentum.resample("1D").last().dropna()
        daily.name = "intraday_fx_momentum"
        return daily

    # ----------------------------------------------------------------
    # Feature 1: USDTRY Momentum (for Crisis Override)
    # ----------------------------------------------------------------

    def compute_usdtry_momentum(self) -> pd.Series:
        """
        20-day % change in USDTRY.
        Positive = lira depreciating (bad for equities).
        """
        self._check_loaded()
        window = self.config.usdtry_momentum_window
        mom = self.usdtry.pct_change(window)
        mom.name = 'usdtry_momentum'
        return mom

    def compute_usdtry_crisis_flag(self) -> pd.Series:
        """
        Binary flag: True when USDTRY depreciation exceeds crisis threshold.
        Triggers only in extreme moves (2018 crisis, 2021 crash, etc).
        """
        mom = self.compute_usdtry_momentum()
        flag = mom > self.config.usdtry_crisis_threshold
        flag.name = 'usdtry_crisis'
        return flag

    # ----------------------------------------------------------------
    # Feature 2: CDS Stress (for CDS Gate)
    # ----------------------------------------------------------------

    def compute_cds_percentile(self) -> pd.Series:
        """
        Rolling percentile rank of CDS proxy over lookback window.
        Higher = more credit stress.
        """
        self._check_loaded()
        if 'cds_proxy' in self.tcmb.columns:
            cds = self.tcmb['cds_proxy'].dropna()
        elif 'eurobond_spread' in self.tcmb.columns:
            cds = self.tcmb['eurobond_spread'].dropna()
        else:
            return pd.Series(dtype=float, name='cds_percentile')
        lookback = self.config.cds_lookback
        pctile = self._rolling_percentile(cds, lookback)
        pctile.name = 'cds_percentile'
        return pctile

    def compute_cds_stress_flag(self) -> pd.Series:
        """
        Binary flag: True when CDS is in stress zone.
        Triggers on either:
          1. CDS percentile > 90th, OR
          2. CDS 20-day change > 20% (rapid spike)
        """
        self._check_loaded()
        pctile = self.compute_cds_percentile()
        if pctile.empty:
            return pd.Series(dtype=bool, name='cds_stress')
        level_stress = pctile >= self.config.cds_stress_percentile

        # Rapid change
        cds_change = (
            self.tcmb['cds_change_20d']
            if 'cds_change_20d' in self.tcmb.columns
            else pd.Series(0, index=self.tcmb.index)
        )
        rapid_stress = cds_change > self.config.cds_rapid_change_threshold

        flag = (level_stress | rapid_stress).reindex(pctile.index).fillna(False)
        flag.name = 'cds_stress'
        return flag

    def compute_eurobond_stress(self) -> pd.Series:
        """
        Sovereign spread stress flag from eurobond yields.

        Pipeline:
          1. Avg USD eurobond ask_yield (snapshot spread proxy).
          2. Rolling percentile over configured lookback.
          3. Stress flag when percentile >= configured stress threshold.
        """
        self._check_loaded()
        self._refresh_eurobond_spread_snapshot()
        if 'eurobond_spread' not in self.tcmb.columns:
            return pd.Series(dtype=bool, name='eurobond_stress')
        spread = pd.to_numeric(self.tcmb['eurobond_spread'], errors='coerce').dropna()
        if spread.empty:
            return pd.Series(dtype=bool, name='eurobond_stress')
        pct = self._rolling_percentile(spread, self.config.cds_lookback).reindex(spread.index)
        stress_flag = (pct >= self.config.cds_stress_percentile).fillna(False).astype(bool)
        stress_flag.name = 'eurobond_stress'
        return stress_flag

    def compute_yield_curve_signal(self) -> pd.Series:
        """Yield curve slope (10Y - 2Y). Negative values indicate stress."""
        self._check_loaded()
        if 'yield_curve_slope' in self.tcmb.columns:
            slope = pd.to_numeric(self.tcmb['yield_curve_slope'], errors='coerce')
            slope = self._to_percent_series(slope)
            slope.name = 'yield_curve_slope'
            return slope

        ten_candidates = (
            'tr_10y_yield',
            'turkey_10y_yield',
            'bond_10y',
            '10y',
        )
        two_candidates = (
            'tr_2y_yield',
            'turkey_2y_yield',
            'bond_2y',
            '2y',
        )
        ten_col = next((c for c in ten_candidates if c in self.tcmb.columns), None)
        two_col = next((c for c in two_candidates if c in self.tcmb.columns), None)
        if ten_col is None or two_col is None:
            return pd.Series(dtype=float, name='yield_curve_slope')

        ten = self._to_percent_series(pd.to_numeric(self.tcmb[ten_col], errors='coerce'))
        two = self._to_percent_series(pd.to_numeric(self.tcmb[two_col], errors='coerce'))
        slope = ten - two
        slope.name = 'yield_curve_slope'
        return slope

    def compute_policy_rate_momentum(self) -> pd.Series:
        """Rate of change in TCMB policy rate over the configured momentum window."""
        self._check_loaded()
        rate_col = next(
            (c for c in ('policy_rate', 'tcmb_policy_rate', 'policy') if c in self.tcmb.columns),
            None,
        )
        if rate_col is None:
            return pd.Series(dtype=float, name='policy_rate_momentum')

        policy = self._to_percent_series(pd.to_numeric(self.tcmb[rate_col], errors='coerce'))
        momentum = policy.diff(self.config.usdtry_momentum_window)
        momentum.name = 'policy_rate_momentum'
        return momentum

    def compute_futures_basis_signal(self) -> pd.Series:
        """Raw futures basis percent (positive contango, negative backwardation)."""
        self._check_loaded()
        if 'futures_basis_pct' not in self.tcmb.columns:
            return pd.Series(dtype=float, name='futures_basis_pct')
        basis = pd.to_numeric(self.tcmb['futures_basis_pct'], errors='coerce')
        basis.name = 'futures_basis_pct'
        return basis

    def compute_futures_basis_risk(self) -> pd.Series:
        """
        Futures-basis stress score as rolling percentile.

        Negative basis (backwardation) is treated as higher risk, so the
        percentile is computed on the sign-inverted basis series.
        """
        self._check_loaded()
        basis = self.compute_futures_basis_signal().dropna()
        if basis.empty:
            return pd.Series(dtype=float, name='futures_basis_risk')
        stress_input = -basis
        pct = self._rolling_percentile(stress_input, self.config.risk_index_lookback)
        pct.name = 'futures_basis_risk'
        return pct

    # ----------------------------------------------------------------
    # Feature 3: Composite Risk Index (for Risk Index Scaling)
    # ----------------------------------------------------------------

    def compute_risk_index(self) -> pd.Series:
        """
        Composite macro risk score (0.0 to 1.0).
        
        Components (rolling percentile rank, higher = riskier):
          - CDS proxy:         40% weight
          - VIX:               30% weight
          - USDTRY momentum:   30% weight
        
        Returns a scalar allocation multiplier: 1.0 (no risk) → 0.5 (max risk)
        """
        self._check_loaded()
        lookback = self.config.risk_index_lookback
        configured_weights = dict(self.config.risk_index_weights)

        factors: dict[str, pd.Series] = {}
        cds_pct = self.compute_cds_percentile().dropna()
        if not cds_pct.empty:
            factors['cds'] = cds_pct

        if 'vix' in self.tcmb.columns:
            vix_pct = self._rolling_percentile(self.tcmb['vix'].dropna(), lookback)
            if not vix_pct.empty:
                factors['vix'] = vix_pct

        usdtry_pct = self._rolling_percentile(self.compute_usdtry_momentum().dropna(), lookback)
        if not usdtry_pct.empty:
            factors['usdtry'] = usdtry_pct

        policy_mom = self.compute_policy_rate_momentum().dropna()
        if not policy_mom.empty and self.config.tcmb_policy_rate_weight > 0:
            factors['policy_rate'] = self._rolling_percentile(policy_mom, lookback)
            configured_weights['policy_rate'] = self.config.tcmb_policy_rate_weight

        futures_basis = self.compute_futures_basis_risk().dropna()
        if not futures_basis.empty and self.config.futures_basis_weight > 0:
            factors['futures_basis'] = futures_basis
            configured_weights['futures_basis'] = self.config.futures_basis_weight

        intraday_fx = self.compute_intraday_fx_momentum().dropna()
        if (
            not intraday_fx.empty
            and self.config.use_intraday_fx
            and self.config.intraday_fx_weight > 0
        ):
            intraday_lookback = max(2, int(self.config.intraday_fx_lookback))
            factors['fx_intraday'] = self._rolling_percentile(intraday_fx, intraday_lookback)
            configured_weights['fx_intraday'] = self.config.intraday_fx_weight

        if not factors:
            return pd.Series(dtype=float, name='macro_risk_index')

        available_keys = [key for key in configured_weights if key in factors]
        if not available_keys:
            return pd.Series(dtype=float, name='macro_risk_index')

        combined = pd.DataFrame({key: factors[key] for key in available_keys})
        if combined.empty:
            return pd.Series(dtype=float, name='macro_risk_index')

        total_weight = sum(configured_weights[key] for key in available_keys)
        if total_weight <= 0:
            return pd.Series(dtype=float, name='macro_risk_index')

        normalized = {key: configured_weights[key] / total_weight for key in available_keys}
        weights = pd.Series(normalized)
        weighted = combined.mul(weights, axis=1)
        weight_mask = combined.notna().mul(weights, axis=1)
        effective_weight = weight_mask.sum(axis=1)
        risk_score = weighted.sum(axis=1, min_count=1) / effective_weight.replace(0.0, pd.NA)
        risk_score = risk_score.dropna()
        risk_score.name = 'macro_risk_index'
        return risk_score

    def compute_risk_multiplier(self) -> pd.Series:
        """
        Allocation multiplier derived from risk index.
        1.0 (low risk) → 0.5 (high risk).
        
        Formula: multiplier = 1.0 - scale_factor * risk_score
        """
        risk = self.compute_risk_index()
        scale = self.config.risk_index_scale_factor
        multiplier = 1.0 - scale * risk
        multiplier = multiplier.clip(lower=0.0, upper=1.0)
        multiplier.name = 'risk_multiplier'
        return multiplier

    @staticmethod
    def _extract_event_datetimes(events: pd.DataFrame) -> pd.Series:
        if events.empty or 'Date' not in events.columns:
            return pd.Series(dtype='datetime64[ns]')

        date_text = events['Date'].astype('string').fillna('').str.strip()
        time_text = (
            events['Time'].astype('string').fillna('').str.strip()
            if 'Time' in events.columns
            else pd.Series('', index=events.index, dtype='string')
        )
        missing_time = time_text.eq('')
        combined = pd.to_datetime((date_text + ' ' + time_text).str.strip(), errors='coerce')
        fallback = pd.to_datetime(events['Date'], errors='coerce')
        out = combined.fillna(fallback)
        out = out.where(~missing_time, fallback + pd.Timedelta(hours=23, minutes=59))
        try:
            if out.dt.tz is not None:
                out = out.dt.tz_convert(None)
        except Exception:
            pass
        return out

    def compute_event_risk_flag(
        self,
        days_ahead: int = 7,
        hours_before: int = 24,
        countries: tuple[str, ...] = ('TR', 'US'),
    ) -> pd.Series:
        """
        Binary pre-event risk flag before high-impact events.

        The series is aligned to the daily macro index and flips to True
        on days inside the ``hours_before`` lead window for TR/US events.
        """
        self._check_loaded()

        if self.tcmb is None:
            return pd.Series(dtype=bool, name='event_risk')

        index = pd.DatetimeIndex(self.tcmb.index).sort_values().unique()
        flag = pd.Series(False, index=index, name='event_risk', dtype=bool)
        if flag.empty:
            return flag

        try:
            events = self.economic_calendar.get_high_impact_events(days_ahead=days_ahead)
        except Exception as exc:
            logger.info("  Economic calendar fetch skipped: %s", exc)
            return flag

        if events.empty:
            return flag

        if countries and 'Country' in events.columns:
            wanted = {str(country).strip().upper() for country in countries if str(country).strip()}
            if wanted:
                events = events[
                    events['Country'].astype('string').fillna('').str.upper().isin(wanted)
                ]
        if events.empty:
            return flag

        event_dt = self._extract_event_datetimes(events).dropna()
        if event_dt.empty:
            return flag

        window = pd.Timedelta(hours=max(0, int(hours_before)))
        day_index = pd.DatetimeIndex(flag.index).normalize()
        for ts in event_dt:
            start_day = (pd.Timestamp(ts) - window).normalize()
            end_day = pd.Timestamp(ts).normalize()
            mask = (day_index >= start_day) & (day_index <= end_day)
            if mask.any():
                flag.loc[flag.index[mask]] = True

        return flag.astype(bool)

    # ----------------------------------------------------------------
    # Summary / diagnostics
    # ----------------------------------------------------------------

    def summary(self) -> dict:
        """Return summary stats for all macro features."""
        self._check_loaded()
        usdtry_mom = self.compute_usdtry_momentum()
        cds_pct = self.compute_cds_percentile()
        cds_stress = self.compute_cds_stress_flag()
        risk_index = self.compute_risk_index()
        risk_mult = self.compute_risk_multiplier()
        yield_curve = self.compute_yield_curve_signal()
        policy_mom = self.compute_policy_rate_momentum()
        eurobond_stress = self.compute_eurobond_stress()
        futures_basis = self.compute_futures_basis_signal()
        intraday_fx_mom = self.compute_intraday_fx_momentum()
        put_call_ratio = (
            pd.to_numeric(self.tcmb['put_call_ratio'], errors='coerce')
            if 'put_call_ratio' in self.tcmb.columns
            else pd.Series(dtype=float)
        )

        out = {
            'usdtry_current': self._latest_value(self.usdtry),
            'usdtry_20d_change': self._latest_value(usdtry_mom),
            'usdtry_crisis_active': bool(usdtry_mom.dropna().iloc[-1] > self.config.usdtry_crisis_threshold) if not usdtry_mom.dropna().empty else None,
            'cds_current': self._latest_value(self.tcmb['cds_proxy']) if 'cds_proxy' in self.tcmb.columns else None,
            'cds_percentile': self._latest_value(cds_pct),
            'cds_stress_active': bool(cds_stress.dropna().iloc[-1]) if not cds_stress.dropna().empty else None,
            'risk_index': self._latest_value(risk_index),
            'risk_multiplier': self._latest_value(risk_mult),
            'yield_curve_slope': self._latest_value(yield_curve),
            'yield_curve_inverted': bool(yield_curve.dropna().iloc[-1] < self.config.yield_curve_inversion_threshold) if not yield_curve.dropna().empty else None,
            'policy_rate': self._latest_value(self.tcmb['policy_rate']) if 'policy_rate' in self.tcmb.columns else None,
            'policy_rate_momentum': self._latest_value(policy_mom),
            'eurobond_spread': self._latest_value(self.tcmb['eurobond_spread']) if 'eurobond_spread' in self.tcmb.columns else None,
            'eurobond_stress': bool(eurobond_stress.dropna().iloc[-1]) if not eurobond_stress.dropna().empty else None,
            'futures_basis': self._latest_value(
                pd.to_numeric(self.tcmb['futures_basis'], errors='coerce')
            ) if 'futures_basis' in self.tcmb.columns else None,
            'futures_basis_pct': self._latest_value(futures_basis),
            'intraday_fx_momentum': self._latest_value(intraday_fx_mom),
            'intraday_fx_enabled': bool(self.config.use_intraday_fx),
            'put_call_ratio': self._latest_value(put_call_ratio),
        }
        return out
