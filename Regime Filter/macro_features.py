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

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import warnings
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
            self.data_dir = here.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self.tcmb = None
        self.usdtry = None
        self._loaded = False

    def load(self) -> None:
        """Load TCMB indicators and USDTRY data."""
        # TCMB macro indicators
        tcmb_path = self.data_dir / "tcmb_indicators.csv"
        if tcmb_path.exists():
            self.tcmb = pd.read_csv(tcmb_path, parse_dates=['Date'])
            self.tcmb = self.tcmb.set_index('Date').sort_index()
            self.tcmb = self.tcmb.ffill()  # Forward-fill macro data (published irregularly)
            print(f"  TCMB: {len(self.tcmb)} days, columns: {list(self.tcmb.columns)}")
        else:
            raise FileNotFoundError(f"TCMB data not found: {tcmb_path}")

        # USDTRY exchange rate
        usdtry_path = self.data_dir / "usdtry_data.csv"
        if usdtry_path.exists():
            usd = pd.read_csv(usdtry_path, parse_dates=['Date'])
            usd = usd.set_index('Date').sort_index()
            self.usdtry = usd['USDTRY'].astype(float)
            self.usdtry = self.usdtry.ffill()
            print(f"  USDTRY: {len(self.usdtry)} days")
        else:
            raise FileNotFoundError(f"USDTRY data not found: {usdtry_path}")

        self._loaded = True

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first")

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
        cds = self.tcmb['cds_proxy'].dropna()
        lookback = self.config.cds_lookback
        pctile = cds.rolling(lookback, min_periods=lookback // 2).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
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
        level_stress = pctile >= self.config.cds_stress_percentile

        # Rapid change
        cds_change = self.tcmb['cds_change_20d'] if 'cds_change_20d' in self.tcmb.columns else pd.Series(0, index=self.tcmb.index)
        rapid_stress = cds_change > self.config.cds_rapid_change_threshold

        flag = (level_stress | rapid_stress).reindex(pctile.index).fillna(False)
        flag.name = 'cds_stress'
        return flag

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
        weights = self.config.risk_index_weights

        def _rolling_pctile(series: pd.Series) -> pd.Series:
            return series.rolling(lookback, min_periods=lookback // 2).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

        # CDS percentile (higher = more stress)
        cds_pct = _rolling_pctile(self.tcmb['cds_proxy'].dropna())

        # VIX percentile (higher = more fear)
        vix_pct = _rolling_pctile(self.tcmb['vix'].dropna())

        # USDTRY momentum percentile (higher = faster depreciation = worse)
        usdtry_mom = self.compute_usdtry_momentum()
        usdtry_pct = _rolling_pctile(usdtry_mom.dropna())

        # Align all on common dates
        combined = pd.DataFrame({
            'cds': cds_pct,
            'vix': vix_pct,
            'usdtry': usdtry_pct,
        }).dropna()

        # Weighted composite
        risk_score = sum(combined[k] * w for k, w in weights.items())
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

    # ----------------------------------------------------------------
    # Summary / diagnostics
    # ----------------------------------------------------------------

    def summary(self) -> dict:
        """Return summary stats for all macro features."""
        self._check_loaded()
        return {
            'usdtry_current': float(self.usdtry.iloc[-1]),
            'usdtry_20d_change': float(self.compute_usdtry_momentum().iloc[-1]),
            'usdtry_crisis_active': bool(self.compute_usdtry_crisis_flag().iloc[-1]),
            'cds_current': float(self.tcmb['cds_proxy'].dropna().iloc[-1]),
            'cds_percentile': float(self.compute_cds_percentile().dropna().iloc[-1]),
            'cds_stress_active': bool(self.compute_cds_stress_flag().iloc[-1]),
            'risk_index': float(self.compute_risk_index().iloc[-1]),
            'risk_multiplier': float(self.compute_risk_multiplier().iloc[-1]),
        }
