"""
Portfolio Analytics Module

Comprehensive portfolio risk and performance metrics including:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Beta (market sensitivity)
- Alpha (risk-adjusted excess return)
- Calmar Ratio
- Information Ratio
- Correlation Matrix
- Value at Risk (VaR)
- Expected Shortfall (CVaR)

Can use either:
1. Local calculations with price data (primary)
2. Borsapy Portfolio class (when available)

Usage:
    from Models.analytics import PortfolioAnalytics

    # From returns series
    analytics = PortfolioAnalytics.from_returns(portfolio_returns, benchmark_returns)
    metrics = analytics.get_all_metrics()

    # From holdings
    analytics = PortfolioAnalytics.from_holdings(
        holdings={"THYAO": 100, "AKBNK": 200},
        close_df=price_data
    )
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Union
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FETCHER_DIR = PROJECT_ROOT / "data" / "Fetcher-Scrapper"
if str(FETCHER_DIR) not in sys.path:
    sys.path.insert(0, str(FETCHER_DIR))

# Check borsapy availability
try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None


# =============================================================================
# Constants
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.40  # Turkish risk-free rate (~40% as of 2026)


# =============================================================================
# Standalone Metric Functions
# =============================================================================

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev

    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Annualized Sharpe Ratio
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    # Convert annual risk-free to daily
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - daily_rf
    mean_excess = excess_returns.mean()
    std_returns = returns.std()

    if std_returns == 0:
        return 0.0

    # Annualize
    sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate annualized Sortino Ratio.

    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation

    Only considers downside volatility (negative returns).

    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino Ratio
    """
    if returns.empty:
        return 0.0

    # Convert annual risk-free to daily
    daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - daily_rf
    mean_excess = excess_returns.mean()

    # Downside deviation (only negative returns)
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return 0.0

    downside_std = negative_returns.std()
    if downside_std == 0:
        return 0.0

    # Annualize
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_max_drawdown(
    returns: pd.Series = None,
    prices: pd.Series = None,
) -> tuple[float, Optional[datetime], Optional[datetime]]:
    """
    Calculate Maximum Drawdown.

    Args:
        returns: Daily returns series (optional)
        prices: Price series (optional, used if returns not provided)

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    if prices is not None:
        cumulative = prices
    elif returns is not None:
        cumulative = (1 + returns).cumprod()
    else:
        return 0.0, None, None

    if cumulative.empty:
        return 0.0, None, None

    # Running maximum
    running_max = cumulative.expanding().max()

    # Drawdown series
    drawdown = (cumulative - running_max) / running_max

    # Maximum drawdown
    max_dd = drawdown.min()

    # Find dates
    trough_idx = drawdown.idxmin()
    peak_idx = cumulative[:trough_idx].idxmax() if trough_idx else None

    return float(max_dd), peak_idx, trough_idx


def calculate_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Calculate Beta (market sensitivity).

    Beta = Cov(Portfolio, Benchmark) / Var(Benchmark)

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        Beta coefficient
    """
    # Align series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 1.0

    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    cov = port_ret.cov(bench_ret)
    var = bench_ret.var()

    if var == 0:
        return 1.0

    return float(cov / var)


def calculate_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Jensen's Alpha (risk-adjusted excess return).

    Alpha = Portfolio Return - [Risk Free + Beta * (Benchmark - Risk Free)]

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Alpha
    """
    # Calculate beta
    beta = calculate_beta(portfolio_returns, benchmark_returns)

    # Annualize returns
    port_annual = (1 + portfolio_returns.mean()) ** periods_per_year - 1
    bench_annual = (1 + benchmark_returns.mean()) ** periods_per_year - 1

    # CAPM expected return
    expected = risk_free_rate + beta * (bench_annual - risk_free_rate)

    # Alpha
    alpha = port_annual - expected
    return float(alpha)


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Calmar Ratio.

    Calmar = Annualized Return / |Max Drawdown|

    Args:
        returns: Daily returns series
        periods_per_year: Number of periods per year

    Returns:
        Calmar Ratio
    """
    if returns.empty:
        return 0.0

    # Annualized return
    annual_return = (1 + returns.mean()) ** periods_per_year - 1

    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(returns=returns)

    if max_dd == 0:
        return 0.0

    return float(annual_return / abs(max_dd))


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Information Ratio.

    IR = (Portfolio Return - Benchmark Return) / Tracking Error

    Args:
        portfolio_returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized Information Ratio
    """
    # Align series
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 0.0

    port_ret = aligned.iloc[:, 0]
    bench_ret = aligned.iloc[:, 1]

    # Active return (excess over benchmark)
    active_return = port_ret - bench_ret

    # Tracking error
    tracking_error = active_return.std()
    if tracking_error == 0:
        return 0.0

    # Information ratio (annualized)
    ir = (active_return.mean() / tracking_error) * np.sqrt(periods_per_year)
    return float(ir)


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Daily returns series
        confidence: Confidence level (default 95%)
        method: "historical" or "parametric"

    Returns:
        VaR as a positive number (potential loss)
    """
    if returns.empty:
        return 0.0

    if method == "parametric":
        # Assuming normal distribution
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mean + z_score * std)
    else:
        # Historical VaR
        var = -returns.quantile(1 - confidence)

    return float(var)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that loss exceeds VaR.

    Args:
        returns: Daily returns series
        confidence: Confidence level (default 95%)

    Returns:
        CVaR as a positive number
    """
    if returns.empty:
        return 0.0

    var = calculate_var(returns, confidence, method="historical")
    tail_returns = returns[returns <= -var]

    if tail_returns.empty:
        return var

    cvar = -tail_returns.mean()
    return float(cvar)


def calculate_correlation_matrix(
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple assets.

    Args:
        returns_df: DataFrame with returns (columns = assets)

    Returns:
        Correlation matrix DataFrame
    """
    return returns_df.corr()


def calculate_rolling_metrics(
    returns: pd.Series,
    window: int = 63,  # ~3 months
    benchmark_returns: pd.Series = None,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Daily returns series
        window: Rolling window size
        benchmark_returns: Optional benchmark for beta/alpha

    Returns:
        DataFrame with rolling metrics
    """
    metrics = pd.DataFrame(index=returns.index)

    # Rolling volatility (annualized)
    metrics["volatility"] = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Rolling Sharpe
    daily_rf = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS_PER_YEAR) - 1

    def rolling_sharpe(r):
        if len(r) < window // 2:
            return np.nan
        excess = r - daily_rf
        if r.std() == 0:
            return 0
        return (excess.mean() / r.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    metrics["sharpe"] = returns.rolling(window).apply(rolling_sharpe, raw=False)

    # Rolling max drawdown
    def rolling_max_dd(r):
        cumulative = (1 + r).cumprod()
        running_max = cumulative.expanding().max()
        dd = (cumulative - running_max) / running_max
        return dd.min()

    metrics["max_drawdown"] = returns.rolling(window).apply(rolling_max_dd, raw=False)

    # Rolling beta if benchmark provided
    if benchmark_returns is not None:
        def rolling_beta(port_ret, bench_ret):
            aligned = pd.concat([port_ret, bench_ret], axis=1).dropna()
            if len(aligned) < window // 2:
                return np.nan
            cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
            var = aligned.iloc[:, 1].var()
            return cov / var if var != 0 else 1.0

        betas = []
        for i in range(len(returns)):
            if i < window - 1:
                betas.append(np.nan)
            else:
                port_slice = returns.iloc[i - window + 1:i + 1]
                bench_slice = benchmark_returns.iloc[i - window + 1:i + 1]
                betas.append(rolling_beta(port_slice, bench_slice))

        metrics["beta"] = betas

    return metrics


# =============================================================================
# Portfolio Analytics Class
# =============================================================================

class PortfolioAnalytics:
    """
    Comprehensive portfolio analytics.

    Supports multiple initialization methods:
    - From returns series
    - From holdings and price data
    - From borsapy Portfolio object
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = RISK_FREE_RATE,
        name: str = "Portfolio",
    ):
        """
        Initialize from returns series.

        Args:
            returns: Daily portfolio returns
            benchmark_returns: Daily benchmark returns (optional)
            risk_free_rate: Annual risk-free rate
            name: Portfolio name
        """
        self.returns = returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.name = name

        # Cache for computed metrics
        self._metrics_cache: dict = {}

    @classmethod
    def from_holdings(
        cls,
        holdings: dict[str, float],
        close_df: pd.DataFrame,
        benchmark_col: str = None,
        weights: dict[str, float] = None,
        name: str = "Portfolio",
    ) -> "PortfolioAnalytics":
        """
        Create analytics from holdings and price data.

        Args:
            holdings: Dict mapping symbol -> quantity
            close_df: Price DataFrame (Date x Ticker)
            benchmark_col: Benchmark column name in close_df (optional)
            weights: Custom weights (optional, else equal weight)
            name: Portfolio name

        Returns:
            PortfolioAnalytics instance
        """
        symbols = list(holdings.keys())

        # Filter to available symbols
        available = [s for s in symbols if s in close_df.columns]
        if not available:
            raise ValueError(f"No holdings found in price data: {symbols}")

        # Calculate returns
        returns_df = close_df[available].pct_change().dropna()

        # Calculate weights
        if weights is None:
            # Equal weight by default
            weights = {s: 1.0 / len(available) for s in available}
        else:
            # Normalize provided weights
            total_weight = sum(weights.get(s, 0) for s in available)
            if total_weight > 0:
                weights = {s: weights.get(s, 0) / total_weight for s in available}
            else:
                weights = {s: 1.0 / len(available) for s in available}

        # Portfolio returns (weighted)
        weight_series = pd.Series(weights)
        portfolio_returns = (returns_df[available] * weight_series).sum(axis=1)

        # Benchmark returns
        benchmark_returns = None
        if benchmark_col and benchmark_col in close_df.columns:
            benchmark_returns = close_df[benchmark_col].pct_change().dropna()

        return cls(
            returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            name=name,
        )

    @classmethod
    def from_equity_curve(
        cls,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series = None,
        name: str = "Portfolio",
    ) -> "PortfolioAnalytics":
        """
        Create analytics from equity curve (cumulative values).

        Args:
            equity_curve: Cumulative portfolio value series
            benchmark_curve: Cumulative benchmark value series (optional)
            name: Portfolio name

        Returns:
            PortfolioAnalytics instance
        """
        returns = equity_curve.pct_change().dropna()
        benchmark_returns = benchmark_curve.pct_change().dropna() if benchmark_curve is not None else None

        return cls(
            returns=returns,
            benchmark_returns=benchmark_returns,
            name=name,
        )

    # -------------------------------------------------------------------------
    # Metric Properties
    # -------------------------------------------------------------------------

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio."""
        if "sharpe" not in self._metrics_cache:
            self._metrics_cache["sharpe"] = calculate_sharpe_ratio(
                self.returns, self.risk_free_rate
            )
        return self._metrics_cache["sharpe"]

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino Ratio."""
        if "sortino" not in self._metrics_cache:
            self._metrics_cache["sortino"] = calculate_sortino_ratio(
                self.returns, self.risk_free_rate
            )
        return self._metrics_cache["sortino"]

    @property
    def max_drawdown(self) -> float:
        """Maximum Drawdown."""
        if "max_dd" not in self._metrics_cache:
            dd, _, _ = calculate_max_drawdown(returns=self.returns)
            self._metrics_cache["max_dd"] = dd
        return self._metrics_cache["max_dd"]

    @property
    def calmar_ratio(self) -> float:
        """Calmar Ratio (return / max drawdown)."""
        if "calmar" not in self._metrics_cache:
            self._metrics_cache["calmar"] = calculate_calmar_ratio(self.returns)
        return self._metrics_cache["calmar"]

    @property
    def beta(self) -> Optional[float]:
        """Beta vs benchmark."""
        if self.benchmark_returns is None:
            return None
        if "beta" not in self._metrics_cache:
            self._metrics_cache["beta"] = calculate_beta(
                self.returns, self.benchmark_returns
            )
        return self._metrics_cache["beta"]

    @property
    def alpha(self) -> Optional[float]:
        """Jensen's Alpha (annualized)."""
        if self.benchmark_returns is None:
            return None
        if "alpha" not in self._metrics_cache:
            self._metrics_cache["alpha"] = calculate_alpha(
                self.returns, self.benchmark_returns, self.risk_free_rate
            )
        return self._metrics_cache["alpha"]

    @property
    def information_ratio(self) -> Optional[float]:
        """Information Ratio vs benchmark."""
        if self.benchmark_returns is None:
            return None
        if "ir" not in self._metrics_cache:
            self._metrics_cache["ir"] = calculate_information_ratio(
                self.returns, self.benchmark_returns
            )
        return self._metrics_cache["ir"]

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        if "volatility" not in self._metrics_cache:
            self._metrics_cache["volatility"] = float(
                self.returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
        return self._metrics_cache["volatility"]

    @property
    def downside_volatility(self) -> float:
        """Annualized downside volatility."""
        if "downside_vol" not in self._metrics_cache:
            negative_returns = self.returns[self.returns < 0]
            if len(negative_returns) > 1:
                self._metrics_cache["downside_vol"] = float(
                    negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                )
            else:
                self._metrics_cache["downside_vol"] = 0.0
        return self._metrics_cache["downside_vol"]

    @property
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        if "cagr" not in self._metrics_cache:
            if len(self.returns) < 2:
                self._metrics_cache["cagr"] = 0.0
            else:
                total_return = (1 + self.returns).prod() - 1
                years = len(self.returns) / TRADING_DAYS_PER_YEAR
                if years > 0 and total_return > -1:
                    self._metrics_cache["cagr"] = float(
                        (1 + total_return) ** (1 / years) - 1
                    )
                else:
                    self._metrics_cache["cagr"] = 0.0
        return self._metrics_cache["cagr"]

    @property
    def total_return(self) -> float:
        """Total cumulative return."""
        if "total_return" not in self._metrics_cache:
            self._metrics_cache["total_return"] = float(
                (1 + self.returns).prod() - 1
            )
        return self._metrics_cache["total_return"]

    @property
    def var_95(self) -> float:
        """95% Value at Risk (daily)."""
        if "var_95" not in self._metrics_cache:
            self._metrics_cache["var_95"] = calculate_var(self.returns, 0.95)
        return self._metrics_cache["var_95"]

    @property
    def cvar_95(self) -> float:
        """95% Conditional VaR (Expected Shortfall)."""
        if "cvar_95" not in self._metrics_cache:
            self._metrics_cache["cvar_95"] = calculate_cvar(self.returns, 0.95)
        return self._metrics_cache["cvar_95"]

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    def get_all_metrics(self) -> dict:
        """
        Get all portfolio metrics as a dictionary.

        Returns:
            Dict with all computed metrics
        """
        metrics = {
            "name": self.name,
            "start_date": self.returns.index.min().isoformat() if len(self.returns) > 0 else None,
            "end_date": self.returns.index.max().isoformat() if len(self.returns) > 0 else None,
            "trading_days": len(self.returns),
            "cagr": self.cagr,
            "total_return": self.total_return,
            "volatility": self.volatility,
            "downside_volatility": self.downside_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
        }

        # Add benchmark-relative metrics
        if self.benchmark_returns is not None:
            metrics["beta"] = self.beta
            metrics["alpha"] = self.alpha
            metrics["information_ratio"] = self.information_ratio

        return metrics

    def get_rolling_metrics(self, window: int = 63) -> pd.DataFrame:
        """
        Get rolling performance metrics.

        Args:
            window: Rolling window size (default 63 = ~3 months)

        Returns:
            DataFrame with rolling metrics
        """
        return calculate_rolling_metrics(
            self.returns,
            window=window,
            benchmark_returns=self.benchmark_returns,
        )

    def get_drawdown_series(self) -> pd.Series:
        """Get the drawdown time series."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max

    def get_equity_curve(self, initial_value: float = 100.0) -> pd.Series:
        """
        Get the equity curve (cumulative value).

        Args:
            initial_value: Starting portfolio value

        Returns:
            Equity curve series
        """
        return initial_value * (1 + self.returns).cumprod()

    def compare_to_benchmark(self) -> dict:
        """
        Compare portfolio performance to benchmark.

        Returns:
            Dict with comparison metrics
        """
        if self.benchmark_returns is None:
            return {"error": "No benchmark provided"}

        # Align returns
        aligned = pd.concat(
            [self.returns, self.benchmark_returns], axis=1, keys=["portfolio", "benchmark"]
        ).dropna()

        port_ret = aligned["portfolio"]
        bench_ret = aligned["benchmark"]

        # Metrics for both
        port_cagr = (1 + port_ret.mean()) ** TRADING_DAYS_PER_YEAR - 1
        bench_cagr = (1 + bench_ret.mean()) ** TRADING_DAYS_PER_YEAR - 1

        port_vol = port_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        bench_vol = bench_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        port_dd, _, _ = calculate_max_drawdown(returns=port_ret)
        bench_dd, _, _ = calculate_max_drawdown(returns=bench_ret)

        return {
            "portfolio": {
                "cagr": port_cagr,
                "volatility": port_vol,
                "max_drawdown": port_dd,
                "sharpe": calculate_sharpe_ratio(port_ret, self.risk_free_rate),
            },
            "benchmark": {
                "cagr": bench_cagr,
                "volatility": bench_vol,
                "max_drawdown": bench_dd,
                "sharpe": calculate_sharpe_ratio(bench_ret, self.risk_free_rate),
            },
            "relative": {
                "excess_return": port_cagr - bench_cagr,
                "beta": self.beta,
                "alpha": self.alpha,
                "information_ratio": self.information_ratio,
                "correlation": port_ret.corr(bench_ret),
            },
        }

    def summary(self) -> str:
        """Get a formatted summary string."""
        metrics = self.get_all_metrics()

        lines = [
            f"Portfolio: {self.name}",
            f"Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['trading_days']} days)",
            "",
            "Performance:",
            f"  CAGR:           {metrics['cagr']*100:>8.2f}%",
            f"  Total Return:   {metrics['total_return']*100:>8.2f}%",
            "",
            "Risk:",
            f"  Volatility:     {metrics['volatility']*100:>8.2f}%",
            f"  Max Drawdown:   {metrics['max_drawdown']*100:>8.2f}%",
            f"  VaR (95%):      {metrics['var_95']*100:>8.2f}%",
            "",
            "Risk-Adjusted:",
            f"  Sharpe Ratio:   {metrics['sharpe_ratio']:>8.2f}",
            f"  Sortino Ratio:  {metrics['sortino_ratio']:>8.2f}",
            f"  Calmar Ratio:   {metrics['calmar_ratio']:>8.2f}",
        ]

        if "beta" in metrics:
            lines.extend([
                "",
                "Benchmark Relative:",
                f"  Beta:           {metrics['beta']:>8.2f}",
                f"  Alpha:          {metrics['alpha']*100:>8.2f}%",
                f"  Info Ratio:     {metrics['information_ratio']:>8.2f}",
            ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PortfolioAnalytics(name='{self.name}', days={len(self.returns)}, sharpe={self.sharpe_ratio:.2f})"
