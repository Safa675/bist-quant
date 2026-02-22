from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
def compute_yearly_metrics(returns, benchmark_returns=None, xautry_returns=None):
    df = pd.DataFrame({"ret": returns})
    if benchmark_returns is not None:
        df["bench"] = benchmark_returns
    if xautry_returns is not None:
        df["xautry"] = xautry_returns

    df = df.dropna(subset=["ret"])
    df["year"] = df.index.year

    yearly_rows = []
    for year, group in df.groupby("year"):
        if group.empty:
            continue
        daily_ret = group["ret"]
        ann_return = (1.0 + daily_ret).prod() - 1.0

        mean_ret = daily_ret.mean()
        std_ret = daily_ret.std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret and std_ret > 0 else 0.0

        downside = daily_ret[daily_ret < 0]
        downside_std = downside.std()
        sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std and downside_std > 0 else 0.0

        bench_return = np.nan
        if "bench" in group.columns:
            common = group.dropna(subset=["bench"])
            if not common.empty:
                bench_return = (1.0 + common["bench"]).prod() - 1.0

        xautry_return = np.nan
        if "xautry" in group.columns:
            common = group.dropna(subset=["xautry"])
            if not common.empty:
                xautry_return = (1.0 + common["xautry"]).prod() - 1.0

        yearly_rows.append(
            {
                "Year": int(year),
                "Return": ann_return,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "XU100_Return": bench_return,
                "Excess_vs_XU100": ann_return - bench_return if pd.notna(bench_return) else np.nan,
                "XAUTRY_Return": xautry_return,
                "Excess_vs_XAUTRY": ann_return - xautry_return if pd.notna(xautry_return) else np.nan,
            }
        )

    return pd.DataFrame(yearly_rows).sort_values("Year")


def compute_capm_metrics(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_daily: float = 0.0,
) -> dict:
    df = pd.DataFrame({"strategy": strategy_returns, "market": market_returns}).dropna()

    n_obs = len(df)
    if n_obs < 30:
        return {
            "n_obs": n_obs,
            "alpha_daily": np.nan,
            "alpha_annual": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "residual_vol_annual": np.nan,
        }

    y = (df["strategy"] - risk_free_daily).values.astype(float)
    x = (df["market"] - risk_free_daily).values.astype(float)

    x_var = np.var(x, ddof=1)
    if not np.isfinite(x_var) or x_var <= 0:
        return {
            "n_obs": n_obs,
            "alpha_daily": np.nan,
            "alpha_annual": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "residual_vol_annual": np.nan,
        }

    X = np.column_stack([np.ones_like(x), x])
    alpha_daily, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = alpha_daily + beta * x
    resid = y - y_hat

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    correlation = np.corrcoef(y, x)[0, 1] if n_obs > 1 else np.nan

    alpha_annual = (1.0 + alpha_daily) ** 252 - 1.0 if np.isfinite(alpha_daily) else np.nan
    residual_vol_annual = np.std(resid, ddof=1) * np.sqrt(252) if n_obs > 1 else np.nan

    return {
        "n_obs": int(n_obs),
        "alpha_daily": float(alpha_daily),
        "alpha_annual": float(alpha_annual),
        "beta": float(beta),
        "r_squared": float(r_squared) if np.isfinite(r_squared) else np.nan,
        "correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        "residual_vol_annual": float(residual_vol_annual) if np.isfinite(residual_vol_annual) else np.nan,
    }


def compute_rolling_beta_series(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    min_periods: int = 126,
    risk_free_daily: float = 0.0,
) -> pd.Series:
    df = pd.DataFrame({"strategy": strategy_returns, "market": market_returns}).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    x = df["market"] - risk_free_daily
    y = df["strategy"] - risk_free_daily
    cov_xy = y.rolling(window=window, min_periods=min_periods).cov(x)
    var_x = x.rolling(window=window, min_periods=min_periods).var()
    beta = cov_xy / var_x.replace(0.0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    beta.name = "Rolling_Beta"
    return beta


def compute_yearly_rolling_beta_metrics(rolling_beta: pd.Series) -> pd.DataFrame:
    if rolling_beta is None or rolling_beta.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Observations",
                "Beta_Start",
                "Beta_End",
                "Beta_Mean",
                "Beta_Median",
                "Beta_Min",
                "Beta_Max",
                "Beta_Std",
                "Beta_Change",
            ]
        )

    series = rolling_beta.dropna()
    if series.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Observations",
                "Beta_Start",
                "Beta_End",
                "Beta_Mean",
                "Beta_Median",
                "Beta_Min",
                "Beta_Max",
                "Beta_Std",
                "Beta_Change",
            ]
        )

    series = series.sort_index()
    rows = []
    for year, group in series.groupby(series.index.year):
        if group.empty:
            continue
        beta_start = group.iloc[0]
        beta_end = group.iloc[-1]
        rows.append(
            {
                "Year": int(year),
                "Observations": int(len(group)),
                "Beta_Start": float(beta_start),
                "Beta_End": float(beta_end),
                "Beta_Mean": float(group.mean()),
                "Beta_Median": float(group.median()),
                "Beta_Min": float(group.min()),
                "Beta_Max": float(group.max()),
                "Beta_Std": float(group.std(ddof=1)) if len(group) > 1 else np.nan,
                "Beta_Change": float(beta_end - beta_start),
            }
        )

    return pd.DataFrame(rows).sort_values("Year")


@dataclass(frozen=True)
class BenchmarkContext:
    """Benchmark series and labels used by reporting outputs."""

    benchmark_name: str
    benchmark_returns: pd.Series | None
    xu100_returns: pd.Series | None
    xautry_returns: pd.Series
    yearly_bench_col: str
    yearly_excess_col: str
    yearly_table_label: str


@dataclass
class ReportMetricsBundle:
    """Precomputed metrics reused across file outputs and summary text."""

    yearly_metrics: pd.DataFrame
    capm_metrics: dict
    rolling_beta: pd.Series
    yearly_rolling_beta: pd.DataFrame


@dataclass(frozen=True)
class CsvWriteTask:
    """Write contract for one DataFrame-to-CSV output."""

    frame: pd.DataFrame
    path: Path
    index: bool = True


class BenchmarkResolver:
    """Builds aligned benchmark return series for reporting."""

    def resolve(
        self,
        returns: pd.Series,
        xu100_prices: pd.Series | None,
        xautry_returns: pd.Series,
    ) -> BenchmarkContext:
        xu100_returns = None
        if xu100_prices is not None:
            xu100_returns = xu100_prices.shift(-1) / xu100_prices - 1.0
            xu100_returns = xu100_returns.reindex(returns.index)

        benchmark_name = "XU100"
        benchmark_returns = xu100_returns
        yearly_bench_col = "XU100_Return"
        yearly_excess_col = "Excess_vs_XU100"
        yearly_table_label = "XU100"

        if benchmark_returns is not None:
            aligned_bench = (
                pd.concat([returns, benchmark_returns], axis=1, keys=["strategy", "benchmark"])
                .dropna()
            )
            if aligned_bench.empty:
                raise ValueError("XU100 benchmark has no overlap with strategy returns")
            bench_coverage = len(aligned_bench) / max(len(returns), 1)
            logger.info(
                f"   Benchmark alignment (XU100): {len(aligned_bench)}/{len(returns)} days ({bench_coverage:.1%})"
            )

        return BenchmarkContext(
            benchmark_name=benchmark_name,
            benchmark_returns=benchmark_returns,
            xu100_returns=xu100_returns,
            xautry_returns=xautry_returns,
            yearly_bench_col=yearly_bench_col,
            yearly_excess_col=yearly_excess_col,
            yearly_table_label=yearly_table_label,
        )


class MetricsCalculator:
    """Computes reusable report metrics from strategy and benchmark returns."""

    def compute(self, returns: pd.Series, benchmark_context: BenchmarkContext) -> ReportMetricsBundle:
        yearly_metrics = compute_yearly_metrics(
            returns,
            benchmark_context.xu100_returns,
            benchmark_context.xautry_returns,
        )

        if benchmark_context.benchmark_returns is not None:
            capm_metrics = compute_capm_metrics(returns, benchmark_context.benchmark_returns)
            rolling_beta = compute_rolling_beta_series(
                strategy_returns=returns,
                market_returns=benchmark_context.benchmark_returns,
                window=252,
                min_periods=126,
                risk_free_daily=0.0,
            )
        else:
            capm_metrics = {
                "n_obs": 0,
                "alpha_daily": np.nan,
                "alpha_annual": np.nan,
                "beta": np.nan,
                "r_squared": np.nan,
                "correlation": np.nan,
                "residual_vol_annual": np.nan,
            }
            rolling_beta = pd.Series(dtype=float)

        capm_metrics = dict(capm_metrics)
        capm_metrics["benchmark"] = benchmark_context.benchmark_name
        yearly_rolling_beta = compute_yearly_rolling_beta_metrics(rolling_beta)
        return ReportMetricsBundle(
            yearly_metrics=yearly_metrics,
            capm_metrics=capm_metrics,
            rolling_beta=rolling_beta,
            yearly_rolling_beta=yearly_rolling_beta,
        )


class ReportWriter:
    """Persists output artifacts while separating pure computation from I/O."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_core_reports(
        self,
        results: dict,
        returns: pd.Series,
        benchmark_context: BenchmarkContext,
        metrics: ReportMetricsBundle,
    ) -> None:
        returns_df = pd.DataFrame(
            {
                "Return": returns,
                "XAU_TRY_Return": benchmark_context.xautry_returns.squeeze(),
            }
        )
        if benchmark_context.xu100_returns is not None:
            returns_df["XU100_Return"] = benchmark_context.xu100_returns.squeeze()

        rolling_beta_df = metrics.rolling_beta.to_frame().reset_index()
        rolling_beta_df.columns = ["date", "Rolling_Beta"]

        tasks = [
            CsvWriteTask(
                frame=pd.DataFrame({"Equity": results["equity"]}),
                path=self.output_dir / "equity_curve.csv",
            ),
            CsvWriteTask(
                frame=returns_df,
                path=self.output_dir / "returns.csv",
            ),
            CsvWriteTask(
                frame=metrics.yearly_metrics,
                path=self.output_dir / "yearly_metrics.csv",
                index=False,
            ),
            CsvWriteTask(
                frame=pd.DataFrame([metrics.capm_metrics]),
                path=self.output_dir / "capm_metrics.csv",
                index=False,
            ),
            CsvWriteTask(
                frame=rolling_beta_df,
                path=self.output_dir / "rolling_beta.csv",
                index=False,
            ),
            CsvWriteTask(
                frame=metrics.yearly_rolling_beta,
                path=self.output_dir / "yearly_rolling_beta.csv",
                index=False,
            ),
            CsvWriteTask(
                frame=pd.DataFrame(results["regime_performance"]).T,
                path=self.output_dir / "regime_performance.csv",
            ),
        ]
        self._write_csv_tasks(tasks)

    def write_optional_reports(self, results: dict) -> None:
        tasks: list[CsvWriteTask] = []

        if "yearly_axis_winners" in results:
            yearly_axis = results["yearly_axis_winners"]
            if isinstance(yearly_axis, pd.DataFrame) and not yearly_axis.empty:
                tasks.append(
                    CsvWriteTask(
                        frame=yearly_axis,
                        path=self.output_dir / "yearly_axis_winners.csv",
                        index=False,
                    )
                )
                self._write_yearly_axis_text(yearly_axis)

        if results.get("holdings_history"):
            holdings_df = pd.DataFrame(results["holdings_history"])
            
            # Extract fundamental quarter and append to holdings_df
            try:
                from bist_quant.common.data_loader import DataLoader
                loader = DataLoader()
                fundamentals_df = loader.load_fundamentals_parquet()
                if fundamentals_df is not None and not fundamentals_df.empty:
                    date_cols = [c for c in fundamentals_df.columns if "/" in str(c)]
                    if date_cols:
                        df_dates = fundamentals_df[date_cols].copy()
                        ticker_notna = df_dates.notna().groupby(level="ticker").any()
                        latest_quarters = {}
                        for ticker, row in ticker_notna.iterrows():
                            valid_cols = row[row].index.tolist()
                            if valid_cols:
                                latest_quarters[ticker] = max(valid_cols)
                        holdings_df["fundamental_quarter"] = holdings_df["ticker"].map(latest_quarters)
            except Exception as e:
                logger.warning(f"Failed to append fundamental_quarter: {e}")

            tasks.append(
                CsvWriteTask(
                    frame=holdings_df,
                    path=self.output_dir / "holdings.csv",
                    index=False,
                )
            )
            holdings_pivot = (
                holdings_df.pivot_table(index="date", columns="ticker", values="weight", aggfunc="first")
                .fillna(0)
            )
            tasks.append(
                CsvWriteTask(
                    frame=holdings_pivot,
                    path=self.output_dir / "holdings_matrix.csv",
                )
            )

        returns_detailed = results.get("returns_df")
        if isinstance(returns_detailed, pd.DataFrame) and not returns_detailed.empty:
            tasks.append(
                CsvWriteTask(
                    frame=returns_detailed,
                    path=self.output_dir / "returns_detailed.csv",
                )
            )

        sanity_checks = results.get("sanity_checks")
        if isinstance(sanity_checks, pd.DataFrame) and not sanity_checks.empty:
            tasks.append(
                CsvWriteTask(
                    frame=sanity_checks,
                    path=self.output_dir / "sanity_checks.csv",
                )
            )

        gold_tracking_metrics = results.get("gold_tracking_metrics")
        if isinstance(gold_tracking_metrics, pd.DataFrame) and not gold_tracking_metrics.empty:
            tasks.append(
                CsvWriteTask(
                    frame=gold_tracking_metrics,
                    path=self.output_dir / "gold_tracking_metrics.csv",
                    index=False,
                )
            )

        gold_spread_zscores = results.get("gold_spread_zscores")
        if isinstance(gold_spread_zscores, pd.DataFrame) and not gold_spread_zscores.empty:
            tasks.append(
                CsvWriteTask(
                    frame=gold_spread_zscores,
                    path=self.output_dir / "gold_spread_zscores.csv",
                )
            )

        if tasks:
            self._write_csv_tasks(tasks)

        if "gold_selected_codes" in results:
            selected_codes = results.get("gold_selected_codes") or []
            with open(self.output_dir / "gold_selected_codes.json", "w", encoding="utf-8") as handle:
                json.dump(selected_codes, handle, ensure_ascii=False, indent=2)

    def write_summary_text(
        self,
        results: dict,
        factor_name: str,
        returns: pd.Series,
        benchmark_context: BenchmarkContext,
        metrics: ReportMetricsBundle,
    ) -> None:
        with open(self.output_dir / "summary.txt", "w") as handle:
            handle.write("=" * 60 + "\n")
            handle.write(f"{factor_name.upper()} FACTOR MODEL\n")
            handle.write("=" * 60 + "\n\n")
            handle.write(f"Total Return: {results['total_return'] * 100:.2f}%\n")
            handle.write(f"CAGR: {results['cagr'] * 100:.2f}%\n")
            handle.write(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%\n")
            handle.write(f"Sharpe Ratio: {results['sharpe']:.2f}\n")
            handle.write(f"Sortino Ratio: {results['sortino']:.2f}\n")
            handle.write(f"Win Rate: {results['win_rate'] * 100:.2f}%\n")
            handle.write(f"Trading Days: {len(returns)}\n")
            handle.write(f"Rebalance Days: {results['rebalance_count']}\n")
            handle.write(f"Total Trades: {results['trade_count']}\n")

            if benchmark_context.benchmark_returns is not None:
                bench_aligned = benchmark_context.benchmark_returns.dropna()
                if len(bench_aligned) > 0:
                    bench_total = (1 + bench_aligned).prod() - 1
                    handle.write(
                        f"\nBenchmark ({benchmark_context.benchmark_name}) Return: {bench_total * 100:.2f}%\n"
                    )
                    handle.write(
                        f"Excess vs {benchmark_context.benchmark_name}: "
                        f"{(results['total_return'] - bench_total) * 100:.2f}%\n"
                    )

            handle.write(f"\nCAPM vs {benchmark_context.benchmark_name}\n")
            handle.write(f"Observations: {metrics.capm_metrics['n_obs']}\n")
            handle.write(
                f"Beta: {metrics.capm_metrics['beta']:.4f}\n"
                if pd.notna(metrics.capm_metrics["beta"])
                else "Beta: NaN\n"
            )
            handle.write(
                f"Alpha (annualized): {metrics.capm_metrics['alpha_annual'] * 100:.2f}%\n"
                if pd.notna(metrics.capm_metrics["alpha_annual"])
                else "Alpha (annualized): NaN\n"
            )
            handle.write(
                f"R-squared: {metrics.capm_metrics['r_squared']:.4f}\n"
                if pd.notna(metrics.capm_metrics["r_squared"])
                else "R-squared: NaN\n"
            )

            rb_valid = metrics.rolling_beta.dropna()
            if not rb_valid.empty:
                handle.write(f"Latest Rolling Beta (252d): {rb_valid.iloc[-1]:.4f}\n")

            if benchmark_context.benchmark_name != "XAU/TRY":
                xautry_aligned = benchmark_context.xautry_returns.dropna()
                if len(xautry_aligned) > 0:
                    xautry_total = (1 + xautry_aligned).prod() - 1
                    handle.write(f"\nBenchmark (XAU/TRY) Return: {xautry_total * 100:.2f}%\n")
                    handle.write(
                        f"Excess vs XAU/TRY: {(results['total_return'] - xautry_total) * 100:.2f}%\n"
                    )

    @staticmethod
    def _write_csv_tasks(tasks: list[CsvWriteTask]) -> None:
        if not tasks:
            return
        max_workers = min(max(len(tasks), 1), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(task.frame.to_csv, task.path, index=task.index)
                for task in tasks
            ]
            for future in as_completed(futures):
                future.result()

    def _write_yearly_axis_text(self, yearly_axis: pd.DataFrame) -> None:
        with open(self.output_dir / "yearly_axis_winners.txt", "w") as handle:
            handle.write("=" * 70 + "\n")
            handle.write("FIVE-FACTOR YEARLY AXIS WINNERS\n")
            handle.write("=" * 70 + "\n\n")
            for year in sorted(yearly_axis["Year"].unique()):
                handle.write(f"{int(year)}\n")
                handle.write("-" * 70 + "\n")
                year_rows = yearly_axis[yearly_axis["Year"] == year].sort_values("Axis")
                for _, row in year_rows.iterrows():
                    handle.write(
                        f"{row['Axis']:<14} Winner: {row['Winner']:<12} | "
                        f"{row['High_Side']}: {row['High_Side_Return']:+.2%} | "
                        f"{row['Low_Side']}: {row['Low_Side_Return']:+.2%} | "
                        f"Spread: {row['Spread_Winner_Minus_Loser']:+.2%}\n"
                    )
                handle.write("\n")


class ReportGenerator:
    """Persists results and prints standardized summaries."""

    def __init__(self, models_dir: Path, data_dir: Path, loader) -> None:
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.loader = loader

    def save_results(
        self,
        results: dict,
        factor_name: str,
        xu100_prices: pd.Series | None,
        xautry_prices: pd.Series | None,
        factor_capm_store: dict,
        factor_yearly_rolling_beta_store: dict,
        output_dir: Path | None = None,
    ) -> None:
        from bist_quant.settings import get_output_dir

        if output_dir is None:
            # Use config utility - defaults to CWD or BIST_QUANT_OUTPUT_DIR env var
            output_path = get_output_dir("signals", factor_name)
        else:
            output_path = Path(output_dir)
        returns = results["returns"]
        xautry_returns = results["xautry_returns"]
        if isinstance(xautry_returns, pd.DataFrame):
            xautry_returns = xautry_returns.squeeze()
        xautry_returns = pd.Series(xautry_returns, index=returns.index)
        _ = xautry_prices  # Backward-compatible signature; strategy uses xautry_returns from backtest payload.

        benchmark_context = BenchmarkResolver().resolve(
            returns=returns,
            xu100_prices=xu100_prices,
            xautry_returns=xautry_returns,
        )
        metrics = MetricsCalculator().compute(
            returns=returns,
            benchmark_context=benchmark_context,
        )

        writer = ReportWriter(output_dir=output_path)
        writer.write_core_reports(
            results=results,
            returns=returns,
            benchmark_context=benchmark_context,
            metrics=metrics,
        )
        writer.write_optional_reports(results=results)
        writer.write_summary_text(
            results=results,
            factor_name=factor_name,
            returns=returns,
            benchmark_context=benchmark_context,
            metrics=metrics,
        )

        factor_capm_store[factor_name] = dict(metrics.capm_metrics)
        factor_yearly_rolling_beta_store[factor_name] = metrics.yearly_rolling_beta.copy()

        logger.info(f"\n💾 Results saved to: {output_path}")
        if pd.notna(metrics.capm_metrics.get("beta", np.nan)):
            logger.info(
                f"   CAPM (vs {benchmark_context.benchmark_name}) -> "
                f"Beta: {metrics.capm_metrics['beta']:.3f}, "
                f"Alpha: {metrics.capm_metrics['alpha_annual']*100:.2f}%, "
                f"R²: {metrics.capm_metrics['r_squared']:.3f}"
            )

        logger.info("\n" + "=" * 70)
        logger.info("YEARLY RESULTS")
        logger.info("=" * 70)
        logger.info(
            f"{'Year':<6} {'Model':>10} "
            f"{benchmark_context.yearly_table_label:>10} {'Excess':>10}"
        )
        logger.info("-" * 40)
        for _, row in metrics.yearly_metrics.iterrows():
            bench_ret = row[benchmark_context.yearly_bench_col]
            bench_ret = bench_ret if pd.notna(bench_ret) else 0
            excess = row[benchmark_context.yearly_excess_col]
            excess = excess if pd.notna(excess) else row["Return"]
            logger.info(
                f"{int(row['Year']):<6} {row['Return']*100:>9.1f}% "
                f"{bench_ret*100:>9.1f}% {excess*100:>9.1f}%"
            )

    def save_correlation_matrix(
        self,
        factor_returns: dict[str, pd.Series],
        xautry_prices: pd.Series | None,
        output_dir: Path | None = None,
    ):
        from bist_quant.settings import get_output_dir

        if not factor_returns:
            logger.warning("⚠️  No factor returns stored - run factors first")
            return

        if output_dir is None:
            # Use config utility - defaults to CWD or BIST_QUANT_OUTPUT_DIR env var
            output_dir = get_output_dir()
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_returns = {factor_name: returns for factor_name, returns in factor_returns.items()}
        xu100_file = self.data_dir / "xu100_prices.csv"
        if xu100_file.exists():
            xu100_df = pd.read_csv(xu100_file)
            xu100_df["Date"] = pd.to_datetime(xu100_df["Date"])
            xu100_df = xu100_df.set_index("Date").sort_index()
            if "Close" in xu100_df.columns:
                all_returns["XU100"] = xu100_df["Close"].pct_change().dropna()

        if xautry_prices is not None:
            xautry_returns = self.loader.load_xautry_prices(self.data_dir / "xau_try_2013_2026.csv").pct_change().dropna()
            all_returns["XAUTRY"] = xautry_returns

        returns_df = pd.DataFrame(all_returns).dropna(how="all")
        corr_matrix = returns_df.corr()

        labels = list(corr_matrix.columns)
        col_width = max(max(len(label) for label in labels), 6) + 2
        logger.info("\n" + "=" * 70)
        logger.info("RETURN CORRELATION MATRIX")
        logger.info("=" * 70)
        header = " " * col_width + "".join(f"{label:>{col_width}}" for label in labels)
        logger.info(header)
        logger.info("-" * len(header))
        for row_label in labels:
            row_str = f"{row_label:<{col_width}}"
            for col_label in labels:
                row_str += f"{corr_matrix.loc[row_label, col_label]:>{col_width}.4f}"
            logger.info(row_str)

        full_corr_file = output_dir / "factor_correlation_matrix.csv"
        corr_matrix.to_csv(full_corr_file)
        logger.info(f"\n💾 Correlation matrix saved to: {full_corr_file}")
        return corr_matrix

    @staticmethod
    def save_capm_summary(factor_capm: dict, output_dir: Path | None = None, models_dir: Path | None = None):
        from bist_quant.settings import get_output_dir

        if not factor_capm:
            logger.warning("⚠️  No CAPM results available")
            return

        if output_dir is None:
            # Use config utility - defaults to CWD or BIST_QUANT_OUTPUT_DIR env var
            output_dir = get_output_dir("signals", "capm_summary")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for factor_name, metrics in factor_capm.items():
            rows.append(
                {
                    "Factor": factor_name,
                    "Benchmark": metrics.get("benchmark", "XU100"),
                    "Observations": metrics.get("n_obs"),
                    "Beta": metrics.get("beta"),
                    "Alpha_Annual": metrics.get("alpha_annual"),
                    "R_squared": metrics.get("r_squared"),
                    "Correlation": metrics.get("correlation"),
                    "ResidualVol_Annual": metrics.get("residual_vol_annual"),
                }
            )

        df = pd.DataFrame(rows).sort_values("Factor")
        out_file = output_dir / "capm_summary.csv"
        df.to_csv(out_file, index=False)

        logger.info("\n" + "=" * 70)
        logger.info("CAPM SUMMARY (factor-specific benchmark)")
        logger.info("=" * 70)
        for _, row in df.iterrows():
            beta = row["Beta"]
            alpha = row["Alpha_Annual"]
            r_squared = row["R_squared"]
            benchmark = row["Benchmark"]
            logger.info(
                f"{row['Factor']:<24} vs {benchmark:<7} beta={beta:>6.3f}  alpha={alpha*100:>7.2f}%  R²={r_squared:>6.3f}"
                if pd.notna(beta) and pd.notna(alpha) and pd.notna(r_squared)
                else f"{row['Factor']:<24} vs {benchmark:<7} beta=  NaN  alpha=   NaN  R²=  NaN"
            )
        logger.info(f"\n💾 CAPM summary saved to: {out_file}")

    @staticmethod
    def save_yearly_rolling_beta_summary(
        factor_yearly_rolling_beta: dict,
        output_dir: Path | None = None,
        models_dir: Path | None = None,
    ) -> None:
        from bist_quant.settings import get_output_dir

        if not factor_yearly_rolling_beta:
            logger.warning("⚠️  No yearly rolling-beta results available")
            return

        if output_dir is None:
            # Use config utility - defaults to CWD or BIST_QUANT_OUTPUT_DIR env var
            output_dir = get_output_dir("signals", "rolling_beta_summary")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for factor_name, yearly_df in factor_yearly_rolling_beta.items():
            if yearly_df is None or yearly_df.empty:
                continue
            data = yearly_df.copy()
            data.insert(0, "Factor", factor_name)
            rows.append(data)

        if rows:
            summary_df = pd.concat(rows, ignore_index=True).sort_values(["Factor", "Year"]).reset_index(drop=True)
        else:
            summary_df = pd.DataFrame(
                columns=[
                    "Factor",
                    "Year",
                    "Observations",
                    "Beta_Start",
                    "Beta_End",
                    "Beta_Mean",
                    "Beta_Median",
                    "Beta_Min",
                    "Beta_Max",
                    "Beta_Std",
                    "Beta_Change",
                ]
            )

        out_file = output_dir / "yearly_rolling_beta_summary.csv"
        summary_df.to_csv(out_file, index=False)
        logger.info(f"💾 Yearly rolling beta summary saved to: {out_file}")
