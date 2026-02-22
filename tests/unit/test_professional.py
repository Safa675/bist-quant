"""Unit tests for bist_quant.analytics.professional."""

from __future__ import annotations
import math, pytest
from bist_quant.analytics.core_metrics import SeriesPoint
from bist_quant.analytics.professional import (
    CryptoTradeInput, OptionGreeksInput, FundCandidate, StressFactorShock,
    CounterpartyExposure, LiquidityPosition, ConcentrationPosition,
    StrategyCondition, StrategyBlock, StrategyDefinition,
    StrategyRiskLimit, StrategyRiskState, SentimentItem, AttributionInput,
    TaxTrade, OrderBookLevel, TransactionRecord, ComplianceRule, AlertCondition,
    NotificationAlert, EarningsSurprise, InsiderActivity, ShortInterest,
    build_crypto_trade_plan, compute_forex_pip_value, compute_option_greeks,
    compute_futures_margin, screen_funds, compute_commodity_spread,
    evaluate_strategy_risk, run_portfolio_stress_test, assess_counterparty_risk,
    monitor_liquidity_risk, detect_concentration_risk, evaluate_strategy_definition,
    diff_strategy_versions, build_strategy_health, build_factor_exposure_model,
    optimize_constrained_portfolio, OptimizationConstraints,
    analyze_sentiment, compute_risk_adjusted_ratios, build_monthly_quarterly_returns,
    compare_benchmarks, compute_attribution, decompose_portfolio_risk_factors,
    build_tax_report, create_iceberg_slices, build_twap_schedule, build_bracket_order,
    simulate_execution_with_slippage, run_compliance_rule_engine,
    evaluate_alert_conditions, group_alerts, build_escalation_plan,
    build_market_intelligence_snapshot, SentimentSummary,
    run_performance_snapshot,
)

def _pts(n=100):
    vals = [0.01,-0.005,0.003,-0.002,0.008,-0.001,0.004,-0.003,0.006,-0.004]*(n//10+1)
    out, y, m, d = [], 2022, 1, 1
    for v in vals[:n]:
        out.append(SeriesPoint(f"{y:04d}-{m:02d}-{d:02d}", v))
        d += 1
        if d > 28: d = 1; m += 1
        if m > 12: m = 1; y += 1
    return out

def _equity(n=100):
    eq, y, m, d = 1.0, 2022, 1, 1
    out = []
    for i in range(n):
        eq *= 1 + 0.001 + (i % 5 - 2) * 0.002
        out.append(SeriesPoint(f"{y:04d}-{m:02d}-{d:02d}", eq))
        d += 1
        if d > 28: d = 1; m += 1
        if m > 12: m = 1; y += 1
    return out

class TestCryptoTrade:
    def test_basic(self):
        r = build_crypto_trade_plan(CryptoTradeInput("btcusd","long",40000,100000,2,10,5))
        assert r.pair == "BTCUSD"
        assert r.notional > 0
        assert r.margin_required > 0
        assert r.quantity > 0

class TestForex:
    def test_jpy(self):
        r = compute_forex_pip_value("USDJPY", 100000)
        assert r.pip_size == 0.01
    def test_eur(self):
        r = compute_forex_pip_value("EURUSD", 100000)
        assert r.pip_size == 0.0001

class TestOptionGreeks:
    def test_call(self):
        r = compute_option_greeks(OptionGreeksInput("call", 100, 100, 0.25, 0.2, 0.05))
        assert 0 < r.delta < 1
        assert r.gamma > 0
        assert r.theoretical_price > 0
    def test_put(self):
        r = compute_option_greeks(OptionGreeksInput("put", 100, 100, 0.25, 0.2, 0.05))
        assert -1 < r.delta < 0
        assert r.theoretical_price > 0

class TestFuturesMargin:
    def test_basic(self):
        r = compute_futures_margin("ES", 2, 4500, 50, 0.1, 0.05, 0.25, 12.5)
        assert r.notional == 450000
        assert r.initial_margin > 0

class TestFundScreening:
    def test_empty(self):
        assert screen_funds([]) == []
    def test_basic(self):
        funds = [FundCandidate("A",0.1,0.5,500,10,15,1.5), FundCandidate("B",0.5,1.0,100,30,5,0.5)]
        r = screen_funds(funds)
        assert len(r) == 2
        assert r[0].score >= r[1].score

class TestCommoditySpread:
    def test_basic(self):
        r = compute_commodity_spread(100, 95, 1000, 2)
        assert r.spread_value == 5.0

class TestStrategyRisk:
    def test_breach(self):
        lim = [StrategyRiskLimit("s1", 1e6, -20, -5, 3)]
        st = [StrategyRiskState("s1", 2e6, -25, -6, 4)]
        r = evaluate_strategy_risk(lim, st)
        assert len(r) >= 1

class TestStressTest:
    def test_basic(self):
        r = run_portfolio_stress_test(1e6, [StressFactorShock("equity",-10,1.2)])
        assert r.scenario_loss_pct < 0

class TestCounterparty:
    def test_basic(self):
        r = assess_counterparty_risk([CounterpartyExposure("CP1",1e6,5e5,75,200)])
        assert len(r) == 1
        assert r[0].rating in ("low","moderate","high")

class TestLiquidity:
    def test_basic(self):
        r = monitor_liquidity_risk([LiquidityPosition("X",1e6,5e5)])
        assert len(r) == 1

class TestConcentration:
    def test_single_name(self):
        pos = [ConcentrationPosition("A","equity","tech","US",15)]
        r = detect_concentration_risk(pos, single_name_limit_pct=12)
        assert any(a.dimension == "single_name" for a in r)

class TestStrategyBuilder:
    def test_evaluate(self):
        s = StrategyDefinition("s1","test",1,[StrategyBlock("b1","AND",[StrategyCondition("c1","rsi",">",30)])],[],"")
        assert evaluate_strategy_definition(s, {"rsi": 50})
        assert not evaluate_strategy_definition(s, {"rsi": 20})
    def test_diff(self):
        s1 = StrategyDefinition("s1","t",1,[StrategyBlock("b1","AND",[StrategyCondition("c1","rsi",">",30)])],[],"")
        s2 = StrategyDefinition("s1","t",2,[StrategyBlock("b1","AND",[StrategyCondition("c1","rsi",">",50),StrategyCondition("c2","vol","<",20)])],[],"")
        d = diff_strategy_versions(s1, s2)
        assert d.added_conditions == 1
        assert d.changed_conditions == 1

class TestStrategyHealth:
    def test_basic(self):
        r = build_strategy_health(_equity(100))
        assert r.alert in ("ok","warning","critical")

class TestFactorExposure:
    def test_basic(self):
        strat = _pts(100)
        factor = [SeriesPoint(s.date, s.value*0.5+0.001) for s in strat]
        r = build_factor_exposure_model(strat, {"momentum": factor})
        assert "momentum" in r.factor_betas

class TestConstrainedOptimization:
    def test_basic(self):
        a, b = _pts(80), [SeriesPoint(s.date, s.value*0.7+0.001) for s in _pts(80)]
        r = optimize_constrained_portfolio({"A": a, "B": b}, OptimizationConstraints(5, 60, iterations=300))
        assert math.isclose(sum(r.weights.values()), 1.0, abs_tol=0.05)

class TestSentiment:
    def test_empty(self):
        r = analyze_sentiment([])
        assert r.overall_score == 0
    def test_basic(self):
        items = [SentimentItem("AAPL","news","Strong growth beat expectations",100),
                 SentimentItem("TSLA","social","Bearish decline weak",50)]
        r = analyze_sentiment(items)
        assert r.by_symbol["AAPL"] > 0
        assert r.by_symbol["TSLA"] < 0

class TestRiskAdjusted:
    def test_basic(self):
        r = compute_risk_adjusted_ratios(_pts(200))
        assert isinstance(r["sharpe"], float)

class TestMonthlyReturns:
    def test_basic(self):
        r = build_monthly_quarterly_returns(_pts(100))
        assert len(r["monthly"]) > 0
        assert len(r["quarterly"]) > 0

class TestBenchmarks:
    def test_basic(self):
        strat = _pts(100)
        bench = [SeriesPoint(s.date, s.value*0.8) for s in strat]
        r = compare_benchmarks(strat, {"SPY": bench})
        assert len(r) == 1

class TestAttribution:
    def test_basic(self):
        r = compute_attribution([AttributionInput("A",3.0), AttributionInput("B",1.5)])
        assert r.total_alpha_pct == 4.5

class TestRiskFactors:
    def test_basic(self):
        r = decompose_portfolio_risk_factors({"A":0.6,"B":0.4}, {"A":{"mkt":1.2},"B":{"mkt":0.8}})
        assert len(r) > 0

class TestTax:
    def test_tr(self):
        r = build_tax_report([TaxTrade("A",1000,100), TaxTrade("B",500,400)], "TR")
        assert r.short_term_tax > 0
        assert r.long_term_tax > 0

class TestExecution:
    def test_iceberg(self):
        r = create_iceberg_slices(1000, 100)
        assert len(r) == 10
        assert sum(s.quantity for s in r) == 1000
    def test_twap(self):
        r = build_twap_schedule(1000, "2024-01-01T09:00:00", "2024-01-01T17:00:00", 4)
        assert len(r) == 4
    def test_bracket(self):
        r = build_bracket_order(100, 5, 10, 50)
        assert r.stop_price < r.entry_price < r.take_profit_price
    def test_slippage_sim(self):
        levels = [OrderBookLevel(100,50), OrderBookLevel(101,50), OrderBookLevel(102,50)]
        r = simulate_execution_with_slippage(80, "buy", levels, 100)
        assert r.filled_qty == 80
        assert r.slippage_bps >= 0

class TestCompliance:
    def test_basic(self):
        rec = TransactionRecord("t1","2024-01-01","u1","o1","AAPL","buy",1000,150,"NYSE","s1")
        rules = [ComplianceRule("r1","quantity",">",500,"Large order","warning")]
        r = run_compliance_rule_engine(rec, rules)
        assert len(r) == 1
        assert r[0].severity == "warning"

class TestAlerts:
    def test_evaluate(self):
        cond = [AlertCondition("a1","High Vol","vol",">",20,"warning",["push"])]
        r = evaluate_alert_conditions({"vol": 25}, cond, {}, "2024-01-01T12:00:00")
        assert len(r["alerts"]) == 1
    def test_group(self):
        a1 = NotificationAlert("a1","x","warning",["push"],"msg",25,"2024-01-01T12:00:00","g1")
        a2 = NotificationAlert("a1","x","warning",["push"],"msg",26,"2024-01-01T12:00:30","g1")
        r = group_alerts([a1, a2], window_sec=120)
        assert len(r) == 1
    def test_escalation(self):
        a = NotificationAlert("a1","x","critical",["push"],"msg",25,"2024-01-01","g1")
        r = build_escalation_plan(a)
        assert len(r) == 4

class TestMarketIntel:
    def test_basic(self):
        sent = SentimentSummary(0.5, {"AAPL": 0.3}, ["AAPL"], [])
        r = build_market_intelligence_snapshot(
            [], [EarningsSurprise("AAPL",1.0,1.5)], 50, sent,
            [InsiderActivity("AAPL",1e6)], [ShortInterest("TSLA",15,20)])
        assert "conviction_score" in r

class TestPerformanceSnapshot:
    def test_basic(self):
        r = run_performance_snapshot(_equity(100), {})
        assert r["metrics"].observations > 0
