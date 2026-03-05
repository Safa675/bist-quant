import { describe, expect, it } from "vitest";
import {
  toAnalyticsUiResult,
  toBacktestUiResult,
  toComplianceUiResult,
  toOptimizationUiResult,
  toScreenerUiResult,
} from "@/lib/adapters";
import type { ComplianceRule } from "@/lib/types";

describe("toBacktestUiResult", () => {
  it("normalizes value/strategy aliases and percent-based monthly returns", () => {
    const result = toBacktestUiResult({
      metrics: { sharpe: 1.5 },
      equity_curve: [
        { date: "2024-01-02", value: 1.0, benchmark: 1.0, drawdown: 0.0 },
        { date: "2024-01-03", value: 1.01, benchmark: 1.005, drawdown: -0.01 },
      ],
      drawdown_curve: [{ date: "2024-01-03", value: -2.5 }],
      monthly_returns: [{ month: "2024-01", strategy_return: 2.1 }],
      top_holdings: [{ ticker: "THYAO", weight: 0.2 }],
      risk_metrics: { tail_risk: { var_95: -0.03 } },
      scenario_analysis: { worst_day: -0.04 },
    });

    expect(result.equity_curve[0].strategy).toBe(1.0);
    expect(result.drawdown_curve[0].drawdown).toBe(-2.5);
    expect(result.monthly_returns["2024"]?.Jan).toBeCloseTo(0.021);
    expect(result.holdings[0]).toEqual({ symbol: "THYAO", weight: 0.2 });
    expect(result.risk_metrics?.tail_risk?.var_95).toBe(-0.03);
    expect(result.scenario_analysis?.worst_day).toBe(-0.04);
  });

  it("normalizes nested monthly maps with numeric and named month keys", () => {
    const result = toBacktestUiResult({
      metrics: {},
      monthly_returns: {
        "2024": {
          "1": 0.5,
          Feb: "1.2",
        },
      },
    });

    expect(result.monthly_returns["2024"]?.Jan).toBeCloseTo(0.5);
    expect(result.monthly_returns["2024"]?.Feb).toBeCloseTo(0.012);
  });
});

describe("toAnalyticsUiResult", () => {
  it("maps performance/rolling/walk-forward/attribution sections", () => {
    const result = toAnalyticsUiResult({
      methods: ["performance", "rolling", "walk_forward", "attribution"],
      performance: { cagr: 12.1, sharpe: 1.2 },
      rolling: [{ date: "2024-01-03", rolling_sharpe_63d: 1.05 }],
      walk_forward: [{ split: 1, train_cagr: 10, test_cagr: 8 }],
      attribution: { momentum: 0.3 },
      benchmark: { symbol: "XU100", points: 300 },
    });

    expect(result.methods).toContain("walk_forward");
    expect(result.metrics.cagr).toBe(12.1);
    expect(result.rolling[0]?.rolling_sharpe).toBeCloseTo(1.05);
    expect(result.walk_forward?.length).toBe(1);
    expect(result.attribution).toEqual({ momentum: 0.3 });
    expect(result.benchmark).toEqual({ symbol: "XU100", points: 300 });
  });
});

describe("toOptimizationUiResult", () => {
  it("derives best params and sweep results from best_trial/trials", () => {
    const result = toOptimizationUiResult(
      {
        best_trial: {
          trial_id: 2,
          params: { top_n: 15, lookback: 63 },
          metrics: { sharpe: 1.71, cagr: 0.15 },
          score: 1.71,
          feasible: true,
        },
        trials: [
          {
            trial_id: 1,
            params: { top_n: 10, lookback: 42 },
            metrics: { sharpe: 1.42 },
            score: 1.42,
            feasible: true,
          },
          {
            trial_id: 2,
            params: { top_n: 15, lookback: 63 },
            metrics: { sharpe: 1.71 },
            score: 1.71,
            feasible: true,
          },
        ],
      },
      "sharpe",
    );

    expect(result.best_params).toEqual({ top_n: 15, lookback: 63 });
    expect(result.best_metric).toBeCloseTo(1.71);
    expect(result.sweep_results).toHaveLength(2);
  });

  it("keeps optional optimization sections", () => {
    const result = toOptimizationUiResult(
      {
        best_trial: {
          trial_id: 1,
          params: { top_n: 20 },
          metrics: { sharpe: 1.1 },
        },
        trials: [],
        pareto_front: [
          { trial_id: 1, params: { top_n: 20 }, metrics: { sharpe: 1.1 } },
        ],
        walk_forward: [{ split: 1 }],
        scenario_analysis: { stress_3sigma: -0.12 },
        constraints: { max_dd: 0.25 },
        objective: { maximize: "sharpe" },
      },
      "sharpe",
    );

    expect(result.pareto_front?.length).toBe(1);
    expect(result.walk_forward?.length).toBe(1);
    expect(result.scenario_analysis).toEqual({ stress_3sigma: -0.12 });
    expect(result.constraints).toEqual({ max_dd: 0.25 });
    expect(result.objective).toEqual({ maximize: "sharpe" });
  });
});

describe("toComplianceUiResult", () => {
  const rules: ComplianceRule[] = [
    {
      id: "max_order_qty",
      description: "Order quantity limit",
      field: "quantity",
      operator: ">",
      threshold: 100_000,
      severity: "warning",
    },
  ];

  it("maps passed to status and fills hit fields from rule fallback", () => {
    const result = toComplianceUiResult(
      {
        transaction_id: "tx-1",
        passed: false,
        hits: [{ rule_id: "max_order_qty" }],
      },
      rules,
    );

    expect(result.status).toBe("FAIL");
    expect(result.hits[0].operator).toBe(">");
    expect(result.hits[0].limit).toBe(100_000);
    expect(result.hits[0].message).toBe("Order quantity limit");
  });

  it("uses comparator/threshold/actual aliases from hit payload", () => {
    const result = toComplianceUiResult(
      {
        status: "FAIL",
        hits: [
          {
            id: "max_order_qty",
            comparator: ">=",
            threshold: 120_000,
            actual: 130_000,
          },
        ],
      },
      rules,
    );

    expect(result.hits[0].rule_id).toBe("max_order_qty");
    expect(result.hits[0].operator).toBe(">=");
    expect(result.hits[0].limit).toBe(120_000);
    expect(result.hits[0].observed).toBe(130_000);
  });
});

describe("toScreenerUiResult", () => {
  it("keeps count fallback and normalizes return_1m scale", () => {
    const result = toScreenerUiResult({
      meta: { total_matches: 2 },
      applied_filters: [{ key: "pe", min: 0, max: 15 }],
      rows: [
        { symbol: "THYAO", return_1m: 12 },
        { symbol: "GARAN", return_1m: 0.08 },
      ],
    });

    expect(result.count).toBe(2);
    expect(result.rows[0].return_1m).toBeCloseTo(0.12);
    expect(result.rows[1].return_1m).toBeCloseTo(0.08);
    expect(result.meta?.total_matches).toBe(2);
    expect(result.applied_filters?.length).toBe(1);
  });
});
