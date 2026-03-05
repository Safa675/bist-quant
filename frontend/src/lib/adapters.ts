import type {
  AnalyticsRawResult,
  AnalyticsUiResult,
  BacktestRawResult,
  BacktestUiResult,
  ComplianceRawResult,
  ComplianceRawRule,
  ComplianceRule,
  ComplianceUiResult,
  OptimizationRawResult,
  OptimizationTrial,
  OptimizationUiResult,
  ScreenerRawResult,
  ScreenerRow,
  ScreenerUiResult,
} from "@/lib/types";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value.trim());
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function monthKeyToLabels(monthKey: string): { year: string; month: string } | null {
  const parts = monthKey.split("-");
  if (parts.length < 2) return null;
  const year = parts[0];
  const monthIdx = Number(parts[1]) - 1;
  if (!year || monthIdx < 0 || monthIdx > 11) return null;
  return { year, month: MONTHS[monthIdx] };
}

function monthLabelFromRaw(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;

  const monthNum = Number(trimmed);
  if (Number.isFinite(monthNum)) {
    const idx = monthNum - 1;
    if (idx >= 0 && idx <= 11) return MONTHS[idx];
  }

  const normalized = trimmed.slice(0, 3).toLowerCase();
  const idx = MONTHS.findIndex((m) => m.toLowerCase() === normalized);
  if (idx >= 0) return MONTHS[idx];
  return null;
}

function normalizeMonthlyReturn(value: number): number {
  // Treat values outside [-1, 1] as percent units and convert to fraction.
  return Math.abs(value) > 1 ? value / 100 : value;
}

function toMonthlyReturnsMap(
  raw: BacktestRawResult["monthly_returns"],
): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {};
  if (!raw) return out;

  if (Array.isArray(raw)) {
    for (const row of raw) {
      if (!isRecord(row)) continue;
      const month = asString(row.month);
      const retRaw = asNumber(row.strategy_return);
      const ret = retRaw === null ? null : normalizeMonthlyReturn(retRaw);
      if (!month || ret === null) continue;
      const labels = monthKeyToLabels(month);
      if (!labels) continue;
      if (!out[labels.year]) out[labels.year] = {};
      out[labels.year][labels.month] = ret;
    }
    return out;
  }

  if (isRecord(raw)) {
    for (const [yearOrMonth, value] of Object.entries(raw)) {
      if (isRecord(value)) {
        const year = yearOrMonth.trim();
        if (!/^\d{4}$/.test(year)) continue;
        if (!out[year]) out[year] = {};
        for (const [month, monthValue] of Object.entries(value)) {
          const monthLabel = monthLabelFromRaw(month);
          const numRaw = asNumber(monthValue);
          if (monthLabel && numRaw !== null) {
            out[year][monthLabel] = normalizeMonthlyReturn(numRaw);
          }
        }
        continue;
      }
      const numRaw = asNumber(value);
      const num = numRaw === null ? null : normalizeMonthlyReturn(numRaw);
      const labels = monthKeyToLabels(yearOrMonth);
      if (num === null || !labels) continue;
      if (!out[labels.year]) out[labels.year] = {};
      out[labels.year][labels.month] = num;
    }
  }
  return out;
}

export function toBacktestUiResult(raw: unknown): BacktestUiResult {
  const payload = isRecord(raw) ? (raw as BacktestRawResult) : ({ metrics: {} } as BacktestRawResult);
  const metrics = isRecord(payload.metrics) ? payload.metrics : {};

  const equity_curve = (Array.isArray(payload.equity_curve) ? payload.equity_curve : [])
    .map((row) => {
      if (!isRecord(row)) return null;
      const date = asString(row.date);
      const strategy = asNumber(row.strategy) ?? asNumber(row.value);
      if (!date || strategy === null) return null;
      return {
        date,
        strategy,
        benchmark: asNumber(row.benchmark),
        drawdown: asNumber(row.drawdown),
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  const drawdown_curve = (Array.isArray(payload.drawdown_curve) ? payload.drawdown_curve : [])
    .map((row) => {
      if (!isRecord(row)) return null;
      const date = asString(row.date);
      const drawdown = asNumber(row.drawdown) ?? asNumber(row.value);
      if (!date || drawdown === null) return null;
      return { date, drawdown };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  const normalized_drawdown = drawdown_curve.length
    ? drawdown_curve
    : equity_curve
        .filter((row) => row.drawdown !== null)
        .map((row) => ({ date: row.date, drawdown: row.drawdown as number }));

  const holdings = (() => {
    if (Array.isArray(payload.holdings)) {
      return payload.holdings
        .map((row) => {
          if (!isRecord(row)) return null;
          const symbol = asString(row.symbol) || asString(row.ticker);
          const weight = asNumber(row.weight);
          if (!symbol || weight === null) return null;
          return { symbol, weight };
        })
        .filter((row): row is NonNullable<typeof row> => row !== null);
    }
    if (Array.isArray(payload.top_holdings)) {
      return payload.top_holdings
        .map((row) => {
          if (!isRecord(row)) return null;
          const symbol = asString(row.ticker);
          const weight = asNumber(row.weight);
          if (!symbol || weight === null) return null;
          return { symbol, weight };
        })
        .filter((row): row is NonNullable<typeof row> => row !== null);
    }
    return [];
  })();

  return {
    metrics,
    equity_curve,
    drawdown_curve: normalized_drawdown,
    monthly_returns: toMonthlyReturnsMap(payload.monthly_returns),
    holdings,
    rolling_metrics: (Array.isArray(payload.rolling_metrics) ? payload.rolling_metrics : [])
      .map((row) => {
        if (!isRecord(row)) return null;
        const date = asString(row.date);
        if (!date) return null;
        return {
          date,
          rolling_sharpe_63d: asNumber(row.rolling_sharpe_63d),
          rolling_volatility_63d: asNumber(row.rolling_volatility_63d),
          rolling_max_drawdown_126d: asNumber(row.rolling_max_drawdown_126d),
        };
      })
      .filter((row): row is NonNullable<typeof row> => row !== null),
    risk_metrics: isRecord(payload.risk_metrics) ? payload.risk_metrics : undefined,
    scenario_analysis: isRecord(payload.scenario_analysis) ? payload.scenario_analysis : undefined,
    sector_exposure: isRecord(payload.sector_exposure)
      ? Object.fromEntries(
          Object.entries(payload.sector_exposure)
            .map(([k, v]) => [k, asNumber(v)])
            .filter(([, v]) => v !== null),
        ) as Record<string, number>
      : undefined,
    summary: isRecord(payload.summary) ? payload.summary : undefined,
    raw: isRecord(payload) ? payload : undefined,
  };
}

export function toAnalyticsUiResult(raw: unknown): AnalyticsUiResult {
  const payload = isRecord(raw) ? (raw as AnalyticsRawResult) : {};
  const methods = Array.isArray(payload.methods) ? payload.methods.filter((x): x is string => typeof x === "string") : [];

  const metrics: Record<string, number | null> = {};
  if (isRecord(payload.performance)) {
    for (const [key, value] of Object.entries(payload.performance)) {
      metrics[key] = asNumber(value);
    }
  }
  if (isRecord(payload.risk)) {
    for (const [key, value] of Object.entries(payload.risk)) {
      if (!(key in metrics)) metrics[key] = asNumber(value);
    }
  }

  const rolling = (Array.isArray(payload.rolling) ? payload.rolling : [])
    .map((row) => {
      if (!isRecord(row)) return null;
      const date = asString(row.date);
      if (!date) return null;
      return {
        date,
        rolling_sharpe: asNumber(row.rolling_sharpe_63d) ?? asNumber(row.rolling_sharpe),
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  return {
    methods,
    metrics,
    rolling,
    performance: isRecord(payload.performance) ? payload.performance : undefined,
    risk: isRecord(payload.risk) ? payload.risk : undefined,
    stress: Array.isArray(payload.stress) ? payload.stress.filter(isRecord) : undefined,
    transaction_costs: isRecord(payload.transaction_costs) ? payload.transaction_costs : undefined,
    monte_carlo: isRecord(payload.monte_carlo) ? payload.monte_carlo : undefined,
    walk_forward: Array.isArray(payload.walk_forward)
      ? payload.walk_forward.filter(isRecord)
      : undefined,
    attribution:
      payload.attribution === null
        ? null
        : isRecord(payload.attribution)
          ? payload.attribution
          : undefined,
    benchmark: isRecord(payload.benchmark) ? payload.benchmark : undefined,
  };
}

function normalizeTrial(raw: unknown): OptimizationTrial | null {
  if (!isRecord(raw)) return null;
  if (!isRecord(raw.params) || !isRecord(raw.metrics)) return null;

  const params: Record<string, number> = {};
  for (const [key, value] of Object.entries(raw.params)) {
    const num = asNumber(value);
    if (num !== null) params[key] = num;
  }

  return {
    trial_id: Number(raw.trial_id ?? 0),
    params,
    metrics: raw.metrics,
    score: asNumber(raw.score),
    feasible: typeof raw.feasible === "boolean" ? raw.feasible : undefined,
  };
}

function trialMetricValue(trial: OptimizationTrial | null, metricKey: string): number | null {
  if (!trial) return null;
  if (isRecord(trial.metrics)) {
    const direct = asNumber(trial.metrics[metricKey]);
    if (direct !== null) return direct;
  }
  return asNumber(trial.score);
}

export function toOptimizationUiResult(raw: unknown, metricKey: string): OptimizationUiResult {
  const payload = isRecord(raw) ? (raw as OptimizationRawResult) : {};
  const trials = (Array.isArray(payload.trials) ? payload.trials : [])
    .map(normalizeTrial)
    .filter((row): row is OptimizationTrial => row !== null);
  const best_trial = normalizeTrial(payload.best_trial);
  const best_metric = trialMetricValue(best_trial, metricKey);
  const best_params = best_trial?.params ?? {};

  const sweep_results = trials
    .map((trial) => {
      const metric = trialMetricValue(trial, metricKey);
      if (metric === null) return null;
      return { params: trial.params, metric };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  return {
    best_params,
    best_metric,
    sweep_results,
    best_trial: best_trial ?? undefined,
    trials,
    pareto_front: (Array.isArray(payload.pareto_front) ? payload.pareto_front : [])
      .map(normalizeTrial)
      .filter((row): row is OptimizationTrial => row !== null),
    walk_forward: Array.isArray(payload.walk_forward) ? payload.walk_forward.filter(isRecord) : undefined,
    scenario_analysis: isRecord(payload.scenario_analysis) ? payload.scenario_analysis : undefined,
    constraints: isRecord(payload.constraints) ? payload.constraints : undefined,
    objective: isRecord(payload.objective) ? payload.objective : undefined,
  };
}

export function toScreenerUiResult(raw: unknown): ScreenerUiResult {
  const payload = isRecord(raw) ? (raw as ScreenerRawResult) : {};
  const rows: ScreenerRow[] = (Array.isArray(payload.rows) ? payload.rows : [])
    .map((row) => {
      if (!isRecord(row)) return null;
      const normalized = { ...row } as ScreenerRow;
      const ret1m = asNumber(normalized.return_1m);
      if (ret1m !== null) {
        normalized.return_1m = Math.abs(ret1m) > 1 ? ret1m / 100 : ret1m;
      }
      return normalized;
    })
    .filter((row): row is ScreenerRow => row !== null);
  const count = asNumber((payload as Record<string, unknown>).count)
    ?? (isRecord(payload.meta) ? asNumber(payload.meta.total_matches) : null)
    ?? rows.length;

  return {
    count,
    rows,
    meta: isRecord(payload.meta) ? payload.meta : undefined,
    applied_filters: Array.isArray(payload.applied_filters)
      ? payload.applied_filters.filter(isRecord)
      : undefined,
  };
}

export function toComplianceUiRules(raw: unknown): { rules: ComplianceRule[] } {
  const payload = isRecord(raw) ? raw : {};
  const rows = Array.isArray(payload.rules) ? payload.rules : [];
  const rules = rows
    .map((rule) => {
      if (!isRecord(rule)) return null;
      const item = rule as ComplianceRawRule;
      const id = asString(item.id);
      if (!id) return null;
      return {
        id,
        description: asString(item.description) || asString(item.message) || id,
        field: asString(item.field),
        operator: asString(item.operator) || asString(item.comparator),
        threshold: asNumber(item.threshold) ?? 0,
        severity: item.severity === "critical" ? "critical" : "warning",
      };
    })
    .filter((row): row is ComplianceRule => row !== null);
  return { rules };
}

export function toComplianceUiResult(
  raw: unknown,
  rules: ComplianceRule[] = [],
): ComplianceUiResult {
  const payload = isRecord(raw) ? (raw as ComplianceRawResult) : {};
  const byRuleId = new Map<string, ComplianceRule>(rules.map((rule) => [rule.id, rule]));
  const statusRaw = asString(payload.status).toUpperCase();
  const status: "PASS" | "FAIL" = statusRaw === "FAIL"
    ? "FAIL"
    : payload.passed === false
      ? "FAIL"
      : "PASS";

  const hits = (Array.isArray(payload.hits) ? payload.hits : [])
    .map((hit) => {
      if (!isRecord(hit)) return null;
      const rule_id = asString(hit.rule_id) || asString(hit.id);
      const fallback = byRuleId.get(rule_id);
      return {
        rule_id,
        message: asString(hit.message) || fallback?.description || "Rule violation",
        severity: asString(hit.severity) || fallback?.severity || "warning",
        field: asString(hit.field) || fallback?.field,
        operator: asString(hit.operator) || asString(hit.comparator) || fallback?.operator,
        observed: asNumber(hit.observed) ?? asNumber(hit.actual) ?? asNumber(hit.value),
        limit: asNumber(hit.limit) ?? asNumber(hit.threshold) ?? fallback?.threshold ?? null,
      };
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  return {
    transaction_id: asString(payload.transaction_id),
    status,
    hits,
  };
}
