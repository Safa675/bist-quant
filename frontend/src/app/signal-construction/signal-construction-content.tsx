"use client";

import * as React from "react";
import {
  runSignalConstructionBacktest,
  runSignalConstructionFiveFactor,
  runSignalConstructionOrthogonalization,
  runSignalConstructionSnapshot,
} from "@/lib/api";
import { toBacktestUiResult } from "@/lib/adapters";
import type { BacktestUiResult } from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { KpiCard } from "@/components/shared/kpi-card";
import { ApiError } from "@/components/shared/api-error";
import { DataTable } from "@/components/shared/data-table";
import {
  PageScaffold,
  PageSidebar,
  PageMain,
  PageSectionStack,
} from "@/components/shared/page-scaffold";
import { FormField, FormGrid, FormLabel, FormRow } from "@/components/shared/form-field";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LazyBarMetricsChart as BarMetricsChart,
  LazyEquityCurveChart as EquityCurveChart,
} from "@/components/charts/lazy";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SelectInput } from "@/components/ui/select-input";
import { Checkbox } from "@/components/ui/checkbox";
import type { ColumnDef } from "@tanstack/react-table";
import { Loader2, Play, Wand2 } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

interface IndicatorMeta {
  key: string;
  label: string;
  category: string;
}

const INDICATORS: IndicatorMeta[] = [
  { key: "rsi", label: "RSI", category: "Oscillator" },
  { key: "macd", label: "MACD Histogram", category: "Trend" },
  { key: "bollinger", label: "Bollinger %B", category: "Volatility" },
  { key: "atr", label: "ATR", category: "Volatility" },
  { key: "stochastic", label: "Stochastic %K", category: "Oscillator" },
  { key: "adx", label: "ADX", category: "Trend" },
  { key: "supertrend", label: "Supertrend", category: "Trend" },
  { key: "ta_consensus", label: "TA Consensus", category: "Consensus" },
];

const INDICATOR_DEFAULTS: Record<string, Record<string, number | string>> = {
  rsi: { period: 14, oversold: 30, overbought: 70 },
  macd: { fast: 12, slow: 26, signal: 9, threshold: 0 },
  bollinger: { period: 20, std_dev: 2, lower: 0.2, upper: 0.8 },
  atr: { period: 14, lower_pct: 0.3, upper_pct: 0.7 },
  stochastic: { k_period: 14, d_period: 3, oversold: 20, overbought: 80 },
  adx: { period: 14, trend_threshold: 25 },
  supertrend: { period: 10, multiplier: 3 },
  ta_consensus: { interval: "1d", batch_size: 20 },
};

const FIVE_FACTOR_AXES = [
  "size",
  "value",
  "profitability",
  "investment",
  "momentum",
  "risk",
  "quality",
  "liquidity",
  "trading_intensity",
  "sentiment",
  "fundmom",
  "carry",
  "defensive",
] as const;

const snapshotColumns: ColumnDef<Record<string, unknown>, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol" },
  { accessorKey: "action", header: "Action" },
  {
    accessorKey: "combined_score",
    header: "Score",
    cell: ({ getValue }) => {
      const val = getValue<number | null>();
      if (val === null || val === undefined) return "—";
      return val.toFixed(3);
    },
  },
  { accessorKey: "buy_votes", header: "Buy" },
  { accessorKey: "sell_votes", header: "Sell" },
  { accessorKey: "hold_votes", header: "Hold" },
];

const indicatorSummaryCols: ColumnDef<Record<string, unknown>, unknown>[] = [
  { accessorKey: "name", header: "Indicator" },
  { accessorKey: "buy_count", header: "Buy" },
  { accessorKey: "hold_count", header: "Hold" },
  { accessorKey: "sell_count", header: "Sell" },
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

export function SignalConstructionContent() {
  const [enabledIndicators, setEnabledIndicators] = React.useState<Record<string, boolean>>({
    rsi: true,
    macd: true,
    supertrend: true,
  });
  const [indicatorParams, setIndicatorParams] = React.useState<Record<string, Record<string, number | string>>>(INDICATOR_DEFAULTS);

  const [universe, setUniverse] = React.useState("XU100");
  const [period, setPeriod] = React.useState("6mo");
  const [topN, setTopN] = React.useState(20);
  const [buyThreshold, setBuyThreshold] = React.useState(0.2);
  const [sellThreshold, setSellThreshold] = React.useState(-0.2);

  const [snapshotResult, setSnapshotResult] = React.useState<Record<string, unknown> | null>(null);
  const [backtestResult, setBacktestResult] = React.useState<BacktestUiResult | null>(null);

  const [axesEnabled, setAxesEnabled] = React.useState<Record<string, boolean>>(
    Object.fromEntries(FIVE_FACTOR_AXES.map((axis) => [axis, true])),
  );
  const [axisWeights, setAxisWeights] = React.useState<Record<string, number>>(
    Object.fromEntries(FIVE_FACTOR_AXES.map((axis) => [axis, 1])),
  );
  const [ffStartDate, setFfStartDate] = React.useState("2016-01-01");
  const [ffEndDate, setFfEndDate] = React.useState(todayIso());
  const [ffTopN, setFfTopN] = React.useState(20);
  const [fiveFactorResult, setFiveFactorResult] = React.useState<BacktestUiResult | null>(null);

  const [orthEnabled, setOrthEnabled] = React.useState(false);
  const [orthAxes, setOrthAxes] = React.useState<string[]>(["momentum", "value", "quality", "size"]);
  const [orthMinOverlap, setOrthMinOverlap] = React.useState(20);
  const [orthEpsilon, setOrthEpsilon] = React.useState(1e-8);
  const [orthResult, setOrthResult] = React.useState<Record<string, unknown> | null>(null);

  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  const activeIndicators = INDICATORS.filter((item) => enabledIndicators[item.key]);

  const buildPayload = React.useCallback(() => {
    const indicators: Record<string, Record<string, unknown>> = {};
    for (const item of INDICATORS) {
      indicators[item.key] = {
        enabled: !!enabledIndicators[item.key],
        params: indicatorParams[item.key] ?? INDICATOR_DEFAULTS[item.key] ?? {},
      };
    }

    return {
      universe,
      period,
      interval: "1d",
      top_n: topN,
      max_symbols: 100,
      buy_threshold: buyThreshold,
      sell_threshold: sellThreshold,
      indicators,
    };
  }, [buyThreshold, enabledIndicators, indicatorParams, period, sellThreshold, topN, universe]);

  const runSnapshot = async () => {
    if (activeIndicators.length === 0) {
      setError(new Error("Enable at least one indicator."));
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const response = await runSignalConstructionSnapshot(buildPayload());
      setSnapshotResult(response);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const runBacktest = async () => {
    if (activeIndicators.length === 0) {
      setError(new Error("Enable at least one indicator."));
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const response = await runSignalConstructionBacktest(buildPayload());
      setBacktestResult(toBacktestUiResult(response));
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const runFiveFactor = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      const response = await runSignalConstructionFiveFactor({
        factor_name: "five_factor_rotation",
        start_date: ffStartDate,
        end_date: ffEndDate,
        rebalance_frequency: "monthly",
        top_n: ffTopN,
      });
      setFiveFactorResult(response);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const applyOrthogonalization = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      const response = await runSignalConstructionOrthogonalization({
        enabled: orthEnabled,
        axes: orthAxes,
        min_overlap: orthMinOverlap,
        epsilon: orthEpsilon,
      });
      setOrthResult(response);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const snapshotSignals = React.useMemo(() => {
    if (!snapshotResult || !Array.isArray(snapshotResult.signals)) return [];
    return snapshotResult.signals.filter(isRecord);
  }, [snapshotResult]);

  const indicatorSummary = React.useMemo(() => {
    if (!snapshotResult || !Array.isArray(snapshotResult.indicator_summaries)) return [];
    return snapshotResult.indicator_summaries.filter(isRecord);
  }, [snapshotResult]);

  const backtestMetrics = backtestResult?.metrics;

  const enabledAxesWithWeights = React.useMemo(() => {
    const active = FIVE_FACTOR_AXES.filter((axis) => axesEnabled[axis]);
    const total = active.reduce((sum, axis) => sum + Math.max(axisWeights[axis] ?? 0, 0), 0);
    return active.map((axis) => ({
      axis,
      weight: total > 0 ? Math.max(axisWeights[axis] ?? 0, 0) / total : 0,
    }));
  }, [axesEnabled, axisWeights]);

  const orthConfigRows = React.useMemo(
    () =>
      orthAxes.map((axis, idx) => ({
        order: idx + 1,
        axis,
        enabled: axesEnabled[axis] ? "Yes" : "No",
        weight: axisWeights[axis] ?? 0,
      })),
    [axisWeights, axesEnabled, orthAxes],
  );

  return (
    <StaggerReveal>
      <StaggerItem><PageHeader title="Signal Construction" subtitle="Indicator builder, five-factor pipeline, and orthogonalization" /></StaggerItem>

      <ApiError error={error} className="mb-6" />

      <PageScaffold>
        <PageSidebar>
          <StaggerItem><SectionCard title="Indicator Controls">
            <FormGrid>
              <FormRow>
                <FormField label="Universe" htmlFor="signal-universe">
                  <SelectInput id="signal-universe" value={universe} onChange={(e) => setUniverse(e.target.value)}>
                    <option value="XU100">XU100</option>
                    <option value="XU030">XU030</option>
                    <option value="XUTUM">XUTUM</option>
                  </SelectInput>
                </FormField>
                <FormField label="Period" htmlFor="signal-period">
                  <SelectInput id="signal-period" value={period} onChange={(e) => setPeriod(e.target.value)}>
                    <option value="1mo">1mo</option>
                    <option value="3mo">3mo</option>
                    <option value="6mo">6mo</option>
                    <option value="1y">1y</option>
                    <option value="2y">2y</option>
                  </SelectInput>
                </FormField>
              </FormRow>

              <FormRow>
                <FormField label="Top N" htmlFor="signal-topn">
                  <Input id="signal-topn" type="number" min={5} max={200} value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
                </FormField>
                <FormField label="Buy Threshold" htmlFor="signal-buy-threshold">
                  <Input id="signal-buy-threshold" type="number" min={-1} max={1} step={0.05} value={buyThreshold} onChange={(e) => setBuyThreshold(Number(e.target.value))} />
                </FormField>
              </FormRow>

              <FormField label="Sell Threshold" htmlFor="signal-sell-threshold">
                <Input id="signal-sell-threshold" type="number" min={-1} max={1} step={0.05} value={sellThreshold} onChange={(e) => setSellThreshold(Number(e.target.value))} />
              </FormField>

              <FormLabel>Indicators</FormLabel>
              {INDICATORS.map((item) => (
                <div key={item.key} className="space-y-2 rounded-[var(--radius-sm)] border border-[var(--border)] p-3">
                  <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                    <Checkbox
                      checked={!!enabledIndicators[item.key]}
                      onChange={(e) =>
                        setEnabledIndicators((prev) => ({
                          ...prev,
                          [item.key]: e.target.checked,
                        }))
                      }
                    />
                    {item.label} ({item.category})
                  </label>

                  {enabledIndicators[item.key] && (
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(indicatorParams[item.key] ?? INDICATOR_DEFAULTS[item.key] ?? {}).map(([paramKey, paramValue]) => (
                        <Input
                          key={`${item.key}-${paramKey}`}
                          type={typeof paramValue === "number" ? "number" : "text"}
                          step={typeof paramValue === "number" ? 0.1 : undefined}
                          value={String(paramValue)}
                          onChange={(e) => {
                            const raw = e.target.value;
                            const nextValue = typeof paramValue === "number" ? Number(raw) : raw;
                            setIndicatorParams((prev) => ({
                              ...prev,
                              [item.key]: {
                                ...(prev[item.key] ?? {}),
                                [paramKey]: nextValue,
                              },
                            }));
                          }}
                          aria-label={`${item.label} ${paramKey}`}
                        />
                      ))}
                    </div>
                  )}
                </div>
              ))}

              <div className="grid grid-cols-2 gap-2">
                <Button data-testid="signal-snapshot-run" onClick={runSnapshot} disabled={isSubmitting || activeIndicators.length === 0}>
                  {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Wand2 className="mr-2 h-4 w-4" />}
                  Snapshot
                </Button>
                <Button data-testid="signal-backtest-run" onClick={runBacktest} disabled={isSubmitting || activeIndicators.length === 0}>
                  {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                  Backtest
                </Button>
              </div>
            </FormGrid>
          </SectionCard></StaggerItem>
        </PageSidebar>

        <PageMain>
          <PageSectionStack>
            <StaggerItem><Tabs defaultValue="builder">
              <TabsList className="w-full overflow-x-auto">
                <TabsTrigger value="builder">Indicator Builder</TabsTrigger>
                <TabsTrigger value="fivefactor">Five-Factor Pipeline</TabsTrigger>
                <TabsTrigger value="orth">Orthogonalization</TabsTrigger>
              </TabsList>

              <TabsContent value="builder" className="space-y-[var(--section-gap)]">
                {snapshotResult && (
                  <SectionCard title="Snapshot Outputs">
                    <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-4">
                      <KpiCard title="Signals" value={snapshotSignals.length} decimals={0} animate={false} />
                      <KpiCard title="Buy" value={snapshotSignals.filter((row) => row.action === "BUY").length} decimals={0} animate={false} />
                      <KpiCard title="Sell" value={snapshotSignals.filter((row) => row.action === "SELL").length} decimals={0} animate={false} />
                      <KpiCard title="Hold" value={snapshotSignals.filter((row) => row.action === "HOLD").length} decimals={0} animate={false} />
                    </div>

                    <Tabs defaultValue="rank">
                      <TabsList className="w-full overflow-x-auto">
                        <TabsTrigger value="rank">Rank</TabsTrigger>
                        <TabsTrigger value="table">Signal Table</TabsTrigger>
                        <TabsTrigger value="indicator">Indicator Breakdown</TabsTrigger>
                      </TabsList>

                      <TabsContent value="rank">
                        <SectionCard title="Cross-Sectional Rank">
                          <BarMetricsChart
                            data={snapshotSignals
                              .slice(0, 40)
                              .map((row) => ({
                                name: String(row.symbol ?? "?"),
                                value: toNumber(row.combined_score) ?? 0,
                              }))}
                            horizontal
                            height={Math.max(280, snapshotSignals.length * 20)}
                          />
                        </SectionCard>
                      </TabsContent>

                      <TabsContent value="table">
                        <SectionCard title="Signal Table" noPadding>
                          <DataTable
                            columns={snapshotColumns}
                            data={snapshotSignals}
                            pageSize={15}
                            enableExport
                            exportFilename="signal-snapshot.csv"
                          />
                        </SectionCard>
                      </TabsContent>

                      <TabsContent value="indicator">
                        <SectionCard title="Indicator Votes" noPadding>
                          <DataTable columns={indicatorSummaryCols} data={indicatorSummary} pageSize={10} />
                        </SectionCard>
                      </TabsContent>
                    </Tabs>
                  </SectionCard>
                )}

                {backtestResult && (
                  <>
                    <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-5">
                      <KpiCard title="CAGR" value={backtestMetrics?.cagr ?? null} decimals={3} animate={false} />
                      <KpiCard title="Sharpe" value={backtestMetrics?.sharpe ?? null} decimals={3} animate={false} />
                      <KpiCard title="Max DD" value={backtestMetrics?.max_drawdown ?? null} decimals={3} animate={false} />
                      <KpiCard title="Win Rate" value={backtestMetrics?.win_rate ?? null} decimals={3} animate={false} />
                      <KpiCard title="Volatility" value={backtestMetrics?.annualized_volatility ?? null} decimals={3} animate={false} />
                    </div>

                    <SectionCard title="Indicator Backtest Equity Curve">
                      <EquityCurveChart data={backtestResult.equity_curve} height={360} />
                    </SectionCard>

                    <SectionCard title="Current Holdings" noPadding>
                      <DataTable
                        columns={[
                          { accessorKey: "symbol", header: "Symbol" },
                          {
                            accessorKey: "weight",
                            header: "Weight",
                            cell: ({ getValue }) => `${(((getValue<number>() ?? 0) as number) * 100).toFixed(2)}%`,
                          },
                        ]}
                        data={backtestResult.holdings}
                        pageSize={12}
                        enableExport
                        exportFilename="signal-current-holdings.csv"
                      />
                    </SectionCard>
                  </>
                )}

                {!snapshotResult && !backtestResult && (
                  <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                    <p className="text-small text-[var(--text-muted)]">
                      Configure indicators on the left and run Snapshot or Backtest.
                    </p>
                  </SectionCard>
                )}
              </TabsContent>

              <TabsContent value="fivefactor" className="space-y-[var(--section-gap)]">
                <SectionCard title="Axes & Weights">
                  <FormGrid>
                    <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                      {FIVE_FACTOR_AXES.map((axis) => (
                        <div key={axis} className="rounded-[var(--radius-sm)] border border-[var(--border)] p-3">
                          <label className="mb-2 flex items-center gap-2 text-small text-[var(--text-muted)]">
                            <Checkbox
                              checked={!!axesEnabled[axis]}
                              onChange={(e) =>
                                setAxesEnabled((prev) => ({
                                  ...prev,
                                  [axis]: e.target.checked,
                                }))
                              }
                            />
                            {axis}
                          </label>
                          <Input
                            type="number"
                            min={0}
                            max={5}
                            step={0.1}
                            value={axisWeights[axis] ?? 0}
                            onChange={(e) =>
                              setAxisWeights((prev) => ({
                                ...prev,
                                [axis]: Number(e.target.value),
                              }))
                            }
                            disabled={!axesEnabled[axis]}
                          />
                        </div>
                      ))}
                    </div>

                    <FormRow>
                      <FormField label="Start" htmlFor="five-factor-start">
                        <Input id="five-factor-start" type="date" value={ffStartDate} onChange={(e) => setFfStartDate(e.target.value)} />
                      </FormField>
                      <FormField label="End" htmlFor="five-factor-end">
                        <Input id="five-factor-end" type="date" value={ffEndDate} onChange={(e) => setFfEndDate(e.target.value)} />
                      </FormField>
                      <FormField label="Top N" htmlFor="five-factor-topn">
                        <Input id="five-factor-topn" type="number" min={5} max={100} value={ffTopN} onChange={(e) => setFfTopN(Number(e.target.value))} />
                      </FormField>
                    </FormRow>

                    <Button data-testid="five-factor-run" onClick={runFiveFactor} disabled={isSubmitting}>
                      {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                      Run Five-Factor Backtest
                    </Button>
                  </FormGrid>
                </SectionCard>

                <SectionCard title="Normalized Axis Allocation">
                  <BarMetricsChart
                    data={enabledAxesWithWeights.map((row) => ({ name: row.axis, value: row.weight * 100 }))}
                    height={320}
                  />
                </SectionCard>

                {fiveFactorResult && (
                  <>
                    <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-4">
                      <KpiCard title="CAGR" value={fiveFactorResult.metrics.cagr ?? null} decimals={3} animate={false} />
                      <KpiCard title="Sharpe" value={fiveFactorResult.metrics.sharpe ?? null} decimals={3} animate={false} />
                      <KpiCard title="Sortino" value={fiveFactorResult.metrics.sortino ?? null} decimals={3} animate={false} />
                      <KpiCard title="Max DD" value={fiveFactorResult.metrics.max_drawdown ?? null} decimals={3} animate={false} />
                    </div>

                    <SectionCard title="Five-Factor Equity Curve">
                      <EquityCurveChart data={fiveFactorResult.equity_curve} height={360} />
                    </SectionCard>
                  </>
                )}
              </TabsContent>

              <TabsContent value="orth" className="space-y-[var(--section-gap)]">
                <SectionCard title="Orthogonalization Settings">
                  <FormGrid>
                    <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                      <Checkbox checked={orthEnabled} onChange={(e) => setOrthEnabled(e.target.checked)} />
                      Enable orthogonalization
                    </label>

                    <FormField label="Axes (comma-separated)" htmlFor="orth-axes">
                      <Input
                        id="orth-axes"
                        value={orthAxes.join(",")}
                        onChange={(e) =>
                          setOrthAxes(
                            e.target.value
                              .split(",")
                              .map((x) => x.trim())
                              .filter(Boolean),
                          )
                        }
                      />
                    </FormField>

                    <FormRow>
                      <FormField label="Min Overlap" htmlFor="orth-min-overlap">
                        <Input
                          id="orth-min-overlap"
                          type="number"
                          min={2}
                          max={1000}
                          value={orthMinOverlap}
                          onChange={(e) => setOrthMinOverlap(Number(e.target.value))}
                        />
                      </FormField>
                      <FormField label="Epsilon" htmlFor="orth-epsilon">
                        <Input
                          id="orth-epsilon"
                          type="number"
                          step="0.00000001"
                          value={orthEpsilon}
                          onChange={(e) => setOrthEpsilon(Number(e.target.value))}
                        />
                      </FormField>
                    </FormRow>

                    <Button onClick={applyOrthogonalization} disabled={isSubmitting}>
                      {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Wand2 className="mr-2 h-4 w-4" />}
                      Apply Orthogonalization
                    </Button>
                  </FormGrid>
                </SectionCard>

                <SectionCard title="Configuration Summary" noPadding>
                  <DataTable
                    columns={[
                      { accessorKey: "order", header: "Order" },
                      { accessorKey: "axis", header: "Axis" },
                      { accessorKey: "enabled", header: "Enabled" },
                      { accessorKey: "weight", header: "Weight" },
                    ]}
                    data={orthConfigRows}
                    pageSize={10}
                  />
                </SectionCard>

                {orthResult && (
                  <SectionCard title="Endpoint Diagnostics">
                    <div className="overflow-auto rounded-[var(--radius)] bg-[var(--surface-2)] p-3">
                      <code className="block whitespace-pre-wrap text-xs text-[var(--text-muted)]">
                        {JSON.stringify(orthResult, null, 2)}
                      </code>
                    </div>
                  </SectionCard>
                )}
              </TabsContent>
            </Tabs></StaggerItem>
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
