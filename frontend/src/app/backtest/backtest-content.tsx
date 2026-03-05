"use client";

import * as React from "react";
import { useApi } from "@/hooks/use-api";
import { useJobPolling } from "@/hooks/use-job-polling";
import { getFactorCatalog, createJob, listJobs, cancelJob } from "@/lib/api";
import type { BacktestResult, FactorCatalog } from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { KpiCard } from "@/components/shared/kpi-card";
import { ApiError } from "@/components/shared/api-error";
import { JobHistoryPanel } from "@/components/shared/job-history-panel";
import { StatusBadge } from "@/components/shared/status-badge";
import { DataTable } from "@/components/shared/data-table";
import {
  PageScaffold,
  PageSidebar,
  PageMain,
  PageSectionStack,
  PageKpiRow,
} from "@/components/shared/page-scaffold";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";
import { FormField, FormGrid, FormRow } from "@/components/shared/form-field";
import {
  LazyEquityCurveChart as EquityCurveChart,
  LazyDrawdownChart as DrawdownChart,
  LazyMonthlyReturnsHeatmap as MonthlyReturnsHeatmap,
  LazyBarMetricsChart as BarMetricsChart,
  LazyScatterPlot as ScatterPlot,
} from "@/components/charts/lazy";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SelectInput } from "@/components/ui/select-input";
import { Checkbox } from "@/components/ui/checkbox";
import type { ColumnDef } from "@tanstack/react-table";
import { Download, Play, Loader2 } from "lucide-react";

function pct(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  const normalized = Math.abs(value) <= 1 ? value * 100 : value;
  return `${normalized > 0 ? "+" : ""}${normalized.toFixed(2)}%`;
}

function triggerDownload(filename: string, mime: string, content: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

type HoldingRow = { symbol: string; weight: number };
const holdingColumns: ColumnDef<HoldingRow, unknown>[] = [
  { accessorKey: "symbol", header: "Ticker" },
  {
    accessorKey: "weight",
    header: "Weight %",
    cell: ({ getValue }) => `${((getValue<number>() ?? 0) * 100).toFixed(2)}%`,
  },
];

type SectorRow = { sector: string; weight: number };
const sectorColumns: ColumnDef<SectorRow, unknown>[] = [
  { accessorKey: "sector", header: "Sector" },
  {
    accessorKey: "weight",
    header: "Weight %",
    cell: ({ getValue }) => `${((getValue<number>() ?? 0) * 100).toFixed(2)}%`,
  },
];

export function BacktestContent() {
  const [factorName, setFactorName] = React.useState("");
  const [startDate, setStartDate] = React.useState("2020-01-01");
  const [endDate, setEndDate] = React.useState("2025-12-31");
  const [topN, setTopN] = React.useState(20);
  const [maxWeight, setMaxWeight] = React.useState(0.25);
  const [rebalFreq, setRebalFreq] = React.useState("monthly");

  const [useRegime, setUseRegime] = React.useState(true);
  const [useLiquidity, setUseLiquidity] = React.useState(true);
  const [useSlippage, setUseSlippage] = React.useState(true);
  const [slippageBps, setSlippageBps] = React.useState(5);
  const [useStopLoss, setUseStopLoss] = React.useState(false);
  const [stopLossThreshold, setStopLossThreshold] = React.useState(0.15);
  const [useVolTargeting, setUseVolTargeting] = React.useState(false);
  const [targetDownsideVol, setTargetDownsideVol] = React.useState(0.2);

  const [jobId, setJobId] = React.useState<string | null>(null);
  const [submitError, setSubmitError] = React.useState<Error | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const catalogFetcher = React.useCallback(() => getFactorCatalog(), []);
  const { data: catalog } = useApi<FactorCatalog>(catalogFetcher);
  const { job, isPolling } = useJobPolling(jobId);

  const jobsFetcher = React.useCallback(() => listJobs(15), []);
  const { data: jobsData, refetch: refetchJobs } = useApi(jobsFetcher);

  React.useEffect(() => {
    if (catalog?.signals?.length && !factorName) {
      setFactorName(catalog.signals[0]);
    }
  }, [catalog, factorName]);

  React.useEffect(() => {
    try {
      const raw = window.localStorage.getItem("bq_backtest_prefill");
      if (!raw) return;
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      const prefillSignal = typeof parsed.signal === "string" ? parsed.signal : null;
      const prefillTopN = typeof parsed.top_n === "number" ? parsed.top_n : null;
      if (prefillSignal) setFactorName(prefillSignal);
      if (prefillTopN !== null) setTopN(prefillTopN);
      window.localStorage.removeItem("bq_backtest_prefill");
    } catch {
      // Keep existing defaults when local prefill is malformed.
    }
  }, []);

  React.useEffect(() => {
    if (job && (job.status === "completed" || job.status === "failed")) {
      refetchJobs();
    }
  }, [job?.status, refetchJobs, job]);

  const handleSubmit = async () => {
    if (!factorName) return;
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      const created = await createJob("backtest", {
        factor_name: factorName,
        start_date: startDate,
        end_date: endDate,
        top_n: topN,
        max_position_weight: maxWeight,
        rebalance_frequency: rebalFreq,
        use_regime_filter: useRegime,
        use_liquidity_filter: useLiquidity,
        use_slippage: useSlippage,
        slippage_bps: slippageBps,
        use_stop_loss: useStopLoss,
        stop_loss_threshold: stopLossThreshold,
        use_vol_targeting: useVolTargeting,
        target_downside_vol: targetDownsideVol,
        benchmark: "XU100",
      });
      setJobId(created.id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancelJob = async (id: string) => {
    try {
      await cancelJob(id);
      refetchJobs();
    } catch {
      // no-op
    }
  };

  const result = job?.status === "completed" ? (job.result as unknown as BacktestResult) : null;
  const metrics = result?.metrics;
  const holdingsRows: HoldingRow[] = (result?.holdings ?? []).map((h) => ({
    symbol: h.symbol,
    weight: h.weight,
  }));
  const sectorRows: SectorRow[] = Object.entries(result?.sector_exposure ?? {}).map(([sector, weight]) => ({
    sector,
    weight,
  }));

  const rollingSharpePoints = (result?.rolling_metrics ?? [])
    .filter((p) => typeof p.rolling_sharpe_63d === "number")
    .map((p, idx) => ({ x: idx, y: p.rolling_sharpe_63d as number, label: p.date }));

  const rollingVolPoints = (result?.rolling_metrics ?? [])
    .filter((p) => typeof p.rolling_volatility_63d === "number")
    .map((p, idx) => ({ x: idx, y: (p.rolling_volatility_63d as number) * 100, label: p.date }));

  const risk = result?.risk_metrics;
  const scenario = result?.scenario_analysis;

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader title="Backtest" subtitle="Factor-based backtesting engine" actions={job ? <StatusBadge status={job.status} /> : undefined} />
      </StaggerItem>

      <PageScaffold>
        <PageSidebar>
          <StaggerItem>
          <SectionCard title="Configuration">
            <FormGrid>
              <FormField label="Factor" htmlFor="backtest-factor">
                <SelectInput id="backtest-factor" value={factorName} onChange={(e) => setFactorName(e.target.value)}>
                  {catalog?.signals?.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </SelectInput>
              </FormField>

              <FormRow>
                <FormField label="Start Date" htmlFor="backtest-start-date">
                  <Input id="backtest-start-date" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                </FormField>
                <FormField label="End Date" htmlFor="backtest-end-date">
                  <Input id="backtest-end-date" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                </FormField>
              </FormRow>

              <FormRow>
                <FormField label="Top N" htmlFor="backtest-top-n">
                  <Input id="backtest-top-n" type="number" min={5} max={100} value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
                </FormField>
                <FormField label="Max Weight" htmlFor="backtest-max-weight">
                  <Input id="backtest-max-weight" type="number" min={0.05} max={1} step={0.01} value={maxWeight} onChange={(e) => setMaxWeight(Number(e.target.value))} />
                </FormField>
              </FormRow>

              <FormField label="Rebalance" htmlFor="backtest-rebalance">
                <SelectInput id="backtest-rebalance" value={rebalFreq} onChange={(e) => setRebalFreq(e.target.value)}>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                  <option value="quarterly">Quarterly</option>
                </SelectInput>
              </FormField>

              <fieldset className="space-y-[var(--space-2)] rounded-[var(--radius)] border border-[var(--border)] p-[var(--space-3)]">
                <legend className="px-1 text-micro font-semibold uppercase tracking-wide text-[var(--text-faint)]">Risk Controls</legend>
                <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                  <Checkbox checked={useRegime} onChange={(e) => setUseRegime(e.target.checked)} />
                  Regime filter
                </label>
                <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                  <Checkbox checked={useLiquidity} onChange={(e) => setUseLiquidity(e.target.checked)} />
                  Liquidity filter
                </label>
                <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                  <Checkbox checked={useSlippage} onChange={(e) => setUseSlippage(e.target.checked)} />
                  Slippage
                </label>
                <FormField label="Slippage (bps)" htmlFor="backtest-slippage">
                  <Input
                    id="backtest-slippage"
                    type="number"
                    min={0}
                    max={1000}
                    value={slippageBps}
                    disabled={!useSlippage}
                    onChange={(e) => setSlippageBps(Number(e.target.value))}
                  />
                </FormField>
                <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                  <Checkbox checked={useStopLoss} onChange={(e) => setUseStopLoss(e.target.checked)} />
                  Stop loss
                </label>
                <FormField label="Stop threshold" htmlFor="backtest-stop-threshold">
                  <Input
                    id="backtest-stop-threshold"
                    type="number"
                    min={0.01}
                    max={1}
                    step={0.01}
                    value={stopLossThreshold}
                    disabled={!useStopLoss}
                    onChange={(e) => setStopLossThreshold(Number(e.target.value))}
                  />
                </FormField>
                <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                  <Checkbox checked={useVolTargeting} onChange={(e) => setUseVolTargeting(e.target.checked)} />
                  Vol targeting
                </label>
                <FormField label="Target downside vol" htmlFor="backtest-target-vol">
                  <Input
                    id="backtest-target-vol"
                    type="number"
                    min={0.01}
                    max={2}
                    step={0.01}
                    value={targetDownsideVol}
                    disabled={!useVolTargeting}
                    onChange={(e) => setTargetDownsideVol(Number(e.target.value))}
                  />
                </FormField>
              </fieldset>

              <Button data-testid="backtest-run" className="w-full" onClick={handleSubmit} disabled={isSubmitting || isPolling || !factorName}>
                {isSubmitting || isPolling ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                {isPolling ? "Running…" : "Run Backtest"}
              </Button>
            </FormGrid>
          </SectionCard>
          </StaggerItem>

          <ApiError error={submitError} />
          <ApiError error={job?.status === "failed" ? job.error ?? "Job failed" : null} />

          {jobsData && <JobHistoryPanel jobs={jobsData.jobs} onCancel={handleCancelJob} onRefresh={refetchJobs} />}
        </PageSidebar>

        <PageMain>
          <PageSectionStack>
            {result && metrics && (
              <>
                <StaggerItem>
                <PageKpiRow>
                  <KpiCard title="CAGR" value={metrics.cagr} suffix="%" decimals={2} />
                  <KpiCard title="Sharpe" value={metrics.sharpe} decimals={2} />
                  <KpiCard title="Sortino" value={metrics.sortino} decimals={2} />
                  <KpiCard title="Max DD" value={metrics.max_drawdown} suffix="%" decimals={2} />
                </PageKpiRow>
                </StaggerItem>

                <StaggerItem>
                <Tabs defaultValue="equity">
                  <TabsList className="w-full overflow-x-auto">
                    <TabsTrigger value="equity">Equity Curve</TabsTrigger>
                    <TabsTrigger value="monthly">Monthly Returns</TabsTrigger>
                    <TabsTrigger value="rolling">Rolling Metrics</TabsTrigger>
                    <TabsTrigger value="risk">Risk</TabsTrigger>
                    <TabsTrigger value="holdings">Holdings</TabsTrigger>
                    <TabsTrigger value="export">Export</TabsTrigger>
                  </TabsList>

                  <TabsContent value="equity">
                    <SectionCard title="Equity Curve">
                      <EquityCurveChart data={result.equity_curve} height={320} />
                    </SectionCard>
                    {result.drawdown_curve.length > 0 && (
                      <SectionCard title="Drawdown">
                        <DrawdownChart data={result.drawdown_curve} height={220} />
                      </SectionCard>
                    )}
                    <SectionCard title="All Metrics">
                      <BarMetricsChart
                        data={Object.entries(metrics)
                          .filter(([, v]) => typeof v === "number" && v !== null)
                          .map(([name, value]) => ({ name, value: value as number }))}
                        height={300}
                      />
                    </SectionCard>
                  </TabsContent>

                  <TabsContent value="monthly">
                    <SectionCard title="Monthly Heatmap">
                      <MonthlyReturnsHeatmap data={result.monthly_returns} />
                    </SectionCard>
                  </TabsContent>

                  <TabsContent value="rolling">
                    <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-2">
                      <SectionCard title="Rolling Sharpe (63d)">
                        <ScatterPlot data={rollingSharpePoints} xLabel="Observation" yLabel="Sharpe" referenceY={0} height={240} />
                      </SectionCard>
                      <SectionCard title="Rolling Volatility (63d)">
                        <ScatterPlot data={rollingVolPoints} xLabel="Observation" yLabel="Volatility %" referenceY={0} height={240} />
                      </SectionCard>
                    </div>
                  </TabsContent>

                  <TabsContent value="risk">
                    <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-2">
                      <SectionCard title="Tail Risk">
                        <div className="grid grid-cols-2 gap-[var(--space-2)]">
                          <KpiCard title="VaR 95%" value={risk?.tail_risk?.var_95 ?? null} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="CVaR 95%" value={risk?.tail_risk?.cvar_95 ?? null} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="VaR 99%" value={risk?.tail_risk?.var_99 ?? null} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="CVaR 99%" value={risk?.tail_risk?.cvar_99 ?? null} suffix="%" decimals={2} animate={false} />
                        </div>
                      </SectionCard>
                      <SectionCard title="Scenario Analysis">
                        <div className="grid grid-cols-2 gap-[var(--space-2)]">
                          <KpiCard title="Best Day" value={scenario?.best_day ?? null} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="Worst Day" value={scenario?.worst_day ?? null} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="-2σ Day" value={scenario?.stress_1d_minus_2sigma ?? null} suffix="%" decimals={2} animate={false} />
                          <KpiCard title="-3σ Day" value={scenario?.stress_1d_minus_3sigma ?? null} suffix="%" decimals={2} animate={false} />
                        </div>
                      </SectionCard>
                    </div>
                    <SectionCard title="MAE / MFE">
                      <div className="grid grid-cols-2 gap-[var(--space-2)] lg:grid-cols-4">
                        <KpiCard title="MAE 1d" value={risk?.mae_mfe?.mae_1d ?? null} suffix="%" decimals={2} animate={false} />
                        <KpiCard title="MFE 1d" value={risk?.mae_mfe?.mfe_1d ?? null} suffix="%" decimals={2} animate={false} />
                        <KpiCard title="Worst 5d" value={risk?.mae_mfe?.worst_5d ?? null} suffix="%" decimals={2} animate={false} />
                        <KpiCard title="Best 5d" value={risk?.mae_mfe?.best_5d ?? null} suffix="%" decimals={2} animate={false} />
                      </div>
                    </SectionCard>
                    {sectorRows.length > 0 && (
                      <SectionCard title="Sector Exposure">
                        <DataTable columns={sectorColumns} data={sectorRows} pageSize={10} />
                      </SectionCard>
                    )}
                  </TabsContent>

                  <TabsContent value="holdings">
                    <SectionCard title="Top Holdings" noPadding>
                      <DataTable columns={holdingColumns} data={holdingsRows} pageSize={20} enableExport exportFilename="backtest-holdings.csv" />
                    </SectionCard>
                  </TabsContent>

                  <TabsContent value="export">
                    <SectionCard title="Export Results">
                      <div className="flex flex-wrap gap-[var(--space-2)]">
                        <Button
                          variant="outline"
                          onClick={() => {
                            const rows = result.equity_curve.map((row) => `${row.date},${row.strategy},${row.benchmark ?? ""},${row.drawdown ?? ""}`);
                            const csv = ["date,strategy,benchmark,drawdown", ...rows].join("\n");
                            triggerDownload("backtest_equity_curve.csv", "text/csv", csv);
                          }}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          Equity Curve CSV
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => {
                            triggerDownload("backtest_result.json", "application/json", JSON.stringify(result.raw ?? result, null, 2));
                          }}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          Full Result JSON
                        </Button>
                      </div>
                      <p className="mt-3 text-small text-[var(--text-muted)]">
                        Latest tail risk: VaR95 {pct(risk?.tail_risk?.var_95)} | CVaR95 {pct(risk?.tail_risk?.cvar_95)}
                      </p>
                    </SectionCard>
                  </TabsContent>
                </Tabs>
                </StaggerItem>
              </>
            )}

            {!result && !isPolling && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <p className="text-small text-[var(--text-muted)]">Configure and run a backtest to see results here.</p>
              </SectionCard>
            )}

            {isPolling && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <div className="flex flex-col items-center gap-[var(--space-3)]">
                  <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
                  <p className="text-small text-[var(--text-muted)]">Running backtest…</p>
                </div>
              </SectionCard>
            )}
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
