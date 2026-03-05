"use client";

import * as React from "react";
import { useApi } from "@/hooks/use-api";
import { useJobPolling } from "@/hooks/use-job-polling";
import { getFactorCatalog, getFactorDetail, runOptimization } from "@/lib/api";
import { toOptimizationUiResult } from "@/lib/adapters";
import type { FactorCatalog } from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { KpiCard } from "@/components/shared/kpi-card";
import { ApiError } from "@/components/shared/api-error";
import { StatusBadge } from "@/components/shared/status-badge";
import { DataTable } from "@/components/shared/data-table";
import {
  PageScaffold,
  PageSidebar,
  PageMain,
  PageSectionStack,
} from "@/components/shared/page-scaffold";
import { FormField, FormGrid, FormRow } from "@/components/shared/form-field";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  LazyOptimizationHeatmap as OptimizationHeatmap,
  LazyScatterPlot as ScatterPlot,
} from "@/components/charts/lazy";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SelectInput } from "@/components/ui/select-input";
import type { ColumnDef } from "@tanstack/react-table";
import { Settings2, Loader2 } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

type ParamType = "int" | "float";
interface SweepSpec {
  key: string;
  type: ParamType;
  defaultMin: number;
  defaultMax: number;
  defaultStep: number;
}

interface ActiveSweepSpec extends SweepSpec {
  enabled: boolean;
  min: number;
  max: number;
  step: number;
}

const trialColumns: ColumnDef<Record<string, unknown>, unknown>[] = [
  { accessorKey: "trial_id", header: "Trial" },
  { accessorKey: "score", header: "Score" },
  { accessorKey: "feasible", header: "Feasible" },
  { accessorKey: "params", header: "Params" },
];

function inferParamSpecs(detail: Record<string, unknown> | null): SweepSpec[] {
  if (!detail) return [];
  const specs: SweepSpec[] = [];

  const signalParams =
    detail.signal_params && typeof detail.signal_params === "object"
      ? (detail.signal_params as Record<string, unknown>)
      : {};
  for (const [key, value] of Object.entries(signalParams)) {
    if (typeof value !== "number" || Number.isNaN(value)) continue;
    const intType = Number.isInteger(value);
    const base = Math.abs(value) > 0 ? Math.abs(value) : 1;
    specs.push({
      key: `factors.0.signal_params.${key}`,
      type: intType ? "int" : "float",
      defaultMin: intType ? Math.max(1, Math.floor(base / 2)) : Number((base / 2).toFixed(4)),
      defaultMax: intType ? Math.ceil(base * 3) : Number((base * 2).toFixed(4)),
      defaultStep: intType ? Math.max(1, Math.floor(base / 5)) : Number((base / 5).toFixed(4)),
    });
  }

  const portfolioOptions =
    detail.portfolio_options && typeof detail.portfolio_options === "object"
      ? (detail.portfolio_options as Record<string, unknown>)
      : {};
  const topN = typeof portfolioOptions.top_n === "number" ? portfolioOptions.top_n : 20;
  specs.push({
    key: "top_n",
    type: "int",
    defaultMin: Math.max(5, Math.floor(topN / 2)),
    defaultMax: Math.min(100, Math.ceil(topN * 3)),
    defaultStep: 5,
  });
  return specs;
}

export function OptimizationContent() {
  const [signal, setSignal] = React.useState("");
  const [method, setMethod] = React.useState<"grid" | "random">("grid");
  const [metric, setMetric] = React.useState("sharpe");
  const [startDate, setStartDate] = React.useState("2020-01-01");
  const [endDate, setEndDate] = React.useState("2025-12-31");
  const [maxTrials, setMaxTrials] = React.useState(50);
  const [trainRatio, setTrainRatio] = React.useState(0.7);

  const [activeSpecs, setActiveSpecs] = React.useState<ActiveSweepSpec[]>([]);
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [submitError, setSubmitError] = React.useState<Error | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const catalogFetcher = React.useCallback(() => getFactorCatalog(), []);
  const { data: catalog } = useApi<FactorCatalog>(catalogFetcher);

  React.useEffect(() => {
    if (catalog?.signals?.length && !signal) setSignal(catalog.signals[0]);
  }, [catalog, signal]);

  const detailFetcher = React.useCallback(
    () => (signal ? getFactorDetail(signal) : Promise.resolve(null)),
    [signal],
  );
  const { data: signalDetail } = useApi<Record<string, unknown> | null>(detailFetcher, {
    skip: !signal,
  });

  React.useEffect(() => {
    const specs = inferParamSpecs(signalDetail ?? null).map((spec) => ({
      ...spec,
      enabled: false,
      min: spec.defaultMin,
      max: spec.defaultMax,
      step: spec.defaultStep,
    }));
    setActiveSpecs(specs);
  }, [signalDetail]);

  const { job, isPolling } = useJobPolling(jobId);
  const result = React.useMemo(() => {
    if (job?.status !== "completed" || !job.result) return null;
    return toOptimizationUiResult(job.result, metric);
  }, [job, metric]);

  const enabledSpecs = activeSpecs.filter((s) => s.enabled);

  const handleSubmit = async () => {
    if (!signal) return;
    if (enabledSpecs.length === 0) {
      setSubmitError(new Error("Enable at least one parameter to optimize."));
      return;
    }
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      const parameter_space = enabledSpecs.map((spec) => ({
        key: spec.key,
        type: spec.type,
        min: spec.min,
        max: spec.max,
        step: spec.step,
      }));

      const created = await runOptimization({
        signal,
        method,
        max_trials: maxTrials,
        train_ratio: trainRatio,
        parameter_space,
        params: {
          start_date: startDate,
          end_date: endDate,
          factor_name: null,
          factors: [{ name: signal, weight: 1.0 }],
          rebalance_frequency: "monthly",
          top_n: 20,
        },
      });
      setJobId(created.id);
    } catch (err) {
      setSubmitError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const sweepResults = result?.sweep_results ?? [];
  const heatmap = React.useMemo(() => {
    if (!sweepResults.length) return null;
    const paramKeys = Object.keys(sweepResults[0].params);
    if (paramKeys.length < 2) return null;
    const xKey = paramKeys[0];
    const yKey = paramKeys[1];
    const xValues = [...new Set(sweepResults.map((r) => r.params[xKey]))].sort((a, b) => a - b);
    const yValues = [...new Set(sweepResults.map((r) => r.params[yKey]))].sort((a, b) => a - b);
    const matrix = yValues.map((y) =>
      xValues.map((x) => {
        const row = sweepResults.find((r) => r.params[xKey] === x && r.params[yKey] === y);
        return row?.metric ?? 0;
      }),
    );
    return { xKey, yKey, xValues, yValues, matrix };
  }, [sweepResults]);

  const scatterData = sweepResults.map((row) => {
    const firstKey = Object.keys(row.params)[0];
    return {
      x: row.params[firstKey],
      y: row.metric,
      label: Object.entries(row.params).map(([k, v]) => `${k}=${v}`).join(", "),
    };
  });

  const trialRows: Record<string, unknown>[] = (result?.trials ?? []).map((trial) => ({
    trial_id: trial.trial_id,
    score: trial.score ?? null,
    feasible: trial.feasible ?? false,
    params: JSON.stringify(trial.params),
  }));

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader
          title="Optimization"
          subtitle="Parameter sweep & signal optimization"
          actions={job ? <StatusBadge status={job.status} /> : undefined}
        />
      </StaggerItem>

      <PageScaffold>
        <PageSidebar>
          <StaggerItem>
          <SectionCard title="Sweep Config">
            <FormGrid>
              <FormField label="Signal" htmlFor="optimization-signal">
                <SelectInput id="optimization-signal" value={signal} onChange={(e) => setSignal(e.target.value)}>
                  {catalog?.signals?.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </SelectInput>
              </FormField>

              <FormRow>
                <FormField label="Method" htmlFor="optimization-method">
                  <SelectInput
                    id="optimization-method"
                    value={method}
                    onChange={(e) => setMethod(e.target.value as "grid" | "random")}
                  >
                    <option value="grid">Grid</option>
                    <option value="random">Random</option>
                  </SelectInput>
                </FormField>
                <FormField label="Metric" htmlFor="optimization-metric">
                  <SelectInput id="optimization-metric" value={metric} onChange={(e) => setMetric(e.target.value)}>
                    <option value="sharpe">Sharpe</option>
                    <option value="cagr">CAGR</option>
                    <option value="sortino">Sortino</option>
                    <option value="calmar">Calmar</option>
                  </SelectInput>
                </FormField>
              </FormRow>

              <FormRow>
                <FormField label="Start" htmlFor="optimization-start">
                  <Input id="optimization-start" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                </FormField>
                <FormField label="End" htmlFor="optimization-end">
                  <Input id="optimization-end" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                </FormField>
              </FormRow>

              <FormRow>
                <FormField label="Max Trials" htmlFor="optimization-max-trials">
                  <Input id="optimization-max-trials" type="number" min={4} max={500} value={maxTrials} onChange={(e) => setMaxTrials(Number(e.target.value))} />
                </FormField>
                <FormField label="Train Ratio" htmlFor="optimization-train-ratio">
                  <Input
                    id="optimization-train-ratio"
                    type="number"
                    min={0.5}
                    max={0.9}
                    step={0.05}
                    value={trainRatio}
                    onChange={(e) => setTrainRatio(Number(e.target.value))}
                  />
                </FormField>
              </FormRow>

              <fieldset className="space-y-[var(--space-2)] rounded-[var(--radius)] border border-[var(--border)] p-[var(--space-3)]">
                <legend className="px-1 text-micro font-semibold uppercase tracking-wide text-[var(--text-faint)]">
                  Parameter Sweep
                </legend>
                {activeSpecs.map((spec, idx) => (
                  <div key={spec.key} className="rounded-[var(--radius-sm)] border border-[var(--border)] p-2">
                    <label className="mb-2 flex items-center gap-2 text-small text-[var(--text-muted)]">
                      <input
                        type="checkbox"
                        checked={spec.enabled}
                        onChange={(e) =>
                          setActiveSpecs((prev) =>
                            prev.map((row, i) => (i === idx ? { ...row, enabled: e.target.checked } : row)),
                          )
                        }
                      />
                      {spec.key}
                    </label>
                    {spec.enabled && (
                      <div className="grid grid-cols-3 gap-2">
                        <Input
                          type="number"
                          value={spec.min}
                          onChange={(e) =>
                            setActiveSpecs((prev) =>
                              prev.map((row, i) => (i === idx ? { ...row, min: Number(e.target.value) } : row)),
                            )
                          }
                        />
                        <Input
                          type="number"
                          value={spec.max}
                          onChange={(e) =>
                            setActiveSpecs((prev) =>
                              prev.map((row, i) => (i === idx ? { ...row, max: Number(e.target.value) } : row)),
                            )
                          }
                        />
                        <Input
                          type="number"
                          value={spec.step}
                          onChange={(e) =>
                            setActiveSpecs((prev) =>
                              prev.map((row, i) => (i === idx ? { ...row, step: Number(e.target.value) } : row)),
                            )
                          }
                        />
                      </div>
                    )}
                  </div>
                ))}
              </fieldset>

              <Button data-testid="optimization-run" className="w-full" onClick={handleSubmit} disabled={isSubmitting || isPolling || !signal}>
                {isSubmitting || isPolling ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Settings2 className="mr-2 h-4 w-4" />}
                {isPolling ? "Optimizing…" : "Run Optimization"}
              </Button>
            </FormGrid>
          </SectionCard>
          </StaggerItem>

          <ApiError error={submitError} />
          <ApiError error={job?.status === "failed" ? job.error ?? "Job failed" : null} />
        </PageSidebar>

        <PageMain>
          <PageSectionStack>
            {result?.best_params && (
              <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-3">
                <KpiCard title={`Best ${metric}`} value={result.best_metric} decimals={4} />
                {Object.entries(result.best_params).map(([k, v]) => (
                  <KpiCard key={k} title={`Best ${k}`} value={v} decimals={2} animate={false} />
                ))}
              </div>
            )}

            {result && (
              <Tabs defaultValue="heatmap">
                <TabsList className="w-full overflow-x-auto">
                  <TabsTrigger value="heatmap">Heatmap</TabsTrigger>
                  <TabsTrigger value="trials">All Trials</TabsTrigger>
                  <TabsTrigger value="export">Export</TabsTrigger>
                </TabsList>

                <TabsContent value="heatmap">
                  {heatmap && (
                    <SectionCard title="Parameter Heatmap">
                      <OptimizationHeatmap
                        xValues={heatmap.xValues}
                        yValues={heatmap.yValues}
                        matrix={heatmap.matrix}
                        xLabel={heatmap.xKey}
                        yLabel={heatmap.yKey}
                        metricLabel={metric}
                      />
                    </SectionCard>
                  )}
                  {scatterData.length > 0 && (
                    <SectionCard title={`${metric} vs parameter`}>
                      <ScatterPlot
                        data={scatterData}
                        xLabel={Object.keys(sweepResults[0]?.params ?? { x: 0 })[0]}
                        yLabel={metric}
                        referenceY={result.best_metric ?? undefined}
                        height={260}
                      />
                    </SectionCard>
                  )}
                </TabsContent>

                <TabsContent value="trials">
                  <SectionCard title="Trial Table" noPadding>
                    <DataTable columns={trialColumns} data={trialRows} pageSize={30} enableExport exportFilename="optimization-trials.csv" />
                  </SectionCard>
                </TabsContent>

                <TabsContent value="export">
                  <SectionCard title="Export & Handoff">
                    <div className="flex flex-wrap gap-[var(--space-2)]">
                      <Button
                        variant="outline"
                        onClick={() => {
                          const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
                          const url = URL.createObjectURL(blob);
                          const a = document.createElement("a");
                          a.href = url;
                          a.download = "optimization-result.json";
                          a.click();
                          URL.revokeObjectURL(url);
                        }}
                      >
                        Export Result JSON
                      </Button>
                      <Button
                        onClick={() => {
                          localStorage.setItem(
                            "bq_backtest_prefill",
                            JSON.stringify({
                              signal,
                              top_n: result.best_params.top_n ?? 20,
                            }),
                          );
                          window.location.href = "/backtest";
                        }}
                      >
                        Send Best Config to Backtest
                      </Button>
                    </div>
                  </SectionCard>
                </TabsContent>
              </Tabs>
            )}

            {!result && !isPolling && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <p className="text-small text-[var(--text-muted)]">Configure and run optimization to populate heatmaps and trials.</p>
              </SectionCard>
            )}

            {isPolling && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <div className="flex flex-col items-center gap-[var(--space-3)]">
                  <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
                  <p className="text-small text-[var(--text-muted)]">Running optimization…</p>
                </div>
              </SectionCard>
            )}
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
