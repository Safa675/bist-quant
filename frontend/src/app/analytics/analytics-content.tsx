"use client";

import * as React from "react";
import { runAnalytics, listJobs, getAnalyticsBenchmarkXU100 } from "@/lib/api";
import type { AnalyticsUiResult, JobPayload } from "@/lib/types";
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
  PageKpiRow,
} from "@/components/shared/page-scaffold";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";
import { FormField, FormGrid, FormLabel } from "@/components/shared/form-field";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  LazyBarMetricsChart as BarMetricsChart,
  LazyScatterPlot as ScatterPlot,
} from "@/components/charts/lazy";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { SelectInput } from "@/components/ui/select-input";
import { Input } from "@/components/ui/input";
import type { ColumnDef } from "@tanstack/react-table";
import { BarChart3, Loader2 } from "lucide-react";

const METHOD_OPTIONS = [
  "performance",
  "rolling",
  "walk_forward",
  "monte_carlo",
  "attribution",
  "risk",
  "stress",
  "transaction_costs",
] as const;

const walkForwardCols: ColumnDef<Record<string, unknown>, unknown>[] = [
  { accessorKey: "split", header: "Split" },
  { accessorKey: "train_start", header: "Train Start" },
  { accessorKey: "train_end", header: "Train End" },
  { accessorKey: "test_start", header: "Test Start" },
  { accessorKey: "test_end", header: "Test End" },
  { accessorKey: "train_cagr", header: "IS CAGR %" },
  { accessorKey: "test_cagr", header: "OOS CAGR %" },
  { accessorKey: "train_sharpe", header: "IS Sharpe" },
  { accessorKey: "test_sharpe", header: "OOS Sharpe" },
  { accessorKey: "p_value", header: "p-value" },
];

function parseCsvToCurve(csv: string): Array<{ date: string; value: number }> {
  return csv
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [date, valueRaw] = line.split(",").map((s) => s.trim());
      return { date, value: Number(valueRaw) };
    })
    .filter((row) => row.date && Number.isFinite(row.value));
}

function toCsvFromCurve(curve: Array<{ date: string; value: number }>): string {
  return curve.map((p) => `${p.date},${p.value}`).join("\n");
}

function extractBacktestCurve(job: JobPayload): Array<{ date: string; value: number }> {
  const result = job.result;
  if (!result || typeof result !== "object") return [];
  const rows = Array.isArray((result as Record<string, unknown>).equity_curve)
    ? ((result as Record<string, unknown>).equity_curve as Array<Record<string, unknown>>)
    : [];
  return rows
    .map((row) => ({
      date: typeof row.date === "string" ? row.date : "",
      value:
        typeof row.value === "number"
          ? row.value
          : typeof row.strategy === "number"
            ? row.strategy
            : Number.NaN,
    }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

export function AnalyticsContent() {
  const [source, setSource] = React.useState<"paste" | "job">("paste");
  const [equityCsv, setEquityCsv] = React.useState("");
  const [jobs, setJobs] = React.useState<JobPayload[]>([]);
  const [selectedJobId, setSelectedJobId] = React.useState("");
  const [includeBenchmark, setIncludeBenchmark] = React.useState(true);
  const [walkForwardSplits, setWalkForwardSplits] = React.useState(5);
  const [trainRatio, setTrainRatio] = React.useState(0.7);
  const [mcIterations, setMcIterations] = React.useState(750);
  const [mcHorizon, setMcHorizon] = React.useState(252);
  const [methods, setMethods] = React.useState<string[]>([
    "performance",
    "rolling",
    "walk_forward",
    "monte_carlo",
    "attribution",
    "risk",
  ]);

  const [result, setResult] = React.useState<AnalyticsUiResult | null>(null);
  const [submitError, setSubmitError] = React.useState<Error | null>(null);
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [isLoadingJobs, setIsLoadingJobs] = React.useState(false);

  React.useEffect(() => {
    const loadJobs = async () => {
      setIsLoadingJobs(true);
      try {
        const rows = await listJobs(50);
        const completedBacktests = rows.jobs.filter(
          (job) => job.kind === "backtest" && job.status === "completed" && job.result,
        );
        setJobs(completedBacktests);
        if (completedBacktests.length > 0 && !selectedJobId) {
          setSelectedJobId(completedBacktests[0].id);
        }
      } catch {
        // noop
      } finally {
        setIsLoadingJobs(false);
      }
    };
    void loadJobs();
  }, [selectedJobId]);

  const toggleMethod = (method: string) => {
    setMethods((prev) => {
      if (prev.includes(method)) return prev.filter((m) => m !== method);
      return [...prev, method];
    });
  };

  const currentCurve = React.useMemo(() => {
    if (source === "paste") return parseCsvToCurve(equityCsv);
    const selected = jobs.find((job) => job.id === selectedJobId);
    if (!selected) return [];
    return extractBacktestCurve(selected);
  }, [source, equityCsv, jobs, selectedJobId]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setEquityCsv(text);
    setSource("paste");
  };

  const handleSubmit = async () => {
    if (currentCurve.length < 2) {
      setSubmitError(new Error("Provide at least 2 equity points (date,value)."));
      return;
    }
    setSubmitError(null);
    setIsSubmitting(true);
    try {
      let benchmarkCurve: Array<{ date: string; value: number }> = [];
      if (includeBenchmark) {
        const benchmark = await getAnalyticsBenchmarkXU100();
        benchmarkCurve = benchmark.curve;
      }

      const response = await runAnalytics({
        equity_curve: currentCurve,
        benchmark_curve: benchmarkCurve,
        include_benchmark: includeBenchmark,
        benchmark_symbol: "XU100",
        methods: methods.length ? methods : ["performance"],
        walk_forward_splits: walkForwardSplits,
        train_ratio: trainRatio,
        monte_carlo_iterations: mcIterations,
        monte_carlo_horizon: mcHorizon,
      });
      setResult(response);
    } catch (err) {
      setSubmitError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmitting(false);
    }
  };

  const metricsEntries = Object.entries(result?.metrics ?? {})
    .filter(([, v]) => typeof v === "number" && v !== null);

  const rollingSharpe = (result?.rolling ?? [])
    .filter((p) => typeof p.rolling_sharpe === "number")
    .map((p, i) => ({ x: i, y: p.rolling_sharpe as number, label: p.date }));

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader title="Analytics" subtitle="Deep-dive portfolio analytics" />
      </StaggerItem>

      <PageScaffold>
        <PageSidebar>
          <StaggerItem>
          <SectionCard title="Input & Config">
            <FormGrid>
              <FormField label="Source" htmlFor="analytics-source">
                <SelectInput
                  id="analytics-source"
                  value={source}
                  onChange={(e) => setSource(e.target.value as "paste" | "job")}
                >
                  <option value="paste">Paste CSV</option>
                  <option value="job">Backtest Job History</option>
                </SelectInput>
              </FormField>

              {source === "paste" ? (
                <>
                  <FormField label="Paste CSV (date,value)" htmlFor="analytics-equity-csv">
                    <Textarea
                      id="analytics-equity-csv"
                      className="h-40 resize-y font-mono text-xs"
                      placeholder="2024-01-02,100.00&#10;2024-01-03,101.50&#10;..."
                      value={equityCsv}
                      onChange={(e) => setEquityCsv(e.target.value)}
                    />
                  </FormField>
                  <FormField label="Upload CSV" htmlFor="analytics-upload">
                    <input
                      id="analytics-upload"
                      type="file"
                      accept=".csv,text/csv"
                      onChange={handleFileUpload}
                      className="block w-full rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface-2)] px-3 py-2 text-small"
                    />
                  </FormField>
                </>
              ) : (
                <FormField label="Completed backtest run" htmlFor="analytics-job">
                  <SelectInput
                    id="analytics-job"
                    value={selectedJobId}
                    onChange={(e) => setSelectedJobId(e.target.value)}
                    disabled={isLoadingJobs || jobs.length === 0}
                  >
                    {jobs.length === 0 ? (
                      <option value="">No completed backtest jobs</option>
                    ) : (
                      jobs.map((job) => (
                        <option key={job.id} value={job.id}>
                          {job.id} ({job.created_at.slice(0, 10)})
                        </option>
                      ))
                    )}
                  </SelectInput>
                </FormField>
              )}

              <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                <Checkbox checked={includeBenchmark} onChange={(e) => setIncludeBenchmark(e.target.checked)} />
                Include XU100 benchmark
              </label>

              <div className="grid grid-cols-2 gap-[var(--space-2)]">
                <FormField label="Walk-forward splits" htmlFor="analytics-splits">
                  <Input
                    id="analytics-splits"
                    type="number"
                    min={0}
                    max={20}
                    value={walkForwardSplits}
                    onChange={(e) => setWalkForwardSplits(Number(e.target.value))}
                  />
                </FormField>
                <FormField label="Train ratio" htmlFor="analytics-train-ratio">
                  <Input
                    id="analytics-train-ratio"
                    type="number"
                    min={0.3}
                    max={0.9}
                    step={0.05}
                    value={trainRatio}
                    onChange={(e) => setTrainRatio(Number(e.target.value))}
                  />
                </FormField>
                <FormField label="MC iterations" htmlFor="analytics-mc-iter">
                  <Input
                    id="analytics-mc-iter"
                    type="number"
                    min={100}
                    max={10000}
                    step={50}
                    value={mcIterations}
                    onChange={(e) => setMcIterations(Number(e.target.value))}
                  />
                </FormField>
                <FormField label="MC horizon" htmlFor="analytics-mc-horizon">
                  <Input
                    id="analytics-mc-horizon"
                    type="number"
                    min={30}
                    max={2520}
                    step={10}
                    value={mcHorizon}
                    onChange={(e) => setMcHorizon(Number(e.target.value))}
                  />
                </FormField>
              </div>

              <div className="space-y-[var(--space-2)]">
                <FormLabel>Methods</FormLabel>
                {METHOD_OPTIONS.map((m) => (
                  <label key={m} className="flex items-center gap-2 py-0.5 text-small text-[var(--text-muted)]">
                    <Checkbox checked={methods.includes(m)} onChange={() => toggleMethod(m)} />
                    {m}
                  </label>
                ))}
              </div>

              <Button data-testid="analytics-run" className="w-full" onClick={handleSubmit} disabled={isSubmitting}>
                {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BarChart3 className="mr-2 h-4 w-4" />}
                {isSubmitting ? "Analyzing…" : "Run Analytics"}
              </Button>

              <ApiError error={submitError} />
              <p className="text-micro text-[var(--text-faint)]">Loaded points: {currentCurve.length}</p>
              {source === "job" && currentCurve.length > 0 && (
                <Button
                  variant="outline"
                  onClick={() => {
                    setSource("paste");
                    setEquityCsv(toCsvFromCurve(currentCurve));
                  }}
                >
                  Copy job curve to CSV editor
                </Button>
              )}
            </FormGrid>
          </SectionCard>
          </StaggerItem>
        </PageSidebar>

        <PageMain>
          <PageSectionStack>
            {metricsEntries.length > 0 && (
              <StaggerItem>
              <PageKpiRow>
                {metricsEntries.slice(0, 6).map(([key, val]) => (
                  <KpiCard key={key} title={key.replace(/_/g, " ")} value={val} decimals={3} />
                ))}
              </PageKpiRow>
              </StaggerItem>
            )}

            {result && (
              <StaggerItem>
              <Tabs defaultValue="metrics">
                <TabsList className="w-full overflow-x-auto">
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="rolling">Rolling</TabsTrigger>
                  <TabsTrigger value="walkforward">Walk-Forward</TabsTrigger>
                  <TabsTrigger value="montecarlo">Monte Carlo</TabsTrigger>
                  <TabsTrigger value="attribution">Attribution</TabsTrigger>
                  <TabsTrigger value="risk">Risk</TabsTrigger>
                  <TabsTrigger value="stress">Stress</TabsTrigger>
                  <TabsTrigger value="costs">Costs</TabsTrigger>
                </TabsList>

                <TabsContent value="metrics">
                  <SectionCard title="All Metrics">
                    <BarMetricsChart
                      data={metricsEntries.map(([name, value]) => ({
                        name: name.replace(/_/g, " "),
                        value: value as number,
                      }))}
                      height={300}
                    />
                  </SectionCard>
                </TabsContent>

                <TabsContent value="rolling">
                  <SectionCard title="Rolling Sharpe">
                    <ScatterPlot data={rollingSharpe} xLabel="Observation" yLabel="Rolling Sharpe" referenceY={0} height={300} />
                  </SectionCard>
                </TabsContent>

                <TabsContent value="walkforward">
                  <SectionCard title="Walk-Forward Splits" noPadding>
                    <DataTable columns={walkForwardCols} data={result.walk_forward ?? []} pageSize={10} enableExport exportFilename="analytics-walk-forward.csv" />
                  </SectionCard>
                </TabsContent>

                <TabsContent value="montecarlo">
                  <SectionCard title="Monte Carlo Summary">
                    <div className="overflow-auto rounded-[var(--radius)] bg-[var(--surface-2)] p-3">
                      <code className="block whitespace-pre-wrap text-xs text-[var(--text-muted)]">
                        {JSON.stringify(result.monte_carlo ?? {}, null, 2)}
                      </code>
                    </div>
                  </SectionCard>
                </TabsContent>

                <TabsContent value="attribution">
                  <SectionCard title="Attribution Breakdown">
                    <div className="overflow-auto rounded-[var(--radius)] bg-[var(--surface-2)] p-3">
                      <code className="block whitespace-pre-wrap text-xs text-[var(--text-muted)]">
                        {JSON.stringify(result.attribution ?? {}, null, 2)}
                      </code>
                    </div>
                  </SectionCard>
                </TabsContent>

                <TabsContent value="risk">
                  <SectionCard title="Advanced Risk">
                    <div className="overflow-auto rounded-[var(--radius)] bg-[var(--surface-2)] p-3">
                      <code className="block whitespace-pre-wrap text-xs text-[var(--text-muted)]">
                        {JSON.stringify(result.risk ?? {}, null, 2)}
                      </code>
                    </div>
                  </SectionCard>
                </TabsContent>

                <TabsContent value="stress">
                  <SectionCard title="Stress Scenarios">
                    <div className="overflow-auto rounded-[var(--radius)] bg-[var(--surface-2)] p-3">
                      <code className="block whitespace-pre-wrap text-xs text-[var(--text-muted)]">
                        {JSON.stringify(result.stress ?? [], null, 2)}
                      </code>
                    </div>
                  </SectionCard>
                </TabsContent>

                <TabsContent value="costs">
                  <SectionCard title="Transaction Cost Analysis">
                    <div className="overflow-auto rounded-[var(--radius)] bg-[var(--surface-2)] p-3">
                      <code className="block whitespace-pre-wrap text-xs text-[var(--text-muted)]">
                        {JSON.stringify(result.transaction_costs ?? {}, null, 2)}
                      </code>
                    </div>
                  </SectionCard>
                </TabsContent>
              </Tabs>
              </StaggerItem>
            )}

            {!result && !isSubmitting && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <p className="text-small text-[var(--text-muted)]">Provide an equity curve source and run analytics.</p>
              </SectionCard>
            )}

            {isSubmitting && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <div className="flex flex-col items-center gap-[var(--space-3)]">
                  <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
                  <p className="text-small text-[var(--text-muted)]">Running analytics…</p>
                </div>
              </SectionCard>
            )}
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
