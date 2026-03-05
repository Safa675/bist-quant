"use client";

import * as React from "react";
import { combineFactors, createJob, getFactorCatalog, getFactorDetail } from "@/lib/api";
import { toBacktestUiResult } from "@/lib/adapters";
import { useApi } from "@/hooks/use-api";
import { useJobPolling } from "@/hooks/use-job-polling";
import type { FactorCatalog, SignalInfo } from "@/lib/types";
import { PageHeader } from "@/components/shared/page-header";
import { SectionCard } from "@/components/shared/section-card";
import { ApiError } from "@/components/shared/api-error";
import { DataTable } from "@/components/shared/data-table";
import { KpiCard } from "@/components/shared/kpi-card";
import { StatusBadge } from "@/components/shared/status-badge";
import {
  PageScaffold,
  PageMain,
  PageSidebar,
  PageSectionStack,
} from "@/components/shared/page-scaffold";
import { FormField, FormGrid, FormLabel, FormRow } from "@/components/shared/form-field";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LazyBarMetricsChart as BarMetricsChart,
  LazyEquityCurveChart as EquityCurveChart,
} from "@/components/charts/lazy";
import { KeyValueList } from "@/components/shared/key-value-list";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SelectInput } from "@/components/ui/select-input";
import type { ColumnDef } from "@tanstack/react-table";
import { Combine, FlaskConical, Loader2, Search } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

const catalogColumns: ColumnDef<SignalInfo, unknown>[] = [
  { accessorKey: "name", header: "Signal" },
  {
    accessorKey: "category",
    header: "Category",
    cell: ({ getValue }) => <span className="text-small text-[var(--text-muted)]">{(getValue<string>()) ?? "—"}</span>,
  },
  {
    accessorKey: "description",
    header: "Description",
    cell: ({ getValue }) => (
      <span className="line-clamp-2 text-xs text-[var(--text-muted)]">{(getValue<string>()) ?? "—"}</span>
    ),
  },
];

interface BreakdownRow {
  signal: string;
  category: string;
  weight: number;
  contribution: number | null;
}

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

function toSignalDetails(catalog: FactorCatalog | null): SignalInfo[] {
  if (!catalog) return [];
  if (Array.isArray(catalog.details) && catalog.details.length > 0) return catalog.details;
  return catalog.signals.map((name) => ({ name, category: "Uncategorized", description: "" }));
}

function buildCorrelationRows(raw: unknown): Array<Record<string, string | number>> {
  if (!isRecord(raw)) return [];
  const factors = Object.keys(raw);
  if (!factors.length) return [];

  return factors.map((factor) => {
    const rowRaw = raw[factor];
    const row: Record<string, string | number> = { factor };
    if (isRecord(rowRaw)) {
      for (const col of factors) {
        const v = toNumber(rowRaw[col]);
        row[col] = v === null ? "—" : Number(v.toFixed(3));
      }
    }
    return row;
  });
}

export function FactorLabContent() {
  const [categoryFilter, setCategoryFilter] = React.useState("All");
  const [search, setSearch] = React.useState("");

  const [selectedFactors, setSelectedFactors] = React.useState<string[]>([]);
  const [weights, setWeights] = React.useState<Record<string, number>>({});

  const [selectedSignal, setSelectedSignal] = React.useState("");
  const [combineMethod, setCombineMethod] = React.useState<
    "custom" | "equal" | "risk_parity" | "min_variance"
  >("custom");
  const [startDate, setStartDate] = React.useState("2016-01-01");
  const [endDate, setEndDate] = React.useState("2025-12-31");

  const [combineJobId, setCombineJobId] = React.useState<string | null>(null);
  const [combineError, setCombineError] = React.useState<Error | null>(null);
  const [isSubmittingCombine, setIsSubmittingCombine] = React.useState(false);

  const [quickSignal, setQuickSignal] = React.useState("");
  const [quickJobId, setQuickJobId] = React.useState<string | null>(null);
  const [quickError, setQuickError] = React.useState<Error | null>(null);
  const [isSubmittingQuick, setIsSubmittingQuick] = React.useState(false);

  const catalogFetcher = React.useCallback(() => getFactorCatalog(), []);
  const { data: catalog, isLoading: isLoadingCatalog, error: catalogError } = useApi<FactorCatalog>(catalogFetcher);

  const signalRows = React.useMemo(() => toSignalDetails(catalog ?? null), [catalog]);

  React.useEffect(() => {
    if (!quickSignal && signalRows.length > 0) setQuickSignal(signalRows[0].name);
    if (!selectedSignal && signalRows.length > 0) setSelectedSignal(signalRows[0].name);
  }, [quickSignal, selectedSignal, signalRows]);

  const detailFetcher = React.useCallback(
    () => (selectedSignal ? getFactorDetail(selectedSignal) : Promise.resolve(null)),
    [selectedSignal],
  );
  const { data: selectedDetail } = useApi<Record<string, unknown> | null>(detailFetcher, {
    skip: !selectedSignal,
  });

  React.useEffect(() => {
    setWeights((prev) => {
      const next: Record<string, number> = {};
      for (const factor of selectedFactors) {
        next[factor] = prev[factor] ?? 100 / Math.max(1, selectedFactors.length);
      }
      return next;
    });
  }, [selectedFactors]);

  const filteredRows = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    return signalRows.filter((row) => {
      const rowCategory = (row.category ?? "Uncategorized").trim();
      if (categoryFilter !== "All" && rowCategory !== categoryFilter) return false;
      if (!q) return true;
      return (
        row.name.toLowerCase().includes(q)
        || (row.description ?? "").toLowerCase().includes(q)
        || rowCategory.toLowerCase().includes(q)
      );
    });
  }, [categoryFilter, search, signalRows]);

  const categories = React.useMemo(() => {
    const values = new Set<string>();
    signalRows.forEach((row) => values.add((row.category ?? "Uncategorized").trim() || "Uncategorized"));
    return ["All", ...Array.from(values).sort()];
  }, [signalRows]);

  const normalizedWeights = React.useMemo(() => {
    const entries = selectedFactors.map((name) => ({ name, weight: Math.max(0, weights[name] ?? 0) }));
    const total = entries.reduce((sum, row) => sum + row.weight, 0);
    if (total <= 0) return entries.map((row) => ({ ...row, weight: 0 }));
    return entries.map((row) => ({ ...row, weight: row.weight / total }));
  }, [selectedFactors, weights]);

  const { job: combineJob, isPolling: isCombining } = useJobPolling(combineJobId);
  const { job: quickJob, isPolling: isQuickPolling } = useJobPolling(quickJobId);

  const combinePayload = React.useMemo(() => {
    if (!combineJob || combineJob.status !== "completed" || !combineJob.result) return null;
    const raw = combineJob.result;
    if (!isRecord(raw)) return null;
    const backtestRaw = isRecord(raw.backtest) ? raw.backtest : raw;
    const correlationRaw = isRecord(raw.factor_correlation)
      ? raw.factor_correlation
      : (isRecord(backtestRaw.correlation_matrix) ? backtestRaw.correlation_matrix : null);
    const contributionRaw = isRecord(raw.factor_contribution)
      ? raw.factor_contribution
      : (isRecord(raw.attribution) ? raw.attribution : null);

    return {
      backtest: toBacktestUiResult(backtestRaw),
      correlationRows: buildCorrelationRows(correlationRaw),
      contribution: contributionRaw,
      optimizedWeights: isRecord(raw.optimized_weights) ? raw.optimized_weights : null,
    };
  }, [combineJob]);

  const quickResult = React.useMemo(() => {
    if (!quickJob || quickJob.status !== "completed" || !quickJob.result) return null;
    if (!isRecord(quickJob.result)) return null;
    return toBacktestUiResult(quickJob.result);
  }, [quickJob]);

  const runQuickBacktest = async () => {
    if (!quickSignal) return;
    setQuickError(null);
    setIsSubmittingQuick(true);
    try {
      const created = await createJob("backtest", {
        factor_name: quickSignal,
        start_date: "2019-01-01",
        end_date: "2025-12-31",
        top_n: 20,
        rebalance_frequency: "monthly",
      });
      setQuickJobId(created.id);
    } catch (err) {
      setQuickError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmittingQuick(false);
    }
  };

  const runCombine = async () => {
    if (selectedFactors.length < 2) {
      setCombineError(new Error("Select at least 2 factors to combine."));
      return;
    }
    setCombineError(null);
    setIsSubmittingCombine(true);
    try {
      const signals =
        combineMethod === "equal"
          ? selectedFactors.map((name) => ({ name, weight: 1 / selectedFactors.length }))
          : normalizedWeights.map((row) => ({ name: row.name, weight: row.weight }));

      const created = await combineFactors({
        signals,
        method: combineMethod,
        start_date: startDate,
        end_date: endDate,
      });
      setCombineJobId(created.id);
    } catch (err) {
      setCombineError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsSubmittingCombine(false);
    }
  };

  const correlationColumns = React.useMemo<ColumnDef<Record<string, string | number>, unknown>[]>(() => {
    if (!combinePayload?.correlationRows?.length) {
      return [{ accessorKey: "factor", header: "Factor" }];
    }
    const keys = Object.keys(combinePayload.correlationRows[0] ?? {});
    return keys.map((key) => ({
      accessorKey: key,
      header: key === "factor" ? "Factor" : key,
    }));
  }, [combinePayload]);

  const breakdownRows: BreakdownRow[] = React.useMemo(() => {
    const contribution = combinePayload?.contribution;
    return normalizedWeights.map((row) => ({
      signal: row.name,
      category: signalRows.find((x) => x.name === row.name)?.category ?? "Uncategorized",
      weight: row.weight,
      contribution:
        contribution && isRecord(contribution)
          ? toNumber(contribution[row.name])
          : null,
    }));
  }, [combinePayload, normalizedWeights, signalRows]);

  return (
    <StaggerReveal>
      <StaggerItem><PageHeader
        title="Factor Lab"
        subtitle="Catalog, quick tests, and multi-factor combination"
        actions={
          combineJob ? <StatusBadge status={combineJob.status} /> : undefined
        }
      /></StaggerItem>

      <ApiError error={catalogError} className="mb-6" />

      <PageScaffold>
        <PageMain>
          <PageSectionStack>
            <StaggerItem><SectionCard title="Signal Catalog" subtitle={`${filteredRows.length}/${signalRows.length} signals`} noPadding>
              <div className="border-b border-[var(--border)] p-[var(--space-3)]">
                <FormGrid>
                  <FormRow>
                    <FormField label="Category" htmlFor="factor-lab-category">
                      <SelectInput
                        id="factor-lab-category"
                        value={categoryFilter}
                        onChange={(e) => setCategoryFilter(e.target.value)}
                      >
                        {categories.map((cat) => (
                          <option key={cat} value={cat}>
                            {cat}
                          </option>
                        ))}
                      </SelectInput>
                    </FormField>

                    <FormField label="Search" htmlFor="factor-lab-search">
                      <div className="relative">
                        <Search className="pointer-events-none absolute left-2 top-2.5 h-4 w-4 text-[var(--text-faint)]" />
                        <Input
                          id="factor-lab-search"
                          value={search}
                          onChange={(e) => setSearch(e.target.value)}
                          className="pl-8"
                          placeholder="momentum, value, quality..."
                        />
                      </div>
                    </FormField>
                  </FormRow>
                </FormGrid>
              </div>

              {isLoadingCatalog ? (
                <div className="flex items-center justify-center py-[var(--space-8)]">
                  <Loader2 className="h-6 w-6 animate-spin text-[var(--accent)]" />
                </div>
              ) : (
                <DataTable
                  columns={catalogColumns}
                  data={filteredRows}
                  pageSize={12}
                  searchColumn="name"
                  searchPlaceholder="Search signal..."
                  enableRowSelection
                  onRowSelectionChange={(rows) => setSelectedFactors(rows.map((row) => row.name))}
                />
              )}
            </SectionCard></StaggerItem>

            {combinePayload && (
              <>
                <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-4">
                  <KpiCard title="CAGR" value={combinePayload.backtest.metrics.cagr ?? null} decimals={3} />
                  <KpiCard title="Sharpe" value={combinePayload.backtest.metrics.sharpe ?? null} decimals={3} />
                  <KpiCard title="Sortino" value={combinePayload.backtest.metrics.sortino ?? null} decimals={3} />
                  <KpiCard title="Max DD" value={combinePayload.backtest.metrics.max_drawdown ?? null} decimals={3} />
                </div>

                <Tabs defaultValue="curve">
                  <TabsList className="w-full overflow-x-auto">
                    <TabsTrigger value="curve">Curve</TabsTrigger>
                    <TabsTrigger value="breakdown">Breakdown</TabsTrigger>
                    <TabsTrigger value="correlation">Correlation</TabsTrigger>
                  </TabsList>

                  <TabsContent value="curve">
                    <SectionCard title="Combined Equity Curve">
                      <EquityCurveChart data={combinePayload.backtest.equity_curve} height={360} />
                    </SectionCard>
                  </TabsContent>

                  <TabsContent value="breakdown">
                    <SectionCard title="Factor Breakdown" noPadding>
                      <DataTable
                        columns={[
                          { accessorKey: "signal", header: "Signal" },
                          { accessorKey: "category", header: "Category" },
                          {
                            accessorKey: "weight",
                            header: "Weight",
                            cell: ({ getValue }) => `${(((getValue<number>() ?? 0) as number) * 100).toFixed(1)}%`,
                          },
                          {
                            accessorKey: "contribution",
                            header: "Contribution",
                            cell: ({ getValue }) => {
                              const value = getValue<number | null>();
                              return value === null || value === undefined ? "—" : value.toFixed(3);
                            },
                          },
                        ]}
                        data={breakdownRows}
                        pageSize={10}
                      />
                    </SectionCard>

                    {combinePayload.contribution && (
                      <SectionCard title="Contribution Bars">
                        <BarMetricsChart
                          data={Object.entries(combinePayload.contribution)
                            .map(([name, value]) => ({ name, value: toNumber(value) ?? 0 }))
                            .slice(0, 12)}
                          height={260}
                        />
                      </SectionCard>
                    )}
                  </TabsContent>

                  <TabsContent value="correlation">
                    <SectionCard title="Correlation Matrix" noPadding>
                      <DataTable
                        columns={correlationColumns}
                        data={combinePayload.correlationRows}
                        pageSize={10}
                        enableExport
                        exportFilename="factor-correlation.csv"
                      />
                    </SectionCard>
                  </TabsContent>
                </Tabs>
              </>
            )}

            {!combinePayload && !isCombining && (
              <SectionCard className="flex items-center justify-center py-[var(--space-8)]">
                <p className="text-small text-[var(--text-muted)]">
                  Select factors, assign weights, and run a combined backtest to populate tabs.
                </p>
              </SectionCard>
            )}
          </PageSectionStack>
        </PageMain>

        <PageSidebar>
          <PageSectionStack>
            <StaggerItem><SectionCard title="Signal Detail">
              <FormGrid>
                <FormField label="Signal" htmlFor="factor-lab-detail-signal">
                  <SelectInput
                    id="factor-lab-detail-signal"
                    value={selectedSignal}
                    onChange={(e) => setSelectedSignal(e.target.value)}
                  >
                    {signalRows.map((row) => (
                      <option key={row.name} value={row.name}>
                        {row.name}
                      </option>
                    ))}
                  </SelectInput>
                </FormField>

                {selectedDetail && <KeyValueList data={selectedDetail} />}
              </FormGrid>
            </SectionCard></StaggerItem>

            <StaggerItem><SectionCard title="Quick Backtest">
              <FormGrid>
                <FormField label="Signal" htmlFor="factor-lab-quick-signal">
                  <SelectInput
                    id="factor-lab-quick-signal"
                    value={quickSignal}
                    onChange={(e) => setQuickSignal(e.target.value)}
                  >
                    {signalRows.map((row) => (
                      <option key={row.name} value={row.name}>
                        {row.name}
                      </option>
                    ))}
                  </SelectInput>
                </FormField>

                <Button data-testid="factor-lab-quick-run" className="w-full" onClick={runQuickBacktest} disabled={isSubmittingQuick || isQuickPolling || !quickSignal}>
                  {isSubmittingQuick || isQuickPolling ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <FlaskConical className="mr-2 h-4 w-4" />
                  )}
                  {isQuickPolling ? "Running..." : "Run Quick Backtest"}
                </Button>

                <ApiError error={quickError} />
                {quickJob && <StatusBadge status={quickJob.status} />}

                {quickResult && (
                  <div className="grid grid-cols-2 gap-2">
                    <KpiCard title="CAGR" value={quickResult.metrics.cagr ?? null} decimals={3} animate={false} />
                    <KpiCard title="Sharpe" value={quickResult.metrics.sharpe ?? null} decimals={3} animate={false} />
                    <KpiCard title="Max DD" value={quickResult.metrics.max_drawdown ?? null} decimals={3} animate={false} />
                    <KpiCard title="Sortino" value={quickResult.metrics.sortino ?? null} decimals={3} animate={false} />
                  </div>
                )}
              </FormGrid>
            </SectionCard></StaggerItem>

            <StaggerItem><SectionCard title="Combine Controls">
              <FormGrid>
                <FormLabel>Selected Factors ({selectedFactors.length})</FormLabel>

                {selectedFactors.length > 0 ? (
                  <div className="space-y-2">
                    {selectedFactors.map((factor) => (
                      <div key={factor} className="rounded-[var(--radius-sm)] border border-[var(--border)] p-2">
                        <div className="mb-1 text-xs font-medium text-[var(--text-muted)]">{factor}</div>
                        <Input
                          type="number"
                          min={0}
                          max={100}
                          step={1}
                          value={Number((weights[factor] ?? 0).toFixed(2))}
                          onChange={(e) =>
                            setWeights((prev) => ({
                              ...prev,
                              [factor]: Number(e.target.value),
                            }))
                          }
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-small text-[var(--text-muted)]">Select rows from the catalog table.</p>
                )}

                <Button
                  variant="outline"
                  onClick={() => {
                    if (selectedFactors.length === 0) return;
                    const eq = 100 / selectedFactors.length;
                    const next: Record<string, number> = {};
                    selectedFactors.forEach((factor) => {
                      next[factor] = eq;
                    });
                    setWeights(next);
                  }}
                  disabled={selectedFactors.length === 0}
                >
                  Equal Weights
                </Button>

                <FormRow>
                  <FormField label="Method" htmlFor="factor-lab-combine-method">
                    <SelectInput
                      id="factor-lab-combine-method"
                      value={combineMethod}
                      onChange={(e) =>
                        setCombineMethod(
                          e.target.value as "custom" | "equal" | "risk_parity" | "min_variance",
                        )
                      }
                    >
                      <option value="custom">Custom</option>
                      <option value="equal">Equal</option>
                      <option value="risk_parity">Risk Parity</option>
                      <option value="min_variance">Min Variance</option>
                    </SelectInput>
                  </FormField>

                  <FormField label="Start" htmlFor="factor-lab-start">
                    <Input id="factor-lab-start" type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
                  </FormField>
                </FormRow>

                <FormField label="End" htmlFor="factor-lab-end">
                  <Input id="factor-lab-end" type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
                </FormField>

                <Button
                  data-testid="factor-lab-combine-run"
                  className="w-full"
                  onClick={runCombine}
                  disabled={selectedFactors.length < 2 || isSubmittingCombine || isCombining}
                >
                  {isSubmittingCombine || isCombining ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Combine className="mr-2 h-4 w-4" />
                  )}
                  {isCombining ? "Combining..." : "Run Combined Backtest"}
                </Button>

                <ApiError error={combineError} />
                {combineJob && <StatusBadge status={combineJob.status} />}

                {combinePayload?.optimizedWeights && (
                  <SectionCard title="Optimized Weights" className="border-dashed">
                    <KeyValueList data={combinePayload.optimizedWeights} />
                  </SectionCard>
                )}
              </FormGrid>
            </SectionCard></StaggerItem>
          </PageSectionStack>
        </PageSidebar>
      </PageScaffold>
    </StaggerReveal>
  );
}
