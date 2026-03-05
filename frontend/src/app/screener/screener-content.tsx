"use client";

import * as React from "react";
import { getScreenerMetadata, getSparklines, runScreener } from "@/lib/api";
import type { ScreenerRow, ScreenerResult } from "@/lib/types";
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
  LazySparklineGrid as SparklineGrid,
} from "@/components/charts/lazy";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SelectInput } from "@/components/ui/select-input";
import { Checkbox } from "@/components/ui/checkbox";
import type { ColumnDef } from "@tanstack/react-table";
import { Filter, Loader2, Sparkles } from "lucide-react";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";

interface MetadataResponse {
  indexes?: string[];
  templates?: string[];
  recommendations?: string[];
  [key: string]: unknown;
}

function toPercentDisplay(value: number): number {
  return Math.abs(value) <= 1 ? value * 100 : value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

const screenerColumns: ColumnDef<ScreenerRow, unknown>[] = [
  { accessorKey: "symbol", header: "Symbol" },
  { accessorKey: "sector", header: "Sector" },
  {
    accessorKey: "recommendation",
    header: "Rec",
    cell: ({ getValue }) => (getValue<string | null>() ?? "—"),
  },
  {
    accessorKey: "pe",
    header: "P/E",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      return v === null || v === undefined ? "—" : v.toFixed(1);
    },
  },
  {
    accessorKey: "roe",
    header: "ROE",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      return v === null || v === undefined ? "—" : `${v.toFixed(1)}%`;
    },
  },
  {
    accessorKey: "rsi_14",
    header: "RSI",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      if (v === null || v === undefined) return "—";
      const color = v > 70 ? "var(--bear)" : v < 30 ? "var(--bull)" : "var(--text-muted)";
      return <span style={{ color }}>{v.toFixed(1)}</span>;
    },
  },
  {
    accessorKey: "return_1m",
    header: "1M Ret",
    cell: ({ getValue }) => {
      const raw = getValue<number | null>();
      if (raw === null || raw === undefined) return "—";
      const pct = toPercentDisplay(raw);
      return <span className={pct >= 0 ? "text-[var(--bull)]" : "text-[var(--bear)]"}>{pct.toFixed(1)}%</span>;
    },
  },
  {
    accessorKey: "upside_potential",
    header: "Upside",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      if (v === null || v === undefined) return "—";
      return `${v.toFixed(1)}%`;
    },
  },
];

const groupColumns: Record<string, string[]> = {
  valuation: ["symbol", "sector", "recommendation", "market_cap_usd", "pe", "forward_pe", "pb", "ev_ebitda", "ev_sales", "dividend_yield"],
  quality: ["symbol", "sector", "roe", "roa", "net_margin", "ebitda_margin", "revenue_growth_yoy", "net_income_growth_yoy"],
  momentum: ["symbol", "sector", "rsi_14", "macd_hist", "atr_14_pct", "return_1w", "return_1m", "return_ytd", "return_1y", "upside_potential"],
  flow: ["symbol", "sector", "foreign_ratio", "foreign_change_1w", "foreign_change_1m", "float_ratio", "volume_3m", "volume_12m"],
};

export function ScreenerContent() {
  const [metadata, setMetadata] = React.useState<MetadataResponse | null>(null);

  const [index, setIndex] = React.useState("XU100");
  const [template, setTemplate] = React.useState("");
  const [selectedSectorsInput, setSelectedSectorsInput] = React.useState("");
  const [recommendationInput, setRecommendationInput] = React.useState("");

  const [minPe, setMinPe] = React.useState("");
  const [maxPe, setMaxPe] = React.useState("");
  const [minRoe, setMinRoe] = React.useState("");
  const [maxRoe, setMaxRoe] = React.useState("");
  const [minNetMargin, setMinNetMargin] = React.useState("");
  const [maxNetMargin, setMaxNetMargin] = React.useState("");
  const [minMcap, setMinMcap] = React.useState("");
  const [maxMcap, setMaxMcap] = React.useState("");
  const [minDivYield, setMinDivYield] = React.useState("");

  const [minRsi, setMinRsi] = React.useState("");
  const [maxRsi, setMaxRsi] = React.useState("");
  const [minRet1m, setMinRet1m] = React.useState("");
  const [maxRet1m, setMaxRet1m] = React.useState("");

  const [sortBy, setSortBy] = React.useState("upside_potential");
  const [sortAsc, setSortAsc] = React.useState(false);
  const [topN, setTopN] = React.useState(150);
  const [showSparklines, setShowSparklines] = React.useState(true);
  const [sparkCount, setSparkCount] = React.useState(16);

  const [result, setResult] = React.useState<ScreenerResult | null>(null);
  const [rowsView, setRowsView] = React.useState<ScreenerRow[]>([]);
  const [appliedFilters, setAppliedFilters] = React.useState<string[]>([]);
  const [watchlist, setWatchlist] = React.useState<string[]>([]);
  const [sparklines, setSparklines] = React.useState<Record<string, number[]>>({});

  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    const loadMeta = async () => {
      try {
        const meta = (await getScreenerMetadata()) as MetadataResponse;
        if (cancelled) return;
        setMetadata(meta);
        if (Array.isArray(meta.indexes) && meta.indexes.length > 0) {
          setIndex(meta.indexes[0]);
        }
      } catch {
        // metadata is optional for rendering
      }
    };

    void loadMeta();
    return () => {
      cancelled = true;
    };
  }, []);

  const parsedSectors = React.useMemo(
    () =>
      selectedSectorsInput
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    [selectedSectorsInput],
  );

  const parsedRecommendations = React.useMemo(
    () =>
      recommendationInput
        .split(",")
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean),
    [recommendationInput],
  );

  const handleRun = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const filters: Record<string, { min?: number; max?: number }> = {};
      const setRange = (key: string, minRaw: string, maxRaw: string) => {
        const minVal = minRaw.trim() ? Number(minRaw) : null;
        const maxVal = maxRaw.trim() ? Number(maxRaw) : null;
        const range: { min?: number; max?: number } = {};
        if (minVal !== null && Number.isFinite(minVal)) range.min = minVal;
        if (maxVal !== null && Number.isFinite(maxVal)) range.max = maxVal;
        if (Object.keys(range).length > 0) filters[key] = range;
      };

      setRange("pe", minPe, maxPe);
      setRange("roe", minRoe, maxRoe);
      setRange("net_margin", minNetMargin, maxNetMargin);
      setRange("market_cap_usd", minMcap, maxMcap);
      setRange("dividend_yield", minDivYield, "");
      setRange("rsi_14", minRsi, maxRsi);
      setRange("return_1m", minRet1m, maxRet1m);

      const request = {
        index,
        template: template || undefined,
        sort_by: sortBy,
        sort_asc: sortAsc,
        top_n: topN,
        filters,
      };

      const res = await runScreener(request as never);
      setResult(res);

      const chips: string[] = [];
      if (template) chips.push(`Template: ${template}`);
      chips.push(`Index: ${index}`);
      parsedSectors.forEach((s) => chips.push(`Sector: ${s}`));
      parsedRecommendations.forEach((r) => chips.push(`Rec: ${r}`));
      Object.entries(filters).forEach(([key, range]) => {
        const parts = [key];
        if (range.min !== undefined) parts.push(`>= ${range.min}`);
        if (range.max !== undefined) parts.push(`<= ${range.max}`);
        chips.push(parts.join(" "));
      });
      setAppliedFilters(chips);

      let viewRows = [...res.rows];
      if (parsedSectors.length > 0) {
        const normalized = parsedSectors.map((s) => s.toLowerCase());
        viewRows = viewRows.filter((row) => normalized.includes(String(row.sector ?? "").toLowerCase()));
      }
      if (parsedRecommendations.length > 0) {
        viewRows = viewRows.filter((row) => parsedRecommendations.includes(String(row.recommendation ?? "").toUpperCase()));
      }
      setRowsView(viewRows);

      if (showSparklines && viewRows.length > 0) {
        const symbols = viewRows.slice(0, sparkCount).map((row) => row.symbol);
        const sparkMap = await getSparklines(symbols);
        setSparklines(sparkMap);
      } else {
        setSparklines({});
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  };

  const sparkItems = React.useMemo(() => {
    const visibleRows = rowsView.slice(0, sparkCount);
    return visibleRows
      .map((row) => {
        const series = sparklines[row.symbol];
        if (!series || series.length < 5) return null;
        return {
          name: row.symbol,
          data: series.map((value, i) => ({ date: String(i), value })),
          suffix: "",
        };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null);
  }, [rowsView, sparkCount, sparklines]);

  const recCounts = React.useMemo(() => {
    const counts = { AL: 0, TUT: 0, SAT: 0, NA: 0 };
    rowsView.forEach((row) => {
      const rec = String(row.recommendation ?? "").toUpperCase();
      if (rec === "AL") counts.AL += 1;
      else if (rec === "TUT") counts.TUT += 1;
      else if (rec === "SAT") counts.SAT += 1;
      else counts.NA += 1;
    });
    return counts;
  }, [rowsView]);

  const sectorBars = React.useMemo(() => {
    const map = new Map<string, number>();
    rowsView.forEach((row) => {
      const sector = String(row.sector ?? "Unknown");
      map.set(sector, (map.get(sector) ?? 0) + 1);
    });
    return Array.from(map.entries())
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 12);
  }, [rowsView]);

  const topRoe = React.useMemo(() => {
    return [...rowsView]
      .filter((row) => asNumber(row.roe) !== null)
      .sort((a, b) => (asNumber(b.roe) ?? -Infinity) - (asNumber(a.roe) ?? -Infinity))
      .slice(0, 10);
  }, [rowsView]);

  const lowPe = React.useMemo(() => {
    return [...rowsView]
      .filter((row) => {
        const pe = asNumber(row.pe);
        return pe !== null && pe > 0;
      })
      .sort((a, b) => (asNumber(a.pe) ?? Infinity) - (asNumber(b.pe) ?? Infinity))
      .slice(0, 10);
  }, [rowsView]);

  const groupData = React.useMemo(() => {
    const tableRows: Record<string, unknown>[] = rowsView.map((row) =>
      isRecord(row) ? row : {},
    );

    const pickCols = (cols: string[]) =>
      tableRows.map((row) => {
        const out: Record<string, unknown> = {};
        cols.forEach((col) => {
          out[col] = row[col] ?? null;
        });
        return out;
      });

    return {
      valuation: pickCols(groupColumns.valuation),
      quality: pickCols(groupColumns.quality),
      momentum: pickCols(groupColumns.momentum),
      flow: pickCols(groupColumns.flow),
    };
  }, [rowsView]);

  return (
    <StaggerReveal>
      <StaggerItem><PageHeader title="Screener" subtitle="Advanced filtering, ranking, and monitoring for BIST equities" /></StaggerItem>

      <PageScaffold>
        <PageSidebar>
          <StaggerItem><SectionCard title="Screener Filters">
            <FormGrid>
              <FormRow>
                <FormField label="Universe" htmlFor="screener-index">
                  <SelectInput id="screener-index" value={index} onChange={(e) => setIndex(e.target.value)}>
                    {(metadata?.indexes ?? ["XU100", "XU050", "XU030", "XUTUM"]).map((idx) => (
                      <option key={idx} value={idx}>
                        {idx}
                      </option>
                    ))}
                  </SelectInput>
                </FormField>

                <FormField label="Preset" htmlFor="screener-template">
                  <SelectInput id="screener-template" value={template} onChange={(e) => setTemplate(e.target.value)}>
                    <option value="">Custom</option>
                    {(metadata?.templates ?? []).map((tpl) => (
                      <option key={tpl} value={tpl}>
                        {tpl}
                      </option>
                    ))}
                  </SelectInput>
                </FormField>
              </FormRow>

              <FormField label="Sectors (comma-separated)" htmlFor="screener-sectors">
                <Input
                  id="screener-sectors"
                  value={selectedSectorsInput}
                  onChange={(e) => setSelectedSectorsInput(e.target.value)}
                  placeholder="IMALAT, TEKNOLOJI"
                />
              </FormField>

              <FormField label="Recommendations (comma-separated)" htmlFor="screener-recommendations">
                <Input
                  id="screener-recommendations"
                  value={recommendationInput}
                  onChange={(e) => setRecommendationInput(e.target.value)}
                  placeholder="AL, TUT, SAT"
                />
              </FormField>

              <FormLabel>Valuation</FormLabel>
              <FormRow>
                <Input aria-label="Min P/E" placeholder="Min P/E" type="number" value={minPe} onChange={(e) => setMinPe(e.target.value)} />
                <Input aria-label="Max P/E" placeholder="Max P/E" type="number" value={maxPe} onChange={(e) => setMaxPe(e.target.value)} />
              </FormRow>
              <FormRow>
                <Input aria-label="Min Market Cap" placeholder="Min MCap USD mn" type="number" value={minMcap} onChange={(e) => setMinMcap(e.target.value)} />
                <Input aria-label="Max Market Cap" placeholder="Max MCap USD mn" type="number" value={maxMcap} onChange={(e) => setMaxMcap(e.target.value)} />
              </FormRow>
              <Input aria-label="Min Dividend Yield" placeholder="Min Dividend Yield" type="number" value={minDivYield} onChange={(e) => setMinDivYield(e.target.value)} />

              <FormLabel>Quality</FormLabel>
              <FormRow>
                <Input aria-label="Min ROE" placeholder="Min ROE" type="number" value={minRoe} onChange={(e) => setMinRoe(e.target.value)} />
                <Input aria-label="Max ROE" placeholder="Max ROE" type="number" value={maxRoe} onChange={(e) => setMaxRoe(e.target.value)} />
              </FormRow>
              <FormRow>
                <Input aria-label="Min Net Margin" placeholder="Min Net Margin" type="number" value={minNetMargin} onChange={(e) => setMinNetMargin(e.target.value)} />
                <Input aria-label="Max Net Margin" placeholder="Max Net Margin" type="number" value={maxNetMargin} onChange={(e) => setMaxNetMargin(e.target.value)} />
              </FormRow>

              <FormLabel>Technical / Momentum</FormLabel>
              <FormRow>
                <Input aria-label="Min RSI" placeholder="Min RSI" type="number" value={minRsi} onChange={(e) => setMinRsi(e.target.value)} />
                <Input aria-label="Max RSI" placeholder="Max RSI" type="number" value={maxRsi} onChange={(e) => setMaxRsi(e.target.value)} />
              </FormRow>
              <FormRow>
                <Input aria-label="Min Return 1M" placeholder="Min Return 1M" type="number" value={minRet1m} onChange={(e) => setMinRet1m(e.target.value)} />
                <Input aria-label="Max Return 1M" placeholder="Max Return 1M" type="number" value={maxRet1m} onChange={(e) => setMaxRet1m(e.target.value)} />
              </FormRow>

              <FormRow>
                <FormField label="Sort By" htmlFor="screener-sort-by">
                  <SelectInput id="screener-sort-by" value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                    <option value="upside_potential">Upside Potential</option>
                    <option value="market_cap_usd">Market Cap</option>
                    <option value="pe">P/E</option>
                    <option value="pb">P/B</option>
                    <option value="roe">ROE</option>
                    <option value="net_margin">Net Margin</option>
                    <option value="dividend_yield">Dividend Yield</option>
                    <option value="return_1m">Return 1M</option>
                    <option value="return_1y">Return 1Y</option>
                    <option value="rsi_14">RSI 14</option>
                    <option value="volume_3m">Volume 3M</option>
                  </SelectInput>
                </FormField>

                <FormField label="Top N" htmlFor="screener-top-n">
                  <Input id="screener-top-n" type="number" min={20} max={2000} value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
                </FormField>
              </FormRow>

              <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                <Checkbox checked={sortAsc} onChange={(e) => setSortAsc(e.target.checked)} />
                Sort ascending
              </label>
              <label className="flex items-center gap-2 text-small text-[var(--text-muted)]">
                <Checkbox checked={showSparklines} onChange={(e) => setShowSparklines(e.target.checked)} />
                Show sparklines
              </label>

              <FormField label="Sparkline Count" htmlFor="screener-spark-count">
                <Input id="screener-spark-count" type="number" min={4} max={40} value={sparkCount} onChange={(e) => setSparkCount(Number(e.target.value))} />
              </FormField>

              <Button data-testid="screener-run" className="w-full" onClick={handleRun} disabled={isLoading}>
                {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Filter className="mr-2 h-4 w-4" />}
                {isLoading ? "Running..." : "Run Screener"}
              </Button>
            </FormGrid>
          </SectionCard></StaggerItem>
        </PageSidebar>

        <PageMain>
          <PageSectionStack>
            <ApiError error={error} />

            {result && (
              <>
                <div className="grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-5">
                  <KpiCard title="Engine Matches" value={result.count} decimals={0} animate={false} />
                  <KpiCard title="After Filters" value={rowsView.length} decimals={0} animate={false} />
                  <KpiCard title="AL" value={recCounts.AL} decimals={0} animate={false} />
                  <KpiCard title="TUT" value={recCounts.TUT} decimals={0} animate={false} />
                  <KpiCard title="SAT" value={recCounts.SAT} decimals={0} animate={false} />
                </div>

                {appliedFilters.length > 0 && (
                  <SectionCard title="Applied Filters">
                    <div className="flex flex-wrap gap-2">
                      {appliedFilters.map((chip) => (
                        <span
                          key={chip}
                          className="inline-flex items-center rounded-full border border-[var(--border)] bg-[var(--surface-2)] px-3 py-1 text-xs text-[var(--text-muted)]"
                        >
                          {chip}
                        </span>
                      ))}
                    </div>
                  </SectionCard>
                )}

                <SectionCard title="Results" noPadding>
                  <DataTable
                    columns={screenerColumns}
                    data={rowsView}
                    pageSize={25}
                    searchColumn="symbol"
                    searchPlaceholder="Search ticker..."
                    enableExport
                    exportFilename="screener-results.csv"
                  />
                </SectionCard>

                <SectionCard title={`Watchlist (${watchlist.length})`}>
                  <div className="flex flex-wrap gap-2">
                    {rowsView.slice(0, 12).map((row) => {
                      const inWatchlist = watchlist.includes(row.symbol);
                      return (
                        <Button
                          key={row.symbol}
                          size="sm"
                          variant={inWatchlist ? "default" : "outline"}
                          onClick={() => {
                            setWatchlist((prev) =>
                              inWatchlist
                                ? prev.filter((sym) => sym !== row.symbol)
                                : [...prev, row.symbol],
                            );
                          }}
                        >
                          {inWatchlist ? "✓" : "+"} {row.symbol}
                        </Button>
                      );
                    })}
                  </div>

                  {watchlist.length > 0 && (
                    <div className="mt-3 flex items-center gap-2">
                      <p className="text-small text-[var(--text-muted)]">{watchlist.join(", ")}</p>
                      <Button variant="outline" size="sm" onClick={() => setWatchlist([])}>
                        Clear
                      </Button>
                    </div>
                  )}
                </SectionCard>

                {showSparklines && sparkItems.length > 0 && (
                  <SectionCard title="Sparklines (30-day trend)">
                    <SparklineGrid items={sparkItems} />
                  </SectionCard>
                )}

                <Tabs defaultValue="sector">
                  <TabsList className="w-full overflow-x-auto">
                    <TabsTrigger value="sector">Sector Distribution</TabsTrigger>
                    <TabsTrigger value="topbottom">Top vs Bottom</TabsTrigger>
                    <TabsTrigger value="groups">Full Column Groups</TabsTrigger>
                  </TabsList>

                  <TabsContent value="sector">
                    <SectionCard title="Sector Distribution">
                      <BarMetricsChart data={sectorBars} height={300} />
                    </SectionCard>
                  </TabsContent>

                  <TabsContent value="topbottom">
                    <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-2">
                      <SectionCard title="Top 10 by ROE" noPadding>
                        <DataTable
                          columns={[
                            { accessorKey: "symbol", header: "Symbol" },
                            { accessorKey: "sector", header: "Sector" },
                            { accessorKey: "roe", header: "ROE" },
                            { accessorKey: "net_margin", header: "Net Margin" },
                          ]}
                          data={topRoe}
                          pageSize={10}
                        />
                      </SectionCard>

                      <SectionCard title="Bottom 10 by P/E" noPadding>
                        <DataTable
                          columns={[
                            { accessorKey: "symbol", header: "Symbol" },
                            { accessorKey: "sector", header: "Sector" },
                            { accessorKey: "pe", header: "P/E" },
                            { accessorKey: "roe", header: "ROE" },
                          ]}
                          data={lowPe}
                          pageSize={10}
                        />
                      </SectionCard>
                    </div>
                  </TabsContent>

                  <TabsContent value="groups">
                    <Tabs defaultValue="valuation">
                      <TabsList className="w-full overflow-x-auto">
                        <TabsTrigger value="valuation">Valuation</TabsTrigger>
                        <TabsTrigger value="quality">Quality</TabsTrigger>
                        <TabsTrigger value="momentum">Momentum</TabsTrigger>
                        <TabsTrigger value="flow">Flow</TabsTrigger>
                      </TabsList>

                      {(["valuation", "quality", "momentum", "flow"] as const).map((groupKey) => {
                        const cols = groupColumns[groupKey];
                        const rows = groupData[groupKey];

                        const columns: ColumnDef<Record<string, unknown>, unknown>[] = cols.map((col) => ({
                          accessorKey: col,
                          header: col,
                        }));

                        return (
                          <TabsContent key={groupKey} value={groupKey}>
                            <SectionCard title={`${groupKey[0].toUpperCase()}${groupKey.slice(1)} Fields`} noPadding>
                              <DataTable
                                columns={columns}
                                data={rows}
                                pageSize={12}
                                enableExport
                                exportFilename={`screener-${groupKey}.csv`}
                              />
                            </SectionCard>
                          </TabsContent>
                        );
                      })}
                    </Tabs>
                  </TabsContent>
                </Tabs>
              </>
            )}

            {!result && !isLoading && (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                <div className="text-center">
                  <Sparkles className="mx-auto mb-2 h-6 w-6 text-[var(--accent)]" />
                  <p className="text-small text-[var(--text-muted)]">Set advanced filters and run the screener.</p>
                </div>
              </SectionCard>
            )}
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
