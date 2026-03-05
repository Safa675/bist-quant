"use client";

import * as React from "react";
import { useDashboard } from "@/hooks/use-dashboard";
import { PageHeader } from "@/components/shared/page-header";
import { LookbackNav } from "@/components/shared/lookback-nav";
import { KpiCard } from "@/components/shared/kpi-card";
import { SectionCard } from "@/components/shared/section-card";
import { RegimeBadge } from "@/components/shared/regime-badge";
import { ApiError } from "@/components/shared/api-error";
import { PageScaffold, PageMain, PageSectionStack } from "@/components/shared/page-scaffold";
import { StaggerReveal, StaggerItem } from "@/components/shared/stagger-reveal";
import {
  LazyRegimeTimelineChart as RegimeTimelineChart,
  LazyRegimeDonutChart as RegimeDonutChart,
  LazyMacroLineChart as MacroLineChart,
} from "@/components/charts/lazy";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/shared/data-table";
import type { ColumnDef } from "@tanstack/react-table";
import type { MacroChange, MacroPoint } from "@/lib/types";
import { BarChart3, DollarSign, Gem, Loader2 } from "lucide-react";

const macroColumns: ColumnDef<MacroChange, unknown>[] = [
  { accessorKey: "asset", header: "Asset" },
  {
    accessorKey: "current",
    header: "Current",
    cell: ({ getValue }) => (getValue<number>()).toLocaleString("en-US", { maximumFractionDigits: 2 }),
  },
  {
    accessorKey: "d1_pct",
    header: "1D %",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      if (v === null || v === undefined) return "—";
      return <span className={v >= 0 ? "text-[var(--bull)]" : "text-[var(--bear)]"}>{v > 0 ? "+" : ""}{v.toFixed(2)}%</span>;
    },
  },
  {
    accessorKey: "w1_pct",
    header: "1W %",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      if (v === null || v === undefined) return "—";
      return <span className={v >= 0 ? "text-[var(--bull)]" : "text-[var(--bear)]"}>{v > 0 ? "+" : ""}{v.toFixed(2)}%</span>;
    },
  },
  {
    accessorKey: "m1_pct",
    header: "1M %",
    cell: ({ getValue }) => {
      const v = getValue<number | null>();
      if (v === null || v === undefined) return "—";
      return <span className={v >= 0 ? "text-[var(--bull)]" : "text-[var(--bear)]"}>{v > 0 ? "+" : ""}{v.toFixed(2)}%</span>;
    },
  },
];

interface DetailRow {
  asset: string;
  current: number;
  d1_pct: number | null;
  w1_pct: number | null;
  m1_pct: number | null;
}

function pctChange(points: MacroPoint[], periods: number): number | null {
  if (points.length <= periods) return null;
  const curr = points[points.length - 1]?.value;
  const prev = points[points.length - 1 - periods]?.value;
  if (!Number.isFinite(curr) || !Number.isFinite(prev) || prev === 0) return null;
  return ((curr / prev) - 1) * 100;
}

function lastValue(points: MacroPoint[]): number | null {
  if (points.length === 0) return null;
  const v = points[points.length - 1]?.value;
  return Number.isFinite(v) ? v : null;
}

export function DashboardContent() {
  const { data, isLoading, error, lookback, setLookback } = useDashboard();
  const [macroDetailWindow, setMacroDetailWindow] = React.useState<90 | 126 | 252 | 1260>(252);

  const macroSlices = React.useMemo(() => {
    const usd = data?.macro.series.usdtry ?? [];
    const gold = data?.macro.series.xau_try ?? [];
    return {
      usd: usd.slice(-macroDetailWindow),
      gold: gold.slice(-macroDetailWindow),
    };
  }, [data?.macro.series.usdtry, data?.macro.series.xau_try, macroDetailWindow]);

  const macroDetailRows: DetailRow[] = React.useMemo(() => {
    if (!data) return [];

    const rows: DetailRow[] = [];
    const usd = macroSlices.usd;
    const gold = macroSlices.gold;

    const usdCurrent = lastValue(usd);
    if (usdCurrent !== null) {
      rows.push({
        asset: "USD/TRY",
        current: usdCurrent,
        d1_pct: pctChange(usd, 1),
        w1_pct: pctChange(usd, 5),
        m1_pct: pctChange(usd, 21),
      });
    }

    const goldCurrent = lastValue(gold);
    if (goldCurrent !== null) {
      rows.push({
        asset: "Gold (TRY/oz)",
        current: goldCurrent,
        d1_pct: pctChange(gold, 1),
        w1_pct: pctChange(gold, 5),
        m1_pct: pctChange(gold, 21),
      });
    }

    const xu = (data.timeline ?? [])
      .slice(-macroDetailWindow)
      .map((row) => ({ date: row.date, value: row.close }));
    const xuCurrent = lastValue(xu);
    if (xuCurrent !== null) {
      rows.push({
        asset: "XU100",
        current: xuCurrent,
        d1_pct: pctChange(xu, 1),
        w1_pct: pctChange(xu, 5),
        m1_pct: pctChange(xu, 21),
      });
    }

    return rows;
  }, [data, macroDetailWindow, macroSlices.gold, macroSlices.usd]);

  return (
    <StaggerReveal>
      <StaggerItem>
        <PageHeader
          title="Dashboard"
          subtitle="Market overview, regime monitor, and macro detail"
          actions={<LookbackNav value={lookback} onChange={setLookback} />}
        />
      </StaggerItem>

      <ApiError error={error} />

      <PageScaffold>
        <PageMain className="lg:col-span-12 xl:col-span-12">
          <PageSectionStack>
            {data ? (
              <>
                <StaggerItem>
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] sm:grid-cols-2 lg:grid-cols-3">
                  <KpiCard
                    title="XU100"
                    value={data.kpi.xu100_last}
                    change={data.kpi.xu100_daily_pct}
                    decimals={0}
                    icon={<BarChart3 className="h-4 w-4" />}
                  />
                  <KpiCard
                    title="USD/TRY"
                    value={data.kpi.usdtry_last}
                    change={data.kpi.usdtry_daily_pct}
                    decimals={4}
                    icon={<DollarSign className="h-4 w-4" />}
                  />
                  <KpiCard
                    title="Gold/TRY"
                    value={data.kpi.xau_try_last}
                    change={data.kpi.xau_try_daily_pct}
                    decimals={0}
                    icon={<Gem className="h-4 w-4" />}
                  />
                </div>
                </StaggerItem>

                <StaggerItem>
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-3">
                  <SectionCard
                    title="Regime Timeline"
                    actions={<RegimeBadge label={data.regime.label} />}
                    className="lg:col-span-2"
                  >
                    <RegimeTimelineChart data={data.regime.series} height={140} />
                  </SectionCard>

                  <SectionCard title="Regime Distribution">
                    <RegimeDonutChart data={data.regime.distribution} height={200} />
                  </SectionCard>
                </div>
                </StaggerItem>

                <StaggerItem>
                <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-3">
                  <SectionCard title="Macro Overlay" className="lg:col-span-2">
                    <MacroLineChart
                      usdtry={data.macro.series.usdtry}
                      xauTry={data.macro.series.xau_try}
                      height={240}
                    />
                  </SectionCard>

                  <SectionCard title="Macro Changes" noPadding>
                    <DataTable columns={macroColumns} data={data.macro.changes} pageSize={10} />
                  </SectionCard>
                </div>
                </StaggerItem>

                <StaggerItem>
                <SectionCard title="Macro Detail">
                  <div className="mb-3 flex flex-wrap gap-2">
                    {[
                      { label: "90 Days", value: 90 as const },
                      { label: "6 Months", value: 126 as const },
                      { label: "1 Year", value: 252 as const },
                      { label: "5 Years", value: 1260 as const },
                    ].map((option) => (
                      <Button
                        key={option.value}
                        size="sm"
                        variant={macroDetailWindow === option.value ? "default" : "outline"}
                        onClick={() => setMacroDetailWindow(option.value)}
                      >
                        {option.label}
                      </Button>
                    ))}
                  </div>

                  <div className="grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-2">
                    <SectionCard title="USD/TRY Detail">
                      <MacroLineChart usdtry={macroSlices.usd} xauTry={[]} height={220} />
                    </SectionCard>
                    <SectionCard title="Gold (TRY/oz) Detail">
                      <MacroLineChart usdtry={[]} xauTry={macroSlices.gold} height={220} />
                    </SectionCard>
                  </div>

                  <SectionCard title="Weekly Changes" noPadding>
                    <DataTable
                      columns={[
                        { accessorKey: "asset", header: "Asset" },
                        {
                          accessorKey: "current",
                          header: "Current",
                          cell: ({ getValue }) => {
                            const value = getValue<number>();
                            return Number.isFinite(value) ? value.toLocaleString("en-US", { maximumFractionDigits: 3 }) : "—";
                          },
                        },
                        {
                          accessorKey: "d1_pct",
                          header: "1D Δ",
                          cell: ({ getValue }) => {
                            const v = getValue<number | null>();
                            return v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
                          },
                        },
                        {
                          accessorKey: "w1_pct",
                          header: "1W Δ",
                          cell: ({ getValue }) => {
                            const v = getValue<number | null>();
                            return v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
                          },
                        },
                        {
                          accessorKey: "m1_pct",
                          header: "1M Δ",
                          cell: ({ getValue }) => {
                            const v = getValue<number | null>();
                            return v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}%`;
                          },
                        },
                      ]}
                      data={macroDetailRows}
                      pageSize={10}
                    />
                  </SectionCard>
                </SectionCard>
                </StaggerItem>

                {data.date_range && (
                  <p className="text-right text-micro text-[var(--text-faint)]">
                    Data range: {data.date_range.start} - {data.date_range.end}
                  </p>
                )}
              </>
            ) : (
              <SectionCard className="flex items-center justify-center py-[var(--space-9)]">
                {isLoading ? (
                  <div className="flex flex-col items-center gap-[var(--space-3)]">
                    <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
                    <p className="text-small text-[var(--text-muted)]">Loading dashboard data…</p>
                  </div>
                ) : (
                  <p className="text-small text-[var(--text-muted)]">
                    Dashboard data is temporarily unavailable.
                  </p>
                )}
              </SectionCard>
            )}
          </PageSectionStack>
        </PageMain>
      </PageScaffold>
    </StaggerReveal>
  );
}
