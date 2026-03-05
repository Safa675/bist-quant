"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";
import type { MacroPoint } from "@/lib/types";

interface MacroLineChartProps {
  usdtry: MacroPoint[];
  xauTry: MacroPoint[];
  height?: number;
}

export function MacroLineChart({ usdtry, xauTry, height = 200 }: MacroLineChartProps) {
  const theme = useChartTheme();
  // Merge two series by date
  const dateMap: Record<string, { date: string; usdtry?: number; xau_try?: number }> = {};
  for (const p of usdtry) {
    dateMap[p.date] = { date: p.date, usdtry: p.value };
  }
  for (const p of xauTry) {
    if (dateMap[p.date]) {
      dateMap[p.date].xau_try = p.value;
    } else {
      dateMap[p.date] = { date: p.date, xau_try: p.value };
    }
  }
  const data = Object.values(dateMap).sort((a, b) => a.date.localeCompare(b.date));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={theme.margins}>
        <CartesianGrid {...theme.grid} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(v: string) => v.slice(0, 7)}
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
        />
        <YAxis
          yAxisId="left"
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
          width={48}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
          width={56}
        />
        <Tooltip contentStyle={theme.tooltip.contentStyle} />
        <Legend wrapperStyle={{ fontSize: 12, color: theme.colors.textMuted }} />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="usdtry"
          name="USD/TRY"
          stroke={theme.colors.info}
          strokeWidth={1.5}
          dot={false}
          animationDuration={800}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="xau_try"
          name="Gold/TRY"
          stroke={theme.colors.neutral}
          strokeWidth={1.5}
          dot={false}
          animationDuration={800}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
