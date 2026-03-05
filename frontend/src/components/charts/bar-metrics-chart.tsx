"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";

interface MetricItem {
  name: string;
  value: number;
}

interface BarMetricsChartProps {
  data: MetricItem[];
  height?: number;
  horizontal?: boolean;
}

export function BarMetricsChart({ data, height = 280, horizontal = true }: BarMetricsChartProps) {
  const theme = useChartTheme();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={horizontal ? "vertical" : "horizontal"}
        margin={theme.margins}
      >
        <CartesianGrid {...theme.grid} horizontal={!horizontal} vertical={horizontal} />
        {horizontal ? (
          <>
            <XAxis type="number" tick={theme.axis.tick} axisLine={theme.axis.axisLine} tickLine={theme.axis.tickLine} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ ...theme.axis.tick, fill: theme.colors.textMuted }}
              axisLine={theme.axis.axisLine}
              tickLine={theme.axis.tickLine}
              width={110}
            />
          </>
        ) : (
          <>
            <XAxis dataKey="name" tick={{ ...theme.axis.tick, fill: theme.colors.textMuted }} axisLine={theme.axis.axisLine} tickLine={theme.axis.tickLine} />
            <YAxis tick={theme.axis.tick} axisLine={theme.axis.axisLine} tickLine={theme.axis.tickLine} width={48} />
          </>
        )}
        <Tooltip contentStyle={theme.tooltip.contentStyle} />
        <Bar dataKey="value" radius={4} animationDuration={600}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value >= 0 ? theme.colors.bull : theme.colors.bear} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
