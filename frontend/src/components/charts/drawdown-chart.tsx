"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";

interface DrawdownPoint {
  date: string;
  drawdown: number;
}

interface DrawdownChartProps {
  data: DrawdownPoint[];
  height?: number;
}

export function DrawdownChart({ data, height = 200 }: DrawdownChartProps) {
  const theme = useChartTheme();
  const valueFormatter = (val: number) => `${(val * 100).toFixed(2)}%`;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={theme.margins}>
        <defs>
          <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={theme.colors.bear} stopOpacity={0.3} />
            <stop offset="95%" stopColor={theme.colors.bear} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid {...theme.grid} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={(v: string) => v.slice(0, 7)}
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
        />
        <YAxis
          tickFormatter={valueFormatter}
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
          width={56}
        />
        <Tooltip
          contentStyle={theme.tooltip.contentStyle}
          formatter={(v: unknown) => [valueFormatter(v as number), "Drawdown"]}
        />
        <Area
          type="monotone"
          dataKey="drawdown"
          stroke={theme.colors.bear}
          strokeWidth={1.5}
          fill="url(#ddGradient)"
          animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
