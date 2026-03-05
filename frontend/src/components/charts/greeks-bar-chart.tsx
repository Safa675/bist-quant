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
  ReferenceLine,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";
import type { GreeksResult } from "@/lib/types";

interface GreeksBarChartProps {
  greeks: GreeksResult;
  height?: number;
}

export function GreeksBarChart({ greeks, height = 240 }: GreeksBarChartProps) {
  const theme = useChartTheme();
  const data = [
    { name: "Delta", value: greeks.delta },
    { name: "Gamma", value: greeks.gamma },
    { name: "Theta/day", value: greeks.theta_per_day },
    { name: "Vega/1%", value: greeks.vega_per_1pct },
    { name: "Rho/1%", value: greeks.rho_per_1pct },
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={theme.margins}>
        <CartesianGrid {...theme.grid} horizontal={false} />
        <XAxis
          type="number"
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ ...theme.axis.tick, fill: theme.colors.textMuted }}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
          width={72}
        />
        <Tooltip
          contentStyle={theme.tooltip.contentStyle}
          formatter={(v: unknown) => [(v as number).toFixed(4), ""]}
        />
        <ReferenceLine x={0} stroke={theme.colors.border} />
        <Bar dataKey="value" radius={4} animationDuration={600}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value >= 0 ? theme.colors.bull : theme.colors.bear} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
