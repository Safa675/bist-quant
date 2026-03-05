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
  ReferenceArea,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";
import { regimeColor } from "@/lib/tokens";
import type { EquityCurvePoint, RegimeBand } from "@/lib/types";

interface EquityCurveChartProps {
  data: EquityCurvePoint[];
  height?: number;
  showBenchmark?: boolean;
  /** Optional regime bands for background shading */
  regimeBands?: RegimeBand[];
}

export function EquityCurveChart({
  data,
  height = 300,
  showBenchmark = true,
  regimeBands,
}: EquityCurveChartProps) {
  const theme = useChartTheme();
  const tickFormatter = (val: string) => val.slice(0, 7);
  const valueFormatter = (val: number) => `${(val * 100).toFixed(1)}%`;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={theme.margins}>
        <CartesianGrid {...theme.grid} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={tickFormatter}
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
          labelStyle={theme.tooltip.labelStyle}
          formatter={(v: unknown) => [valueFormatter(v as number), ""]}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, color: theme.colors.textMuted }}
        />

        {/* Regime band shading */}
        {regimeBands?.map((band, i) => (
          <ReferenceArea
            key={`regime-${i}`}
            x1={band.start}
            x2={band.end}
            fill={regimeColor(band.regime)}
            fillOpacity={0.08}
            strokeOpacity={0}
            label=""
          />
        ))}

        <Line
          type="monotone"
          dataKey="strategy"
          name="Strategy"
          stroke={theme.colors.accent}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
          animationDuration={800}
        />
        {showBenchmark && (
          <Line
            type="monotone"
            dataKey="benchmark"
            name="Benchmark"
            stroke={theme.colors.neutral}
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            animationDuration={800}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
