"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Label,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";

interface ScatterPoint {
  x: number;
  y: number;
  label?: string;
}

interface ScatterPlotProps {
  data: ScatterPoint[];
  xLabel?: string;
  yLabel?: string;
  height?: number;
  referenceX?: number;
  referenceY?: number;
}

export function ScatterPlot({
  data,
  xLabel = "X",
  yLabel = "Y",
  height = 280,
  referenceX,
  referenceY,
}: ScatterPlotProps) {
  const theme = useChartTheme();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={theme.margins}>
        <CartesianGrid {...theme.grid} />
        <XAxis
          type="number"
          dataKey="x"
          name={xLabel}
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
        >
          <Label value={xLabel} position="insideBottom" offset={-4} style={{ fill: theme.colors.textMuted, fontSize: 11 }} />
        </XAxis>
        <YAxis
          type="number"
          dataKey="y"
          name={yLabel}
          tick={theme.axis.tick}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
          width={48}
        >
          <Label value={yLabel} angle={-90} position="insideLeft" offset={8} style={{ fill: theme.colors.textMuted, fontSize: 11 }} />
        </YAxis>
        <Tooltip
          contentStyle={theme.tooltip.contentStyle}
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(v: unknown, name: string | undefined) => [
            (v as number).toFixed(3),
            name ?? "",
          ]}
        />
        {referenceX !== undefined && (
          <ReferenceLine x={referenceX} stroke={theme.colors.accent} strokeDasharray="4 4" />
        )}
        {referenceY !== undefined && (
          <ReferenceLine y={referenceY} stroke={theme.colors.neutral} strokeDasharray="4 4" />
        )}
        <Scatter
          data={data}
          fill={theme.colors.accent}
          fillOpacity={0.7}
          animationDuration={800}
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
