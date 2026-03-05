"use client";

import * as React from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
} from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";
import { regimeColor } from "@/lib/tokens";
import type { RegimePoint } from "@/lib/types";

interface RegimeTimelineChartProps {
  data: RegimePoint[];
  height?: number;
}

/** Build contiguous regime bands from point-level data */
function buildBands(points: RegimePoint[]): { start: string; end: string; regime: string }[] {
  if (!points.length) return [];
  const bands: { start: string; end: string; regime: string }[] = [];
  let current = { start: points[0].date, end: points[0].date, regime: points[0].regime };

  for (let i = 1; i < points.length; i++) {
    if (points[i].regime === current.regime) {
      current.end = points[i].date;
    } else {
      bands.push({ ...current });
      current = { start: points[i].date, end: points[i].date, regime: points[i].regime };
    }
  }
  bands.push({ ...current });
  return bands;
}

export function RegimeTimelineChart({ data, height = 120 }: RegimeTimelineChartProps) {
  const theme = useChartTheme();
  const bands = React.useMemo(() => buildBands(data), [data]);

  const chartData = data.map((d) => ({
    date: d.date,
    value: 1,
    regime: d.regime,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ ...theme.margins, top: 4, bottom: 4 }}>
        <XAxis
          dataKey="date"
          tickFormatter={(v: string) => v.slice(0, 7)}
          tick={{ ...theme.axis.tick, fontSize: 10 }}
          axisLine={theme.axis.axisLine}
          tickLine={theme.axis.tickLine}
          interval="preserveStartEnd"
        />
        <YAxis hide domain={[0, 1]} />
        <Tooltip
          contentStyle={theme.tooltip.contentStyle}
          formatter={(_: unknown, __: unknown, item: { payload?: { regime?: string } }) => [
            item?.payload?.regime ?? "",
            "Regime",
          ]}
          labelFormatter={(label: unknown) => String(label)}
        />
        {/* Regime background bands */}
        {bands.map((band, i) => (
          <ReferenceArea
            key={i}
            x1={band.start}
            x2={band.end}
            y1={0}
            y2={1}
            fill={regimeColor(band.regime)}
            fillOpacity={0.35}
            strokeOpacity={0}
          />
        ))}
        {/* Invisible area just to make the chart render a data layer */}
        <Area
          type="stepAfter"
          dataKey="value"
          stroke="transparent"
          fill="transparent"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
