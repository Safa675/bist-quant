"use client";

import * as React from "react";
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";
import { regimeColor } from "@/lib/tokens";
import type { RegimeDistributionItem } from "@/lib/types";

interface RegimeDonutChartProps {
  data: RegimeDistributionItem[];
  height?: number;
}

export function RegimeDonutChart({ data, height = 240 }: RegimeDonutChartProps) {
  const theme = useChartTheme();

  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="percent"
          nameKey="regime"
          cx="50%"
          cy="50%"
          innerRadius="55%"
          outerRadius="80%"
          paddingAngle={2}
          animationDuration={800}
        >
          {data.map((entry, i) => (
            <Cell key={i} fill={regimeColor(entry.regime)} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={theme.tooltip.contentStyle}
          formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, ""]}
        />
        <Legend
          formatter={(value: string) => (
            <span style={{ color: theme.colors.textMuted, fontSize: 12 }}>{value}</span>
          )}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
