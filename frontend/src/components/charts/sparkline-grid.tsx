"use client";

import * as React from "react";
import { ResponsiveContainer, LineChart, Line, Tooltip } from "recharts";
import { useChartTheme } from "@/hooks/use-chart-theme";
import { cn } from "@/lib/utils";

interface SparklineItem {
  name: string;
  data: { date: string; value: number }[];
  color?: string;
  suffix?: string;
}

interface SparklineGridProps {
  items: SparklineItem[];
  height?: number;
  className?: string;
}

function Sparkline({
  data,
  color,
  tooltipStyle,
}: {
  data: { date: string; value: number }[];
  color: string;
  tooltipStyle: React.CSSProperties;
}) {
  return (
    <ResponsiveContainer width="100%" height={48}>
      <LineChart data={data} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
        <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} dot={false} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelFormatter={() => ""}
          formatter={(v: unknown) => [(v as number).toFixed(2), ""]}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SparklineGrid({ items, className }: SparklineGridProps) {
  const theme = useChartTheme();
  const tooltipStyle: React.CSSProperties = {
    ...theme.tooltip.contentStyle,
    borderRadius: 6,
    fontSize: 11,
  };

  return (
    <div className={cn("grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4", className)}>
      {items.map((item) => {
        const last = item.data.at(-1)?.value;
        const first = item.data[0]?.value;
        const change = last !== undefined && first !== undefined && first !== 0
          ? ((last - first) / Math.abs(first)) * 100
          : null;
        const color = item.color ?? (change !== null && change >= 0 ? theme.colors.bull : theme.colors.bear);

        return (
          <div
            key={item.name}
            className="rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-panel)] p-3"
          >
            <p className="text-xs text-[var(--text-muted)] mb-1">{item.name}</p>
            <p className="text-base font-semibold text-[var(--text)] tabular-nums">
              {last !== undefined ? `${last.toFixed(2)}${item.suffix ?? ""}` : "—"}
            </p>
            {change !== null && (
              <p
                className="text-xs font-medium mb-1"
                style={{ color: change >= 0 ? theme.colors.bull : theme.colors.bear }}
              >
                {change >= 0 ? "+" : ""}{change.toFixed(2)}%
              </p>
            )}
            <Sparkline data={item.data} color={color} tooltipStyle={tooltipStyle} />
          </div>
        );
      })}
    </div>
  );
}
