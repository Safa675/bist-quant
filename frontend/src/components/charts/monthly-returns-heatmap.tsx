"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { COLORS } from "@/lib/tokens";

interface MonthlyReturnsHeatmapProps {
  /** { year: { "Jan": 0.05, "Feb": -0.02, ... } } */
  data: Record<string, Record<string, number>>;
  className?: string;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function getCellStyle(value: number | undefined): React.CSSProperties {
  if (value === undefined || value === null)
    return { background: COLORS.bgElevated, color: COLORS.textFaint };
  const pct = value * 100;
  const intensity = Math.min(Math.abs(pct) / 10, 1);
  if (pct > 0) {
    const alpha = 0.15 + intensity * 0.65;
    return {
      background: `rgba(16,185,129,${alpha})`,
      color: alpha > 0.45 ? "#fff" : COLORS.bull,
    };
  }
  const alpha = 0.15 + intensity * 0.65;
  return {
    background: `rgba(239,68,68,${alpha})`,
    color: alpha > 0.45 ? "#fff" : COLORS.bear,
  };
}

export function MonthlyReturnsHeatmap({ data, className }: MonthlyReturnsHeatmapProps) {
  const years = Object.keys(data).sort().reverse();

  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th className="p-1.5 text-left text-[var(--text-faint)] font-medium w-12">Year</th>
            {MONTHS.map((m) => (
              <th key={m} className="p-1.5 text-center text-[var(--text-faint)] font-medium">
                {m}
              </th>
            ))}
            <th className="p-1.5 text-center text-[var(--text-faint)] font-medium">Full</th>
          </tr>
        </thead>
        <tbody>
          {years.map((year) => {
            const row = data[year] ?? {};
            const fullYear = Object.values(row).reduce((acc, v) => acc + v, 0);
            return (
              <tr key={year}>
                <td className="p-1.5 font-mono text-[var(--text-muted)] font-medium">{year}</td>
                {MONTHS.map((month) => {
                  const val = row[month];
                  return (
                    <td
                      key={month}
                      className="p-1 text-center rounded-sm font-mono"
                      style={getCellStyle(val)}
                      title={val !== undefined ? `${(val * 100).toFixed(2)}%` : "—"}
                    >
                      {val !== undefined ? `${(val * 100).toFixed(1)}` : "—"}
                    </td>
                  );
                })}
                <td
                  className="p-1 text-center rounded-sm font-mono font-semibold"
                  style={getCellStyle(fullYear)}
                >
                  {(fullYear * 100).toFixed(1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
