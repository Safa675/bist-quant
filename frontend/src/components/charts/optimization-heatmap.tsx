"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { COLORS } from "@/lib/tokens";

interface OptimizationHeatmapProps {
  /** x-axis param values */
  xValues: number[];
  /** y-axis param values */
  yValues: number[];
  /** Matrix of values [y][x] */
  matrix: number[][];
  xLabel?: string;
  yLabel?: string;
  metricLabel?: string;
  className?: string;
}

function interpolateColor(value: number, min: number, max: number): string {
  if (max === min) return "rgba(255,255,255,0.1)";
  const t = (value - min) / (max - min);
  if (t >= 0.5) {
    const intensity = (t - 0.5) * 2;
    return `rgba(16,185,129,${0.1 + intensity * 0.8})`;
  }
  const intensity = (0.5 - t) * 2;
  return `rgba(239,68,68,${0.1 + intensity * 0.7})`;
}

export function OptimizationHeatmap({
  xValues,
  yValues,
  matrix,
  xLabel = "Param X",
  yLabel = "Param Y",
  metricLabel = "Sharpe",
  className,
}: OptimizationHeatmapProps) {
  const allValues = matrix.flat().filter((v) => v !== null && !isNaN(v));
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);

  return (
    <div className={cn("overflow-x-auto", className)}>
      <div className="text-xs text-[var(--text-muted)] mb-2">
        X: {xLabel} &nbsp;|&nbsp; Y: {yLabel} &nbsp;|&nbsp; Value: {metricLabel}
      </div>
      <table className="w-full border-collapse text-[10px]">
        <thead>
          <tr>
            <th className="p-1 text-[var(--text-faint)]">{yLabel} ↓ / {xLabel} →</th>
            {xValues.map((x) => (
              <th key={x} className="p-1 text-center text-[var(--text-faint)] font-mono">
                {x}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {yValues.map((y, yi) => (
            <tr key={y}>
              <td className="p-1 text-center text-[var(--text-faint)] font-mono">{y}</td>
              {xValues.map((_, xi) => {
                const val = matrix[yi]?.[xi];
                return (
                  <td
                    key={xi}
                    className="p-1 text-center rounded-sm font-mono"
                    style={{
                      background: val !== undefined ? interpolateColor(val, min, max) : COLORS.bgElevated,
                      color: val !== undefined ? COLORS.text : COLORS.textFaint,
                    }}
                    title={val !== undefined ? String(val.toFixed(3)) : "—"}
                  >
                    {val !== undefined ? val.toFixed(2) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
