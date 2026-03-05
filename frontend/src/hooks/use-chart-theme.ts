"use client";

import { COLORS, CHART_MARGINS, FONTS } from "@/lib/tokens";

/**
 * Returns consistent chart theme tokens for use in Recharts components.
 * Reading from the tokens file ensures a single source of truth.
 */
export function useChartTheme() {
  return {
    colors: COLORS,
    margins: CHART_MARGINS,
    fonts: FONTS,
    axis: {
      tick: { fill: COLORS.textFaint, fontSize: 11 },
      axisLine: false as const,
      tickLine: false as const,
    },
    tooltip: {
      contentStyle: {
        background: COLORS.bgElevated,
        border: `1px solid ${COLORS.border}`,
        borderRadius: 8,
        fontSize: 12,
        color: COLORS.text,
      },
      labelStyle: { color: COLORS.textMuted },
    },
    grid: {
      stroke: COLORS.gridLine,
      strokeDasharray: "3 3" as const,
    },
  } as const;
}
