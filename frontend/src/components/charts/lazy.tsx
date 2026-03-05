/**
 * Lazy-loaded chart components via next/dynamic.
 * Heavy Recharts bundles are code-split per page.
 */

import dynamic from "next/dynamic";
import { LoadingSpinner } from "@/components/shared/loading-spinner";

const chartLoader = () => (
  <div className="flex h-64 items-center justify-center">
    <LoadingSpinner size="lg" />
  </div>
);

export const LazyEquityCurveChart = dynamic(
  () => import("./equity-curve-chart").then((m) => ({ default: m.EquityCurveChart })),
  { loading: chartLoader, ssr: false },
);

export const LazyDrawdownChart = dynamic(
  () => import("./drawdown-chart").then((m) => ({ default: m.DrawdownChart })),
  { loading: chartLoader, ssr: false },
);

export const LazyMonthlyReturnsHeatmap = dynamic(
  () => import("./monthly-returns-heatmap").then((m) => ({ default: m.MonthlyReturnsHeatmap })),
  { loading: chartLoader, ssr: false },
);

export const LazyRegimeTimelineChart = dynamic(
  () => import("./regime-timeline-chart").then((m) => ({ default: m.RegimeTimelineChart })),
  { loading: chartLoader, ssr: false },
);

export const LazyRegimeDonutChart = dynamic(
  () => import("./regime-donut-chart").then((m) => ({ default: m.RegimeDonutChart })),
  { loading: chartLoader, ssr: false },
);

export const LazyMacroLineChart = dynamic(
  () => import("./macro-line-chart").then((m) => ({ default: m.MacroLineChart })),
  { loading: chartLoader, ssr: false },
);

export const LazyOptimizationHeatmap = dynamic(
  () => import("./optimization-heatmap").then((m) => ({ default: m.OptimizationHeatmap })),
  { loading: chartLoader, ssr: false },
);

export const LazyScatterPlot = dynamic(
  () => import("./scatter-plot").then((m) => ({ default: m.ScatterPlot })),
  { loading: chartLoader, ssr: false },
);

export const LazyBarMetricsChart = dynamic(
  () => import("./bar-metrics-chart").then((m) => ({ default: m.BarMetricsChart })),
  { loading: chartLoader, ssr: false },
);

export const LazySparklineGrid = dynamic(
  () => import("./sparkline-grid").then((m) => ({ default: m.SparklineGrid })),
  { loading: chartLoader, ssr: false },
);

export const LazyGreeksBarChart = dynamic(
  () => import("./greeks-bar-chart").then((m) => ({ default: m.GreeksBarChart })),
  { loading: chartLoader, ssr: false },
);
