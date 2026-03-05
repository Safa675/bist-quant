import { describe, it, expect } from "vitest";
import { useChartTheme } from "@/hooks/use-chart-theme";
import { renderHook } from "@testing-library/react";

describe("useChartTheme", () => {
  it("returns colors object", () => {
    const { result } = renderHook(() => useChartTheme());
    expect(result.current.colors).toBeDefined();
    expect(result.current.colors.bull).toBe("#4ed08a");
    expect(result.current.colors.bear).toBe("#ff6d76");
  });

  it("returns axis configuration", () => {
    const { result } = renderHook(() => useChartTheme());
    expect(result.current.axis.tick.fontSize).toBe(11);
    expect(result.current.axis.axisLine).toBe(false);
    expect(result.current.axis.tickLine).toBe(false);
  });

  it("returns tooltip styles", () => {
    const { result } = renderHook(() => useChartTheme());
    expect(result.current.tooltip.contentStyle).toHaveProperty("background");
    expect(result.current.tooltip.contentStyle).toHaveProperty("borderRadius");
  });

  it("returns grid configuration", () => {
    const { result } = renderHook(() => useChartTheme());
    expect(result.current.grid.strokeDasharray).toBe("3 3");
  });

  it("returns margins", () => {
    const { result } = renderHook(() => useChartTheme());
    expect(result.current.margins).toHaveProperty("top");
    expect(result.current.margins).toHaveProperty("right");
    expect(result.current.margins).toHaveProperty("bottom");
    expect(result.current.margins).toHaveProperty("left");
  });
});
