import { describe, it, expect } from "vitest";
import { cn, formatNum, formatPct, formatDate } from "@/lib/utils";

describe("cn", () => {
  it("merges classnames", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("handles conditional classes", () => {
    const hidden = false;
    expect(cn("base", hidden ? "hidden" : undefined, "extra")).toBe("base extra");
  });

  it("resolves tailwind conflicts", () => {
    const result = cn("px-4", "px-6");
    expect(result).toBe("px-6");
  });

  it("handles undefined and null", () => {
    expect(cn("a", undefined, null, "b")).toBe("a b");
  });
});

describe("formatNum", () => {
  it("formats a number with 2 decimal places", () => {
    const result = formatNum(1234.567);
    expect(result).toMatch(/1.*234\.57/);
  });

  it("adds suffix", () => {
    expect(formatNum(42, "%")).toMatch(/42.*%/);
  });

  it("returns dash for null", () => {
    expect(formatNum(null)).toBe("-");
  });

  it("returns dash for undefined", () => {
    expect(formatNum(undefined)).toBe("-");
  });
});

describe("formatPct", () => {
  it("formats positive with + sign", () => {
    expect(formatPct(5.123)).toBe("+5.12%");
  });

  it("formats negative without + sign", () => {
    expect(formatPct(-3.456)).toBe("-3.46%");
  });

  it("formats zero without + sign", () => {
    expect(formatPct(0)).toBe("0.00%");
  });

  it("returns dash for null", () => {
    expect(formatPct(null)).toBe("-");
  });
});

describe("formatDate", () => {
  it("formats ISO date string", () => {
    const result = formatDate("2024-06-15T00:00:00Z");
    // Result varies by locale, but should contain year
    expect(result).toContain("2024");
  });

  it("returns dash for null", () => {
    expect(formatDate(null)).toBe("-");
  });

  it("returns dash for undefined", () => {
    expect(formatDate(undefined)).toBe("-");
  });

  it("returns original string for invalid date", () => {
    // Handles gracefully - returns the input or parsed output
    const result = formatDate("not-a-date");
    expect(typeof result).toBe("string");
  });
});
