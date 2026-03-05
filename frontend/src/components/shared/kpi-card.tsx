"use client";

import * as React from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

/** Simple easeOutExpo for smooth count-up. */
function easeOutExpo(t: number): number {
  return t >= 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

/**
 * Animated counter hook — animates from 0 → target over `duration` ms.
 * Returns the current display value as a formatted string.
 */
function useCountUp(
  target: number | null | undefined,
  opts: { duration?: number; decimals?: number; suffix?: string } = {},
): string {
  const { duration = 800, decimals = 2, suffix = "" } = opts;
  const [display, setDisplay] = React.useState<string>("—");

  React.useEffect(() => {
    if (target === null || target === undefined || isNaN(target)) {
      setDisplay("—");
      return;
    }

    let raf: number;
    const start = performance.now();

    function tick(now: number) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = easeOutExpo(progress);
      const current = eased * target!;

      setDisplay(
        current.toLocaleString("en-US", {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        }) + suffix,
      );

      if (progress < 1) {
        raf = requestAnimationFrame(tick);
      }
    }

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, decimals, suffix]);

  return display;
}

interface KpiCardProps {
  title: string;
  value: string | number | null | undefined;
  change?: number | null;
  suffix?: string;
  className?: string;
  icon?: React.ReactNode;
  /** Set to false to disable count-up animation for string values. */
  animate?: boolean;
  /** Number of decimal places in the animated number (default 2). */
  decimals?: number;
}

export function KpiCard({
  title,
  value,
  change,
  suffix,
  className,
  icon,
  animate = true,
  decimals = 2,
}: KpiCardProps) {
  const numericValue =
    typeof value === "number" ? value : typeof value === "string" ? parseFloat(value) : null;

  const animatedDisplay = useCountUp(
    animate && numericValue !== null && !isNaN(numericValue as number) ? numericValue : null,
    { decimals, suffix },
  );

  const displayValue =
    animate && numericValue !== null && !isNaN(numericValue as number)
      ? animatedDisplay
      : value === null || value === undefined
        ? "—"
        : `${value}${suffix ?? ""}`;

  const isPositive = change !== undefined && change !== null && change > 0;
  const isNegative = change !== undefined && change !== null && change < 0;
  const hasChange = change !== undefined && change !== null;

  return (
    <div
      role="status"
      aria-label={`${title}: ${displayValue}`}
      className={cn(
        "relative rounded-[var(--radius-lg)] border border-[var(--color-glass-border)] p-[var(--card-pad)] shadow-[var(--color-frost-card-shadow)] transition-all duration-250 ease-out",
        "bg-[linear-gradient(145deg,rgba(255,255,255,0.07),rgba(148,163,184,0.05)_42%,rgba(15,23,42,0.18)_100%)]",
        "backdrop-blur-[14px]",
        "hover:border-[var(--color-glass-border-accent)] hover:shadow-[var(--color-frost-card-shadow),0_0_30px_rgba(74,158,255,0.08)] hover:-translate-y-[1px]",
        className
      )}
      data-ui-kpi-card
    >
      {/* Inner glow layer */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-[inherit] bg-[linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.01)_40%,rgba(255,255,255,0)_72%)]"
      />

      <div className="relative z-[1]">
        <div className="mb-[var(--space-3)] flex items-center justify-between">
          <span className="text-micro font-semibold uppercase tracking-wide text-[var(--text-faint)]">
            {title}
          </span>
          {icon && <div className="text-[var(--text-faint)]">{icon}</div>}
        </div>

        <div className={cn(
          "text-h2 tabular-nums text-[var(--text)]",
          isPositive && "text-glow-bull",
          isNegative && "text-glow-bear",
          !isPositive && !isNegative && hasChange && "text-glow"
        )}>
          {displayValue}
        </div>

        {hasChange && (
          <div
            className={cn(
              "mt-[var(--space-2)] flex items-center gap-1 text-small font-medium",
              isPositive && "text-[var(--bull)]",
              isNegative && "text-[var(--bear)]",
              !isPositive && !isNegative && "text-[var(--text-muted)]"
            )}
          >
            {isPositive ? (
              <TrendingUp className="h-3.5 w-3.5 animate-pulse" style={{ animationDuration: '2s' }} />
            ) : isNegative ? (
              <TrendingDown className="h-3.5 w-3.5 animate-pulse" style={{ animationDuration: '2s' }} />
            ) : (
              <Minus className="h-3.5 w-3.5" />
            )}
            <span>
              {change > 0 ? "+" : ""}
              {change?.toFixed(2)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
