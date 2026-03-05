"use client";

import * as React from "react";
import { getDashboardOverview } from "@/lib/api";
import type { DashboardOverview } from "@/lib/types";

interface UseDashboardResult {
  data: DashboardOverview | null;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
  lookback: number;
  setLookback: (days: number) => void;
}

export function useDashboard(initialLookback = 504): UseDashboardResult {
  const [lookback, setLookback] = React.useState(initialLookback);
  const [data, setData] = React.useState<DashboardOverview | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);
  const [tick, setTick] = React.useState(0);

  React.useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    getDashboardOverview(lookback)
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setIsLoading(false);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [lookback, tick]);

  const refetch = React.useCallback(() => setTick((t) => t + 1), []);

  return { data, isLoading, error, refetch, lookback, setLookback };
}
