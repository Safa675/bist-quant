"use client";

import * as React from "react";

interface UseApiOptions<T> {
  /** Initial data value */
  initialData?: T;
  /** Skip fetching (useful when params aren't ready) */
  skip?: boolean;
}

interface UseApiState<T> {
  data: T | undefined;
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Simple fetch hook for API calls.
 * Re-fetches whenever `fetcher` reference changes.
 */
export function useApi<T>(
  fetcher: (() => Promise<T>) | null,
  options?: UseApiOptions<T>
): UseApiState<T> {
  const [data, setData] = React.useState<T | undefined>(options?.initialData);
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);
  const [tick, setTick] = React.useState(0);

  React.useEffect(() => {
    if (!fetcher || options?.skip) return;

    let cancelled = false;
    setIsLoading(true);
    setError(null);

    fetcher()
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
  }, [fetcher, tick, options?.skip]);

  const refetch = React.useCallback(() => setTick((t) => t + 1), []);

  return { data, isLoading, error, refetch };
}
