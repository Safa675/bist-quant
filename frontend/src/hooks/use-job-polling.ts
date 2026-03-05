"use client";

import * as React from "react";
import { getJob } from "@/lib/api";
import type { JobPayload } from "@/lib/types";

const TERMINAL_STATES = new Set(["completed", "failed", "cancelled"]);
const POLL_INTERVAL_MS = 2000;

interface UseJobPollingResult {
  job: JobPayload | null;
  isPolling: boolean;
  error: Error | null;
}

/**
 * Polls a job by ID until it reaches a terminal state (completed/failed/cancelled).
 * Returns null if no jobId is provided.
 */
export function useJobPolling(jobId: string | null): UseJobPollingResult {
  const [job, setJob] = React.useState<JobPayload | null>(null);
  const [isPolling, setIsPolling] = React.useState(false);
  const [error, setError] = React.useState<Error | null>(null);

  React.useEffect(() => {
    if (!jobId) {
      setJob(null);
      setIsPolling(false);
      setError(null);
      return;
    }

    setIsPolling(true);
    setError(null);
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    async function poll() {
      if (cancelled) return;
      try {
        const result = await getJob(jobId!);
        if (cancelled) return;
        setJob(result);
        if (TERMINAL_STATES.has(result.status)) {
          setIsPolling(false);
          return;
        }
        timeoutId = setTimeout(poll, POLL_INTERVAL_MS);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error(String(err)));
          setIsPolling(false);
        }
      }
    }

    poll();

    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [jobId]);

  return { job, isPolling, error };
}
