"use client";

import * as React from "react";
import { formatDistanceToNow } from "date-fns";
import { X, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { JobPayload } from "@/lib/types";
import { StatusBadge } from "@/components/shared/status-badge";
import { Button } from "@/components/ui/button";

interface JobHistoryPanelProps {
  jobs: JobPayload[];
  onCancel?: (id: string) => void;
  onRefresh?: () => void;
  className?: string;
}

export function JobHistoryPanel({
  jobs,
  onCancel,
  onRefresh,
  className,
}: JobHistoryPanelProps) {
  if (jobs.length === 0) {
    return (
      <div className={cn("rounded-[var(--radius-lg)] border border-[var(--border)] bg-[var(--bg-panel)] p-8 text-center", className)}>
        <p className="text-sm text-[var(--text-muted)]">No jobs yet.</p>
      </div>
    );
  }

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-[var(--text)]">Job History</h3>
        {onRefresh && (
          <Button variant="ghost" size="icon" onClick={onRefresh} aria-label="Refresh jobs">
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
      {jobs.map((job) => (
        <div
          key={job.id}
          className="flex items-center justify-between gap-3 rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] px-4 py-3"
        >
          <div className="flex items-center gap-3 min-w-0">
            <StatusBadge status={job.status} />
            <div className="min-w-0">
              <p className="text-sm font-medium text-[var(--text)] truncate capitalize">
                {job.kind.replace(/_/g, " ")}
              </p>
              <p className="text-xs text-[var(--text-faint)]">
                {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}
              </p>
            </div>
          </div>
          {onCancel && (job.status === "queued" || job.status === "running") && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onCancel(job.id)}
              aria-label="Cancel job"
              className="shrink-0"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      ))}
    </div>
  );
}
