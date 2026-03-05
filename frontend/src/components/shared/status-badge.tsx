import * as React from "react";
import { cn } from "@/lib/utils";
import type { JobStatus } from "@/lib/types";

const STATUS_CONFIG: Record<
  JobStatus,
  { label: string; color: string; bg: string; border: string; pulse?: boolean }
> = {
  queued: {
    label: "Queued",
    color: "var(--text-muted)",
    bg: "var(--bg-elevated)",
    border: "var(--border)",
  },
  running: {
    label: "Running",
    color: "var(--info)",
    bg: "rgba(59,130,246,0.1)",
    border: "rgba(59,130,246,0.3)",
    pulse: true,
  },
  completed: {
    label: "Completed",
    color: "var(--bull)",
    bg: "var(--bull-dim)",
    border: "rgba(16,185,129,0.3)",
  },
  failed: {
    label: "Failed",
    color: "var(--bear)",
    bg: "var(--bear-dim)",
    border: "rgba(239,68,68,0.3)",
  },
  cancelled: {
    label: "Cancelled",
    color: "var(--text-faint)",
    bg: "var(--bg-elevated)",
    border: "var(--border)",
  },
};

interface StatusBadgeProps {
  status: JobStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.queued;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold",
        className
      )}
      style={{
        color: cfg.color,
        backgroundColor: cfg.bg,
        borderColor: cfg.border,
      }}
    >
      <span
        className={cn("h-1.5 w-1.5 rounded-full", cfg.pulse && "animate-pulse")}
        style={{ backgroundColor: cfg.color }}
      />
      {cfg.label}
    </span>
  );
}
