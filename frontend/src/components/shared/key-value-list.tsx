import * as React from "react";
import { cn } from "@/lib/utils";

interface KeyValueListProps {
  data: Record<string, unknown>;
  className?: string;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    if (!value.length) return "—";
    if (value.every((item) => typeof item === "string" || typeof item === "number")) {
      return value.join(", ");
    }
    return JSON.stringify(value);
  }
  return JSON.stringify(value);
}

export function KeyValueList({ data, className }: KeyValueListProps) {
  const entries = Object.entries(data).filter(([, value]) => value !== undefined);

  if (!entries.length) {
    return (
      <p className={cn("rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)] p-3 text-small text-[var(--text-muted)]", className)}>
        No details available.
      </p>
    );
  }

  return (
    <dl
      className={cn(
        "divide-y divide-[var(--border-muted)] rounded-[var(--radius)] border border-[var(--border)] bg-[var(--bg-elevated)]",
        className
      )}
      data-ui-key-value-list
    >
      {entries.map(([key, value]) => (
        <div key={key} className="grid grid-cols-[minmax(110px,1fr)_2fr] gap-[var(--space-3)] px-3 py-2">
          <dt className="text-micro font-semibold uppercase tracking-wide text-[var(--text-faint)]">{key}</dt>
          <dd className="break-all text-small text-[var(--text)]">{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}
