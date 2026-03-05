import * as React from "react";
import { AlertCircle } from "lucide-react";
import { isApiClientError } from "@/lib/api-error";
import { cn } from "@/lib/utils";

interface ApiErrorProps {
  error: Error | string | null | undefined;
  className?: string;
}

export function ApiError({ error, className }: ApiErrorProps) {
  if (!error) return null;

  const message = error instanceof Error ? error.message : error;
  const detail = isApiClientError(error) ? error.detail : message;
  const hint = isApiClientError(error) ? error.hint : null;
  const code = isApiClientError(error) ? error.code : null;

  return (
    <div
      className={cn(
        "flex items-start gap-3 rounded-[var(--radius)] border border-[var(--bear)] bg-[var(--bear-dim)] p-4",
        className
      )}
      role="alert"
    >
      <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-[var(--bear)]" />
      <div>
        <p className="text-sm font-medium text-[var(--bear)]">Error</p>
        <p className="mt-0.5 text-xs text-[var(--text-muted)]">{detail}</p>
        {hint && <p className="mt-1 text-xs text-[var(--text-muted)]">Hint: {hint}</p>}
        {code && <p className="mt-1 text-[11px] uppercase tracking-wide text-[var(--text-muted)]">Code: {code}</p>}
      </div>
    </div>
  );
}
