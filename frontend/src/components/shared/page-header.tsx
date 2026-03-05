import * as React from "react";
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  className?: string;
}

export function PageHeader({ title, subtitle, actions, className }: PageHeaderProps) {
  return (
    <header
      className={cn(
        "relative mb-[var(--section-gap)] flex flex-col gap-[var(--space-4)] border-b border-[var(--color-glass-border)] pb-[var(--space-4)] md:flex-row md:items-start md:justify-between",
        className
      )}
      data-ui-page-header
    >
      <div>
        <h1 className="text-h1 bg-[linear-gradient(90deg,#ffffff_20%,var(--text-muted)_80%)] bg-clip-text text-transparent drop-shadow-[0_2px_10px_rgba(74,158,255,0.2)] tracking-[-0.01em]">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-[var(--space-2)] max-w-[72ch] text-small text-[var(--text-muted)] tracking-[0.01em]">
            {subtitle}
          </p>
        )}
        {/* Decorative gradient underline accent */}
        <div
          aria-hidden
          className="mt-[var(--space-3)] h-[2px] w-16 rounded-full bg-[linear-gradient(90deg,var(--accent),var(--purple),transparent)] opacity-70"
        />
      </div>
      {actions && (
        <div className="flex shrink-0 items-center gap-[var(--space-2)]">{actions}</div>
      )}
    </header>
  );
}
