"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface SectionCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;
  noPadding?: boolean;
  /** Allow collapsing the body (default: false). */
  collapsible?: boolean;
  /** Initial collapsed state (default: false). */
  defaultCollapsed?: boolean;
}

export function SectionCard({
  title,
  subtitle,
  actions,
  noPadding = false,
  collapsible = false,
  defaultCollapsed = false,
  className,
  children,
  ...props
}: SectionCardProps) {
  const headingId = React.useId();
  const [collapsed, setCollapsed] = React.useState(defaultCollapsed);

  return (
    <div
      role={title ? "region" : undefined}
      aria-labelledby={title ? headingId : undefined}
      className={cn(
        "relative rounded-[var(--radius-lg)] border border-[var(--color-glass-border)] shadow-[var(--color-frost-card-shadow)] transition-all duration-250 ease-out overflow-hidden",
        "bg-[linear-gradient(145deg,rgba(255,255,255,0.07),rgba(148,163,184,0.05)_42%,rgba(15,23,42,0.18)_100%)]",
        "backdrop-blur-[14px]",
        "hover:border-[var(--color-glass-border-accent)] hover:shadow-[var(--color-frost-card-shadow),0_0_30px_rgba(74,158,255,0.06)]",
        className
      )}
      {...props}
    >
      {/* Inner glow layer */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 rounded-[inherit] bg-[linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.01)_40%,rgba(255,255,255,0)_72%)]"
      />

      {(title || actions) && (
        <div className="relative z-[1]">
          {/* Gradient accent line at top of header */}
          <div
            aria-hidden
            className="h-[2px] bg-[linear-gradient(90deg,var(--accent),var(--purple)_50%,transparent)] opacity-50"
          />
          <div
            className={cn(
              "flex items-center justify-between gap-[var(--space-4)] border-b border-[var(--color-glass-border)] px-[var(--card-pad)] py-[var(--space-4)]",
              collapsible && "cursor-pointer select-none"
            )}
            onClick={collapsible ? () => setCollapsed((c) => !c) : undefined}
            aria-expanded={collapsible ? !collapsed : undefined}
          >
            <div className="flex items-center gap-[var(--space-3)]">
              {collapsible && (
                <svg
                  className={cn(
                    "h-3.5 w-3.5 text-[var(--text-faint)] transition-transform duration-200",
                    collapsed ? "-rotate-90" : "rotate-0"
                  )}
                  viewBox="0 0 12 12"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
              <div>
                {title && <h2 id={headingId} className="text-h3 text-[var(--text)]">{title}</h2>}
                {subtitle && (
                  <p className="mt-1 text-small text-[var(--text-muted)]">{subtitle}</p>
                )}
              </div>
            </div>
            {actions && (
              <div
                className="flex shrink-0 items-center gap-[var(--space-2)]"
                onClick={(e) => e.stopPropagation()}
              >
                {actions}
              </div>
            )}
          </div>
        </div>
      )}
      <div
        className={cn(
          "relative z-[1] transition-all duration-300 ease-out",
          collapsed ? "max-h-0 opacity-0 overflow-hidden" : "max-h-[4000px] opacity-100",
          !noPadding && !collapsed && "p-[var(--card-pad)]"
        )}
      >
        {children}
      </div>
    </div>
  );
}
