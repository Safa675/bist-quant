import * as React from "react";
import { cn } from "@/lib/utils";

interface PageScaffoldProps {
  children: React.ReactNode;
  className?: string;
}

interface PageSlotProps {
  children: React.ReactNode;
  className?: string;
}

export function PageScaffold({ children, className }: PageScaffoldProps) {
  return (
    <div
      className={cn("grid grid-cols-1 gap-[var(--grid-gap)] lg:grid-cols-12", className)}
      data-ui-scaffold
    >
      {children}
    </div>
  );
}

export function PageSidebar({ children, className }: PageSlotProps) {
  return (
    <aside
      className={cn("space-y-[var(--section-gap)] lg:col-span-4 xl:col-span-3", className)}
      data-ui-sidebar
    >
      {children}
    </aside>
  );
}

export function PageMain({ children, className }: PageSlotProps) {
  return (
    <section
      className={cn("space-y-[var(--section-gap)] lg:col-span-8 xl:col-span-9", className)}
      data-ui-main
    >
      {children}
    </section>
  );
}

export function PageSectionStack({ children, className }: PageSlotProps) {
  return (
    <div className={cn("space-y-[var(--section-gap)]", className)} data-ui-section-stack>
      {children}
    </div>
  );
}

export function PageKpiRow({ children, className }: PageSlotProps) {
  return (
    <div
      className={cn("grid grid-cols-2 gap-[var(--grid-gap)] sm:grid-cols-4", className)}
      data-ui-kpi-row
    >
      {children}
    </div>
  );
}
