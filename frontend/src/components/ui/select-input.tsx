import * as React from "react";
import { cn } from "@/lib/utils";

export type SelectInputProps = React.SelectHTMLAttributes<HTMLSelectElement>;

const SelectInput = React.forwardRef<HTMLSelectElement, SelectInputProps>(
  ({ className, children, ...props }, ref) => {
    return (
      <select
        ref={ref}
        data-ui-shared-control="select"
        className={cn(
          "flex h-[var(--control-h)] w-full rounded-[var(--radius)] border border-[var(--border)] bg-[var(--surface-2)] px-3 py-1 text-sm text-[var(--text)] shadow-sm transition-colors",
          "focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:border-[var(--accent)]",
          "disabled:cursor-not-allowed disabled:opacity-55",
          className
        )}
        {...props}
      >
        {children}
      </select>
    );
  }
);

SelectInput.displayName = "SelectInput";

export { SelectInput };
