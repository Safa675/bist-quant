import * as React from "react";
import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>;

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-[var(--control-h)] w-full rounded-[var(--radius)] border border-[var(--color-glass-border)] bg-[var(--color-glass)] backdrop-blur-[10px] px-3 py-1 text-sm text-[var(--text)] shadow-sm transition-all duration-200",
          "placeholder:text-[var(--text-faint)]",
          "hover:border-[var(--color-glass-border-accent)] hover:bg-[var(--color-glass-strong)]",
          "focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/40 focus:border-[var(--color-glass-border-accent)] focus:shadow-[0_0_16px_rgba(74,158,255,0.15)]",
          "disabled:cursor-not-allowed disabled:opacity-55",
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
