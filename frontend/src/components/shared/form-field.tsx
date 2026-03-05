import * as React from "react";
import { cn } from "@/lib/utils";

interface FormFieldProps {
  label: string;
  htmlFor?: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
}

interface FormGroupProps {
  children: React.ReactNode;
  className?: string;
}

export function FormField({ label, htmlFor, hint, children, className }: FormFieldProps) {
  return (
    <label className={cn("flex w-full flex-col gap-[var(--space-2)]", className)} htmlFor={htmlFor}>
      <span className="text-micro font-semibold uppercase tracking-wide text-[var(--text-faint)]">
        {label}
      </span>
      {children}
      {hint && <span className="text-small text-[var(--text-muted)]">{hint}</span>}
    </label>
  );
}

export function FormLabel({ children, className }: FormGroupProps) {
  return (
    <span className={cn("text-micro font-semibold uppercase tracking-wide text-[var(--text-faint)]", className)}>
      {children}
    </span>
  );
}

export function FormHint({ children, className }: FormGroupProps) {
  return <span className={cn("text-small text-[var(--text-muted)]", className)}>{children}</span>;
}

export function FormRow({ children, className }: FormGroupProps) {
  return (
    <div className={cn("grid grid-cols-1 gap-[var(--space-3)] sm:grid-cols-2", className)}>
      {children}
    </div>
  );
}

export function FormGrid({ children, className }: FormGroupProps) {
  return (
    <div className={cn("grid grid-cols-1 gap-[var(--space-3)]", className)} data-ui-form-grid>
      {children}
    </div>
  );
}
