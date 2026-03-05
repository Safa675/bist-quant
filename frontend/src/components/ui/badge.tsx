import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none",
  {
    variants: {
      variant: {
        default: "border-transparent bg-[var(--accent-dim)] text-[var(--accent)]",
        secondary: "border-transparent bg-[var(--bg-elevated)] text-[var(--text-muted)]",
        destructive: "border-transparent bg-[var(--bear-dim)] text-[var(--bear)]",
        success: "border-transparent bg-[var(--bull-dim)] text-[var(--bull)]",
        warning: "border-transparent bg-[var(--neutral-dim)] text-[var(--neutral)]",
        outline: "border-[var(--border)] text-[var(--text)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
