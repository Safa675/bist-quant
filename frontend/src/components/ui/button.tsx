import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-[var(--radius)] text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent)] disabled:pointer-events-none disabled:opacity-55 cursor-pointer active:scale-[0.97]",
  {
    variants: {
      variant: {
        default:
          "bg-[var(--accent)] text-white shadow-sm hover:bg-[var(--accent-hover)] hover:shadow-[0_0_20px_rgba(74,158,255,0.35)] hover:scale-[1.02]",
        glass:
          "border border-[var(--color-glass-border)] bg-[var(--color-glass)] backdrop-blur-[12px] text-[var(--text)] hover:bg-[var(--color-glass-strong)] hover:border-[var(--color-glass-border-accent)] hover:shadow-[0_0_18px_rgba(74,158,255,0.15)] hover:scale-[1.02]",
        outline:
          "border border-[var(--color-glass-border)] bg-[var(--surface-1)] hover:bg-[var(--color-glass-strong)] hover:border-[var(--color-glass-border-accent)] hover:shadow-[0_0_14px_rgba(74,158,255,0.1)] text-[var(--text)]",
        ghost:
          "hover:bg-[var(--color-glass-strong)] text-[var(--text-muted)] hover:text-[var(--text)]",
        destructive:
          "bg-[var(--bear)] text-white hover:opacity-90 hover:shadow-[0_0_18px_rgba(255,109,118,0.3)] hover:scale-[1.02]",
        secondary:
          "bg-[var(--surface-2)] text-[var(--text)] hover:bg-[var(--bg-hover)] border border-[var(--color-glass-border)] hover:border-[var(--color-glass-border-accent)] hover:shadow-[0_0_12px_rgba(74,158,255,0.08)]",
        link: "text-[var(--accent)] underline-offset-4 hover:underline",
      },
      size: {
        default: "h-[var(--control-h)] px-4 py-2",
        sm: "h-[calc(var(--control-h)-6px)] rounded-[var(--radius-sm)] px-3 text-xs",
        lg: "h-11 rounded-[var(--radius-lg)] px-8",
        icon: "h-[var(--control-h)] w-[var(--control-h)]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
