import * as React from "react";
import { cn } from "@/lib/utils";
import { regimeColor } from "@/lib/tokens";

interface RegimeBadgeProps {
  label: string;
  className?: string;
}

export function RegimeBadge({ label, className }: RegimeBadgeProps) {
  const color = regimeColor(label);

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold",
        className
      )}
      style={{
        color,
        borderColor: `${color}40`,
        backgroundColor: `${color}18`,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}
