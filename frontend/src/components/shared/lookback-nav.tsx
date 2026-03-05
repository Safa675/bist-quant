"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const LOOKBACK_OPTIONS = [
  { label: "6M", value: 126 },
  { label: "1Y", value: 252 },
  { label: "2Y", value: 504 },
  { label: "3Y", value: 756 },
  { label: "5Y", value: 1260 },
] as const;

interface LookbackNavProps {
  value: number;
  onChange: (days: number) => void;
  className?: string;
}

export function LookbackNav({ value, onChange, className }: LookbackNavProps) {
  return (
    <div role="radiogroup" aria-label="Lookback period" className={cn("flex items-center gap-1", className)}>
      {LOOKBACK_OPTIONS.map((opt) => (
        <Button
          key={opt.value}
          variant={value === opt.value ? "default" : "ghost"}
          size="sm"
          onClick={() => onChange(opt.value)}
          role="radio"
          aria-checked={value === opt.value}
          aria-label={`${opt.label} lookback`}
          className="text-xs px-2.5 h-7"
        >
          {opt.label}
        </Button>
      ))}
    </div>
  );
}

export { LOOKBACK_OPTIONS };
