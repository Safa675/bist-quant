"use client";

import * as React from "react";
import { useDensity } from "@/hooks/use-density";
import { Button } from "@/components/ui/button";
import { Expand, Shrink } from "lucide-react";
import { cn } from "@/lib/utils";

interface DensityToggleProps {
  compactLabel?: boolean;
}

export function DensityToggle({ compactLabel = false }: DensityToggleProps) {
  const { density, toggleDensity } = useDensity();
  const isCompact = density === "compact";

  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      onClick={toggleDensity}
      className={cn(
        "text-[var(--text-muted)] hover:text-[var(--text)]",
        compactLabel ? "w-full justify-center px-0" : "w-full justify-start gap-2",
      )}
      aria-label={isCompact ? "Switch to comfortable density" : "Switch to compact density"}
      data-ui-density-toggle
    >
      {isCompact ? <Expand className="h-4 w-4" /> : <Shrink className="h-4 w-4" />}
      {!compactLabel && <span>{isCompact ? "Comfortable density" : "Compact density"}</span>}
    </Button>
  );
}
