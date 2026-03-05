import * as React from "react";
import { cn } from "@/lib/utils";

function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-[var(--radius)] bg-[var(--bg-elevated)]",
        className
      )}
      {...props}
    >
      {/* Shimmer sweep */}
      <div className="shimmer" />
    </div>
  );
}

export { Skeleton };
