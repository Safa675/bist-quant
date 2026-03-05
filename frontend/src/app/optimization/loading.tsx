import { Skeleton } from "@/components/ui/skeleton";

export default function OptimizationLoading() {
  return (
    <div className="grid grid-cols-4 gap-6 p-6">
      {/* Config panel */}
      <div className="col-span-1 flex flex-col gap-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-10 w-full mt-auto" />
      </div>

      {/* Heatmap / results */}
      <div className="col-span-3 flex flex-col gap-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-80 border border-[var(--border)]" />
        <Skeleton className="h-48 border border-[var(--border)]" />
      </div>
    </div>
  );
}
