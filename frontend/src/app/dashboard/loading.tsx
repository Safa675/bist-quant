import { Skeleton } from "@/components/ui/skeleton";

export default function DashboardLoading() {
  return (
    <div className="flex flex-col gap-6 p-6">
      {/* KPI row — 3 cards */}
      <div className="grid grid-cols-3 gap-6">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-28 rounded-[var(--radius)] border border-[var(--border)]" />
        ))}
      </div>

      {/* 2 chart placeholders */}
      <div className="grid grid-cols-2 gap-6">
        <Skeleton className="h-72 border border-[var(--border)]" />
        <Skeleton className="h-72 border border-[var(--border)]" />
      </div>

      {/* Table skeleton */}
      <Skeleton className="h-64 border border-[var(--border)]" />
    </div>
  );
}
