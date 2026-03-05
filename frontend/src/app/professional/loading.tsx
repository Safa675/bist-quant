import { Skeleton } from "@/components/ui/skeleton";

export default function ProfessionalLoading() {
  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Tab bar */}
      <div className="flex gap-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-9 w-28" />
        ))}
      </div>

      {/* 3-column content */}
      <div className="grid grid-cols-3 gap-6">
        <Skeleton className="h-72 border border-[var(--border)]" />
        <Skeleton className="h-72 border border-[var(--border)]" />
        <Skeleton className="h-72 border border-[var(--border)]" />
      </div>
    </div>
  );
}
