import { Skeleton } from "@/components/ui/skeleton";

export default function AnalyticsLoading() {
  return (
    <div className="grid grid-cols-4 gap-6 p-6">
      {/* Input panel */}
      <div className="col-span-1 flex flex-col gap-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full mt-auto" />
      </div>

      {/* Chart area */}
      <div className="col-span-3 flex flex-col gap-6">
        <Skeleton className="h-72 border border-[var(--border)]" />
        <div className="grid grid-cols-2 gap-6">
          <Skeleton className="h-56 border border-[var(--border)]" />
          <Skeleton className="h-56 border border-[var(--border)]" />
        </div>
      </div>
    </div>
  );
}
