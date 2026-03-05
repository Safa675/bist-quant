import { Skeleton } from "@/components/ui/skeleton";

export default function ScreenerLoading() {
  return (
    <div className="grid grid-cols-4 gap-6 p-6">
      {/* Filter sidebar */}
      <div className="col-span-1 flex flex-col gap-4">
        <Skeleton className="h-8 w-full" />
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-10 w-full" />
        ))}
        <Skeleton className="h-10 w-full mt-2" />
      </div>

      {/* Results table */}
      <div className="col-span-3 flex flex-col gap-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-[calc(100vh-12rem)] border border-[var(--border)]" />
      </div>
    </div>
  );
}
