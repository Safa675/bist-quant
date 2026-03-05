import { Skeleton } from "@/components/ui/skeleton";

export default function FactorLabLoading() {
  return (
    <div className="grid grid-cols-4 gap-6 p-6">
      {/* Factor table */}
      <div className="col-span-3 flex flex-col gap-4">
        <Skeleton className="h-10 w-64" />
        <Skeleton className="h-[calc(100vh-12rem)] border border-[var(--border)]" />
      </div>

      {/* Sidebar */}
      <div className="col-span-1 flex flex-col gap-4">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-48 border border-[var(--border)]" />
        <Skeleton className="h-48 border border-[var(--border)]" />
      </div>
    </div>
  );
}
