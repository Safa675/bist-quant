import { Skeleton } from "@/components/ui/skeleton";

export default function SignalConstructionLoading() {
  return (
    <div className="grid grid-cols-4 gap-6 p-6">
      {/* Config sidebar */}
      <Skeleton className="col-span-1 h-[calc(100vh-8rem)] border border-[var(--border)]" />

      {/* Main area */}
      <div className="col-span-3 flex flex-col gap-6">
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-72 border border-[var(--border)]" />
        <Skeleton className="h-56 border border-[var(--border)]" />
      </div>
    </div>
  );
}
