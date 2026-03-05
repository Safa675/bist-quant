import { Skeleton } from "@/components/ui/skeleton";

export default function AgentsLoading() {
  return (
    <div className="flex flex-col items-center gap-8 p-6">
      {/* Centered card */}
      <Skeleton className="h-32 w-full max-w-xl border border-[var(--border)]" />

      {/* 3-column grid */}
      <div className="grid w-full grid-cols-3 gap-6">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-40 border border-[var(--border)]" />
        ))}
      </div>
    </div>
  );
}
