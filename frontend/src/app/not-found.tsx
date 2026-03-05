import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center">
      <div className="text-6xl font-bold text-[var(--text-faint)]">404</div>
      <div>
        <h2 className="text-lg font-semibold text-[var(--text)]">Page not found</h2>
        <p className="mt-1 text-sm text-[var(--text-muted)]">
          The page you&apos;re looking for doesn&apos;t exist.
        </p>
      </div>
      <Link
        href="/dashboard"
        className="inline-flex items-center gap-2 rounded-[var(--radius)] bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[var(--accent-hover)]"
      >
        Go to Dashboard
      </Link>
    </div>
  );
}
