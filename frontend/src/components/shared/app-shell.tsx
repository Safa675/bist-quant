"use client";

import * as React from "react";
import { Sidebar } from "@/components/shared/sidebar";
import { PageTransition } from "@/components/shared/page-transition";

/**
 * Shell wraps the Sidebar with live regime data fetched client-side.
 * Keeping the actual root layout as a server component is not possible
 * with the Sidebar already being "use client", so this is the thin
 * client boundary.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const [regimeLabel, setRegimeLabel] = React.useState<string | undefined>();

  React.useEffect(() => {
    const apiBase =
      process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8001";

    fetch(`${apiBase}/api/dashboard/overview?lookback=30`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data?.regime?.label) {
          setRegimeLabel(data.regime.label);
        }
      })
      .catch(() => {
        // Regime display is best-effort
      });
  }, []);

  return (
    <div className="app-shell" data-ui-app-shell>
      {/* Subtle noise texture for depth */}
      <div className="noise-overlay" aria-hidden />
      <Sidebar regimeLabel={regimeLabel} />
      <main className="app-main" id="main-content" tabIndex={-1}>
        <div className="page-content">
          <PageTransition>{children}</PageTransition>
        </div>
      </main>
    </div>
  );
}
