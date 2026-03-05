import type { Metadata } from "next";
import { AppShell } from "@/components/shared/app-shell";
import { DensityProvider } from "@/components/shared/density-provider";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    template: "%s | BIST Quant",
    default: "BIST Quant Research Cockpit",
  },
  description: "Quantitative research cockpit for BIST equity markets",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-density="comfortable">
      <body>
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-[var(--radius)] focus:bg-[var(--accent)] focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-white focus:outline-none"
        >
          Skip to main content
        </a>
        <DensityProvider>
          <AppShell>{children}</AppShell>
        </DensityProvider>
      </body>
    </html>
  );
}
