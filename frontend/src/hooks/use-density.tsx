"use client";

import * as React from "react";
import type { UiDensity } from "@/lib/types";

export interface DensityContextValue {
  density: UiDensity;
  setDensity: (value: UiDensity) => void;
  toggleDensity: () => void;
}

const DensityContext = React.createContext<DensityContextValue | null>(null);

const STORAGE_KEY = "bq-ui-density";

function getStoredDensity(): UiDensity {
  if (typeof window === "undefined") return "comfortable";
  const value = window.localStorage.getItem(STORAGE_KEY);
  return value === "compact" ? "compact" : "comfortable";
}

export function DensityProvider({ children }: { children: React.ReactNode }) {
  const [density, setDensity] = React.useState<UiDensity>("comfortable");

  React.useEffect(() => {
    setDensity(getStoredDensity());
  }, []);

  React.useEffect(() => {
    document.documentElement.setAttribute("data-density", density);
    window.localStorage.setItem(STORAGE_KEY, density);
  }, [density]);

  const value = React.useMemo<DensityContextValue>(
    () => ({
      density,
      setDensity,
      toggleDensity: () =>
        setDensity((curr) => (curr === "comfortable" ? "compact" : "comfortable")),
    }),
    [density]
  );

  return <DensityContext.Provider value={value}>{children}</DensityContext.Provider>;
}

export function useDensity(): DensityContextValue {
  const ctx = React.useContext(DensityContext);
  if (!ctx) {
    throw new Error("useDensity must be used within DensityProvider");
  }
  return ctx;
}
