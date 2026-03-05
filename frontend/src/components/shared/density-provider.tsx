"use client";

import * as React from "react";
import { DensityProvider as InternalDensityProvider } from "@/hooks/use-density";

export function DensityProvider({ children }: { children: React.ReactNode }) {
  return <InternalDensityProvider>{children}</InternalDensityProvider>;
}
