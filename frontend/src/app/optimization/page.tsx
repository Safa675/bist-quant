import type { Metadata } from "next";
import { OptimizationContent } from "./optimization-content";
export const metadata: Metadata = { title: "Optimization" };
export default function OptimizationPage() {
  return <OptimizationContent />;
}
