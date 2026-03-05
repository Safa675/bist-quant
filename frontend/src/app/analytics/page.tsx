import type { Metadata } from "next";
import { AnalyticsContent } from "./analytics-content";
export const metadata: Metadata = { title: "Analytics" };
export default function AnalyticsPage() {
  return <AnalyticsContent />;
}
