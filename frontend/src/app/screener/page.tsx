import type { Metadata } from "next";
import { ScreenerContent } from "./screener-content";
export const metadata: Metadata = { title: "Screener" };
export default function ScreenerPage() {
  return <ScreenerContent />;
}
