import type { Metadata } from "next";
import { BacktestContent } from "./backtest-content";

export const metadata: Metadata = { title: "Backtest" };

export default function BacktestPage() {
  return <BacktestContent />;
}
