import type { Metadata } from "next";
import { FactorLabContent } from "./factor-lab-content";

export const metadata: Metadata = { title: "Factor Lab" };

export default function FactorLabPage() {
  return <FactorLabContent />;
}
