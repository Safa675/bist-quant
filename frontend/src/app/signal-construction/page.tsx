import type { Metadata } from "next";
import { SignalConstructionContent } from "./signal-construction-content";
export const metadata: Metadata = { title: "Signal Construction" };
export default function SignalConstructionPage() {
  return <SignalConstructionContent />;
}
