import type { Metadata } from "next";
import { ComplianceContent } from "./compliance-content";
export const metadata: Metadata = { title: "Compliance" };
export default function CompliancePage() {
  return <ComplianceContent />;
}
