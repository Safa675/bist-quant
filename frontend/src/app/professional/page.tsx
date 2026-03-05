import type { Metadata } from "next";
import { ProfessionalContent } from "./professional-content";
export const metadata: Metadata = { title: "Professional" };
export default function ProfessionalPage() {
  return <ProfessionalContent />;
}
