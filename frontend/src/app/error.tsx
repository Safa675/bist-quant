"use client";

import { useEffect } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { scaleIn } from "@/lib/animations";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Route error:", error);
  }, [error]);

  return (
    <motion.div
      variants={scaleIn}
      initial="hidden"
      animate="show"
      className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center"
    >
      <motion.div
        initial={{ rotate: 0 }}
        animate={{ rotate: [0, -8, 8, -4, 4, 0] }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="flex h-14 w-14 items-center justify-center rounded-full bg-[var(--bear-dim)]"
      >
        <AlertTriangle className="h-7 w-7 text-[var(--bear)]" />
      </motion.div>
      <div>
        <h2 className="text-lg font-semibold text-[var(--text)]">Something went wrong</h2>
        <p className="mt-1 max-w-md text-sm text-[var(--text-muted)]">
          {error.message || "An unexpected error occurred while loading this page."}
        </p>
      </div>
      <Button onClick={reset} variant="glass" className="gap-2">
        <RotateCcw className="h-4 w-4" />
        Try again
      </Button>
    </motion.div>
  );
}
