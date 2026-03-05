"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { staggerContainer, fadeUp } from "@/lib/animations";

/* ── Stagger container ── */

interface StaggerRevealProps {
  children: React.ReactNode;
  className?: string;
  /** Additional delay before children start staggering (seconds) */
  delay?: number;
  /** Time between each child animation (seconds) */
  stagger?: number;
}

export function StaggerReveal({
  children,
  className,
  delay = 0,
  stagger = 0.06,
}: StaggerRevealProps) {
  return (
    <motion.div
      variants={staggerContainer(stagger, delay)}
      initial="hidden"
      animate="show"
      className={className}
    >
      {children}
    </motion.div>
  );
}

/* ── Stagger item (child of StaggerReveal) ── */

interface StaggerItemProps {
  children: React.ReactNode;
  className?: string;
}

export function StaggerItem({ children, className }: StaggerItemProps) {
  return (
    <motion.div variants={fadeUp} className={className}>
      {children}
    </motion.div>
  );
}
