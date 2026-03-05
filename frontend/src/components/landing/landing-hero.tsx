"use client";

import * as React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { staggerContainer, fadeUp } from "@/lib/animations";
import {
  LayoutDashboard,
  Activity,
  FlaskConical,
  Filter,
  LineChart,
  Target,
  Briefcase,
  ShieldCheck,
  Bot,
  ArrowRight,
} from "lucide-react";
import { ParticleField } from "./particles";

/* ── Feature showcase data ── */

const features = [
  {
    icon: LayoutDashboard,
    title: "Dashboard",
    desc: "Real-time market overview with regime detection and macro indicators.",
  },
  {
    icon: Activity,
    title: "Backtest Engine",
    desc: "Rigorously test strategies with walk-forward analysis and Monte Carlo.",
  },
  {
    icon: FlaskConical,
    title: "Factor Lab",
    desc: "Research and construct alpha factors with cross-sectional analysis.",
  },
  {
    icon: Filter,
    title: "Screener",
    desc: "Multi-criteria stock screening with fundamental and technical filters.",
  },
  {
    icon: LineChart,
    title: "Analytics",
    desc: "Portfolio attribution, risk decomposition, and performance analytics.",
  },
  {
    icon: Target,
    title: "Optimization",
    desc: "Mean-variance and Black-Litterman portfolio optimization toolkit.",
  },
  {
    icon: Briefcase,
    title: "Professional",
    desc: "Research reports, compliance dashboards, and institutional workflows.",
  },
  {
    icon: ShieldCheck,
    title: "Compliance",
    desc: "Pre-trade checks, exposure limits, and regulatory constraint monitoring.",
  },
  {
    icon: Bot,
    title: "AI Agents",
    desc: "Autonomous research agents for signal mining and data processing.",
  },
];

/* ── Framer Motion variants (from shared library) ── */

const containerVariants = staggerContainer(0.08, 0.5);
const itemVariants = fadeUp;

/* ── Landing hero component ── */

export function LandingHero() {
  return (
    <div className="fixed inset-0 z-50 flex flex-col overflow-y-auto bg-[var(--bg-base)]">
      {/* Background radial gradients */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background: [
            "radial-gradient(ellipse 80% 50% at 50% -10%, rgba(74,158,255,0.2) 0%, transparent 60%)",
            "radial-gradient(ellipse 60% 40% at 80% 60%, rgba(143,134,255,0.12) 0%, transparent 50%)",
            "radial-gradient(ellipse 50% 30% at 10% 80%, rgba(74,158,255,0.08) 0%, transparent 50%)",
          ].join(", "),
        }}
      />

      {/* Animated particle canvas */}
      <ParticleField />

      {/* ── Hero content ── */}
      <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="flex flex-col items-center text-center"
        >
          {/* Logo badge */}
          <motion.div
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="mb-6 flex h-16 w-16 items-center justify-center rounded-[var(--radius-xl)] bg-[linear-gradient(135deg,var(--accent),var(--purple))] shadow-[0_0_40px_rgba(74,158,255,0.35)]"
          >
            <LineChart className="h-8 w-8 text-white" />
          </motion.div>

          {/* Title */}
          <h1 className="text-display bg-[linear-gradient(90deg,#ffffff_10%,var(--accent)_50%,var(--purple)_90%)] bg-clip-text text-transparent drop-shadow-[0_2px_20px_rgba(74,158,255,0.3)]">
            BIST Quant
          </h1>

          {/* Subtitle */}
          <p className="mt-4 max-w-xl text-lg leading-relaxed text-[var(--text-muted)]">
            Quantitative research cockpit for BIST equity markets —{" "}
            <span className="text-gradient-accent font-medium">
              factor models, backtesting, regime detection
            </span>{" "}
            and portfolio optimization in one unified platform.
          </p>

          {/* CTA button */}
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.35 }}
            className="mt-8"
          >
            <Link
              href="/dashboard"
              className="group inline-flex items-center gap-2 rounded-[var(--radius-lg)] bg-[linear-gradient(135deg,var(--accent),var(--purple))] px-8 py-3.5 text-sm font-semibold text-white shadow-[0_0_30px_rgba(74,158,255,0.3)] transition-all duration-200 hover:shadow-[0_0_44px_rgba(74,158,255,0.45)] hover:scale-[1.03] active:scale-[0.98]"
            >
              Enter Dashboard
              <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-1" />
            </Link>
          </motion.div>
        </motion.div>

        {/* ── Feature showcase grid ── */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="mx-auto mt-20 grid max-w-5xl grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3"
        >
          {features.map((f) => (
            <motion.div
              key={f.title}
              variants={itemVariants}
              className="group relative rounded-[var(--radius-lg)] border border-[var(--color-glass-border)] bg-[linear-gradient(145deg,rgba(255,255,255,0.07),rgba(148,163,184,0.04)_42%,rgba(15,23,42,0.15)_100%)] backdrop-blur-[14px] p-6 shadow-[var(--color-frost-card-shadow)] transition-all duration-250 ease-out hover:border-[var(--color-glass-border-accent)] hover:shadow-[var(--color-frost-card-shadow),0_0_30px_rgba(74,158,255,0.08)] hover:-translate-y-[2px]"
            >
              {/* Inner glow */}
              <div
                aria-hidden
                className="pointer-events-none absolute inset-0 rounded-[inherit] bg-[linear-gradient(145deg,rgba(255,255,255,0.05),rgba(255,255,255,0.01)_40%,transparent_72%)]"
              />
              <div className="relative z-[1]">
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-[var(--radius)] bg-[var(--accent-dim)] text-[var(--accent)] transition-colors duration-200 group-hover:bg-[rgba(74,158,255,0.25)] group-hover:shadow-[0_0_16px_rgba(74,158,255,0.2)]">
                  <f.icon className="h-5 w-5" />
                </div>
                <h3 className="text-h3 text-[var(--text)]">{f.title}</h3>
                <p className="mt-1.5 text-small leading-relaxed text-[var(--text-muted)]">
                  {f.desc}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Footer */}
        <div className="mt-16 pb-8 text-center text-micro text-[var(--text-faint)] opacity-60">
          Built for BIST equity research
        </div>
      </div>
    </div>
  );
}
