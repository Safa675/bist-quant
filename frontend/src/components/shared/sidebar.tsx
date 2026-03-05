"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Activity,
  FlaskConical,
  GitCommit,
  Filter,
  LineChart,
  Target,
  Briefcase,
  ShieldCheck,
  Bot,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "@/config/nav";
import { RegimeBadge } from "@/components/shared/regime-badge";
import { DensityToggle } from "@/components/shared/density-toggle";

const ICON_MAP: Record<string, React.ElementType> = {
  LayoutDashboard,
  Activity,
  Beaker: FlaskConical,
  FlaskConical,
  GitCommit,
  Filter,
  LineChart,
  Target,
  Briefcase,
  ShieldCheck,
  Bot,
};

function NavIcon({ name }: { name: string }) {
  const Icon = ICON_MAP[name] ?? LayoutDashboard;
  return <Icon className="h-4 w-4 shrink-0" />;
}

interface SidebarProps {
  regimeLabel?: string;
}

export function Sidebar({ regimeLabel }: SidebarProps) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = React.useState(false);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  // Persist collapsed state in localStorage
  React.useEffect(() => {
    const stored = localStorage.getItem("sidebar-collapsed");
    if (stored !== null) setCollapsed(stored === "true");
  }, []);

  // Close mobile drawer on route change
  React.useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  const toggle = () => {
    setCollapsed((c) => {
      const next = !c;
      localStorage.setItem("sidebar-collapsed", String(next));
      return next;
    });
  };

  const navContent = (
    <>
      {/* Brand */}
      <div className="flex h-14 items-center gap-3 border-b border-[var(--color-glass-border)] bg-[rgba(255,255,255,0.03)] px-4">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-[linear-gradient(135deg,var(--accent),var(--purple))] shadow-[0_0_14px_rgba(74,158,255,0.35)]">
          <LineChart className="h-4 w-4 text-white" />
        </div>
        {!collapsed && (
          <span className="truncate text-sm font-semibold bg-[linear-gradient(90deg,#ffffff,var(--text-muted))] bg-clip-text text-transparent">BIST Quant</span>
        )}
      </div>

      {/* Regime badge */}
      {regimeLabel && !collapsed && (
        <div className="px-4 pt-3 pb-1">
          <RegimeBadge label={regimeLabel} className="w-full justify-center" />
        </div>
      )}
      {regimeLabel && collapsed && (
        <div className="flex justify-center pt-3 pb-1" title={`Regime: ${regimeLabel}`}>
          <RegimeBadge label={regimeLabel} className="px-1" />
        </div>
      )}

      {/* Navigation */}
      <nav aria-label="Main navigation" className="flex-1 overflow-y-auto overflow-x-hidden py-3">
        <ul className="space-y-0.5 px-2">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  title={collapsed ? item.title : undefined}
                  className={cn(
                    "group/nav relative flex items-center gap-3 rounded-[var(--radius-sm)] px-2.5 py-2 text-sm transition-all duration-200",
                    isActive
                      ? "bg-[linear-gradient(90deg,var(--accent-dim),rgba(143,134,255,0.14))] text-[var(--text)] ring-1 ring-[var(--accent)]/40 shadow-[0_0_18px_rgba(74,158,255,0.12)]"
                      : "text-[var(--text-muted)] hover:bg-[var(--color-glass-strong)] hover:text-[var(--text)] hover:shadow-[0_0_12px_rgba(74,158,255,0.06)] hover:translate-x-[2px]"
                  )}
                >
                  <NavIcon name={item.icon} />
                  {!collapsed && (
                    <span className="truncate">{item.title}</span>
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer controls */}
      <div className="border-t border-[var(--color-glass-border)] p-2 space-y-1.5">
        <DensityToggle compactLabel={collapsed} />
        <button
          onClick={toggle}
          className={cn(
            "hidden md:flex w-full items-center rounded-[var(--radius-sm)] p-2 text-[var(--text-muted)] hover:bg-[var(--color-glass-strong)] hover:text-[var(--text)] transition-all duration-200",
            collapsed ? "justify-center" : "justify-between",
          )}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {!collapsed && <span className="text-xs font-medium">Collapse</span>}
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </button>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile hamburger button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="fixed left-3 top-3 z-50 flex h-9 w-9 items-center justify-center rounded-[var(--radius-sm)] border border-[var(--border)] bg-[var(--bg-panel)] text-[var(--text-muted)] shadow-[var(--shadow-sm)] md:hidden"
        aria-label="Open navigation"
      >
        <Menu className="h-5 w-5" />
      </button>

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile drawer */}
      <aside
        aria-label="Mobile navigation"
        className={cn(
          "fixed left-0 top-0 z-50 flex h-screen w-64 flex-col border-r border-[var(--color-glass-border)] bg-[linear-gradient(180deg,var(--surface-1),var(--surface-0))] backdrop-blur-[18px] transition-transform duration-300 ease-out md:hidden",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <button
          onClick={() => setMobileOpen(false)}
          className="absolute right-3 top-3 p-1 text-[var(--text-muted)] hover:text-[var(--text)]"
          aria-label="Close navigation"
        >
          <X className="h-5 w-5" />
        </button>
        {navContent}
      </aside>

      {/* Desktop sidebar */}
      <aside
        aria-label="Main navigation"
        style={{
          width: collapsed ? "var(--sidebar-w-collapsed)" : "var(--sidebar-w)",
          transition: "width 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        }}
        className="hidden md:flex fixed left-0 top-0 z-40 h-screen flex-col border-r border-[var(--color-glass-border)] bg-[linear-gradient(180deg,rgba(17,26,49,0.92),rgba(13,23,48,0.96))] backdrop-blur-[18px] shadow-[var(--shadow),8px_0_32px_rgba(0,0,0,0.3)] overflow-hidden"
      >
        {navContent}
      </aside>
    </>
  );
}
