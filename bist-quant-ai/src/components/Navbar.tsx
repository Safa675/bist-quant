"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, Bot, FlaskConical, SlidersHorizontal, TrendingUp } from "lucide-react";

export default function Navbar() {
    const pathname = usePathname();

    const isActive = (path: string) => pathname === path;

    return (
        <nav className="nav-glass">
            <div
                style={{
                    maxWidth: 1400,
                    margin: "0 auto",
                    padding: "0 24px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    height: 64,
                }}
            >
                {/* Logo */}
                <Link
                    href="/"
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        textDecoration: "none",
                        color: "var(--text-primary)",
                    }}
                >
                    <div
                        style={{
                            width: 36,
                            height: 36,
                            borderRadius: "var(--radius-md)",
                            background: "var(--gradient-accent)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        <TrendingUp size={20} color="#fff" />
                    </div>
                    <span
                        style={{
                            fontSize: "1.15rem",
                            fontWeight: 700,
                            letterSpacing: "-0.02em",
                        }}
                    >
                        Quant<span className="gradient-text">AI</span>
                    </span>
                </Link>

                {/* Navigation Links */}
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    {[
                        { href: "/", label: "Home", icon: Activity },
                        { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
                        { href: "/signal-construction", label: "Signal Builder", icon: SlidersHorizontal },
                        { href: "/factor-lab", label: "Factor Lab", icon: FlaskConical },
                        { href: "/dashboard#agents", label: "AI Agents", icon: Bot },
                    ].map((item) => (
                        <Link
                            key={item.href}
                            href={item.href}
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                padding: "8px 16px",
                                borderRadius: "var(--radius-md)",
                                fontSize: "0.875rem",
                                fontWeight: 500,
                                textDecoration: "none",
                                color: isActive(item.href)
                                    ? "var(--accent-emerald)"
                                    : "var(--text-secondary)",
                                background: isActive(item.href)
                                    ? "var(--accent-emerald-dim)"
                                    : "transparent",
                                transition: "all var(--transition-fast)",
                            }}
                        >
                            <item.icon size={16} />
                            {item.label}
                        </Link>
                    ))}

                    <Link href="/dashboard" className="btn-primary" style={{ marginLeft: 12, padding: "8px 20px", fontSize: "0.85rem" }}>
                        Launch App
                    </Link>
                </div>
            </div>
        </nav>
    );
}
