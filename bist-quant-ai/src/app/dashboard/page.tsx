"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import SignalTable from "@/components/SignalTable";
import EquityChart from "@/components/EquityChart";
import RegimeIndicator from "@/components/RegimeIndicator";
import PortfolioView from "@/components/PortfolioView";
import AgentChat from "@/components/AgentChat";
import PublishedSignalManager from "@/components/PublishedSignalManager";
import { BarChart3, TrendingUp, Shield, Activity, Clock } from "lucide-react";

/* ---------- Types ---------- */

interface Signal {
    name: string;
    enabled: boolean;
    cagr: number;
    sharpe: number;
    beta?: number | null;
    max_dd: number;
    ytd: number;
    last_rebalance: string;
}

interface DashboardData {
    last_update: string;
    current_regime: string;
    regime_distribution: Record<string, number>;
    xu100_ytd: number;
    trading_days: number;
    active_signals: number;
    displayed_signals: number;
    signals: Signal[];
    holdings: Record<string, string[]>;
}

/* ---------- Component ---------- */

export default function DashboardPage() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [selectedSignal, setSelectedSignal] = useState<string>("breakout_value");
    const [activeTab, setActiveTab] = useState<"signals" | "chart" | "holdings">("signals");

    const loadDashboardData = useCallback(async () => {
        try {
            const response = await fetch("/api/dashboard", { cache: "no-store" });
            if (!response.ok) {
                throw new Error(`Dashboard API failed (${response.status})`);
            }

            const payload = (await response.json()) as DashboardData;
            setData(payload);
            setSelectedSignal((prev) => {
                const signalNames = new Set(payload.signals.map((signal) => signal.name));
                if (signalNames.has(prev)) return prev;
                return payload.signals[0]?.name || prev;
            });
        } catch (error) {
            console.error(error);
        }
    }, []);

    useEffect(() => {
        void loadDashboardData();
    }, [loadDashboardData]);

    if (!data) {
        return (
            <>
                <Navbar />
                <div
                    style={{
                        minHeight: "100vh",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        background: "var(--bg-primary)",
                        paddingTop: 64,
                    }}
                >
                    <div style={{ textAlign: "center" }}>
                        <div
                            style={{
                                width: 48,
                                height: 48,
                                borderRadius: "50%",
                                border: "3px solid var(--accent-emerald-dim)",
                                borderTopColor: "var(--accent-emerald)",
                                animation: "spin 1s linear infinite",
                                margin: "0 auto 16px",
                            }}
                        />
                        <p style={{ color: "var(--text-secondary)" }}>Loading dashboard data...</p>
                        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                    </div>
                </div>
            </>
        );
    }

    // Compute aggregate stats
    const topSignal = data.signals[0] ?? {
        name: "n/a",
        enabled: true,
        cagr: 0,
        sharpe: 0,
        max_dd: 0,
        ytd: 0,
        last_rebalance: "n/a",
    };
    const avgCagr = data.signals.length
        ? (data.signals.reduce((a, s) => a + s.cagr, 0) / data.signals.length).toFixed(1)
        : "0.0";
    const avgSharpe = data.signals.length
        ? (data.signals.reduce((a, s) => a + s.sharpe, 0) / data.signals.length).toFixed(2)
        : "0.00";

    return (
        <>
            <Navbar />

            <main
                style={{
                    paddingTop: 100,
                    paddingBottom: 80,
                    minHeight: "100vh",
                    background: "var(--bg-primary)",
                }}
            >
                <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 24px" }}>

                    {/* ===== HEADER ===== */}
                    <div style={{ marginBottom: 32, display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 16 }}>
                        <div>
                            <h1
                                style={{
                                    fontSize: "1.75rem",
                                    fontWeight: 800,
                                    letterSpacing: "-0.02em",
                                    marginBottom: 6,
                                }}
                            >
                                Trading Dashboard
                            </h1>
                            <div style={{ display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
                                <RegimeIndicator regime={data.current_regime} />
                                <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem", color: "var(--text-muted)" }}>
                                    <Clock size={13} />
                                    Updated {data.last_update}
                                </span>
                            </div>
                        </div>
                        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                            <Link href="/factor-lab" style={{ border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", padding: "9px 14px", fontSize: "0.82rem", color: "var(--text-primary)", textDecoration: "none", background: "var(--bg-secondary)" }}>
                                Factor Lab
                            </Link>
                            <Link href="/signal-construction" className="btn-primary" style={{ padding: "10px 16px", fontSize: "0.82rem" }}>
                                Build Signals
                            </Link>
                        </div>
                    </div>

                    {/* ===== STAT CARDS ===== */}
                    <div
                        style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                            gap: 16,
                            marginBottom: 32,
                        }}
                    >
                        {[
                            {
                                icon: TrendingUp,
                                label: "Top Signal CAGR",
                                value: `${topSignal.cagr}%`,
                                sub: topSignal.name.replace(/_/g, " "),
                                positive: true,
                            },
                            {
                                icon: BarChart3,
                                label: "Avg. CAGR",
                                value: `${avgCagr}%`,
                                sub: `Across ${data.active_signals} signals`,
                                positive: true,
                            },
                            {
                                icon: Shield,
                                label: "Avg. Sharpe Ratio",
                                value: avgSharpe,
                                sub: "Risk-adjusted return",
                                positive: parseFloat(avgSharpe) > 1.5,
                            },
                            {
                                icon: Activity,
                                label: "XU100 YTD",
                                value: `${data.xu100_ytd.toFixed(2)}%`,
                                sub: "Benchmark performance",
                                positive: data.xu100_ytd > 0,
                            },
                        ].map((card, i) => (
                            <div key={i} className="glass-card" style={{ padding: 20 }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                                    <div className="metric-label">{card.label}</div>
                                    <div
                                        style={{
                                            width: 32,
                                            height: 32,
                                            borderRadius: "var(--radius-sm)",
                                            background: "var(--accent-emerald-dim)",
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                        }}
                                    >
                                        <card.icon size={16} color="var(--accent-emerald)" />
                                    </div>
                                </div>
                                <div className={`metric-value ${card.positive ? "metric-positive" : "metric-negative"}`}>
                                    {card.value}
                                </div>
                                <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginTop: 4, textTransform: "capitalize" }}>
                                    {card.sub}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* ===== TAB BAR ===== */}
                    <div
                        style={{
                            display: "flex",
                            gap: 4,
                            marginBottom: 24,
                            background: "var(--bg-secondary)",
                            padding: 4,
                            borderRadius: "var(--radius-md)",
                            width: "fit-content",
                        }}
                    >
                        {(["signals", "chart", "holdings"] as const).map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setActiveTab(tab)}
                                style={{
                                    padding: "8px 20px",
                                    fontSize: "0.85rem",
                                    fontWeight: 600,
                                    borderRadius: "var(--radius-sm)",
                                    border: "none",
                                    cursor: "pointer",
                                    background: activeTab === tab ? "var(--accent-emerald-dim)" : "transparent",
                                    color: activeTab === tab ? "var(--accent-emerald)" : "var(--text-muted)",
                                    transition: "all var(--transition-fast)",
                                    textTransform: "capitalize",
                                }}
                            >
                                {tab === "signals" ? "📊 Signal Performance" : tab === "chart" ? "📈 Equity Curves" : "📋 Current Holdings"}
                            </button>
                        ))}
                    </div>

                    {/* ===== MAIN CONTENT GRID ===== */}
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 24, alignItems: "start" }}>

                        {/* Left panel */}
                        <div>
                            {activeTab === "signals" && (
                                <SignalTable
                                    signals={data.signals}
                                    selectedSignal={selectedSignal}
                                    onSelectSignal={setSelectedSignal}
                                />
                            )}

                            {activeTab === "chart" && <EquityChart />}

                            {activeTab === "holdings" && (
                                <PortfolioView holdings={data.holdings} selectedSignal={selectedSignal} onSelectSignal={setSelectedSignal} />
                            )}
                        </div>

                        {/* Right panel — AI Agent Chat */}
                        <div id="agents" style={{ position: "sticky", top: 88, display: "grid", gap: 16 }}>
                            <PublishedSignalManager onChanged={loadDashboardData} />
                            <AgentChat holdings={data.holdings} signals={data.signals} regime={data.current_regime} />
                        </div>
                    </div>

                    {/* ===== REGIME DISTRIBUTION ===== */}
                    <div style={{ marginTop: 32 }}>
                        <div className="glass-card" style={{ padding: 24 }}>
                            <h3 style={{ fontSize: "0.95rem", fontWeight: 700, marginBottom: 16 }}>
                                Historical Regime Distribution
                            </h3>
                            <div style={{ display: "flex", gap: 2, borderRadius: "var(--radius-sm)", overflow: "hidden", height: 32 }}>
                                {Object.entries(data.regime_distribution).map(([regime, pct]) => {
                                    const colors: Record<string, string> = {
                                        Bull: "var(--accent-emerald)",
                                        Bear: "var(--accent-rose)",
                                        Recovery: "var(--accent-amber)",
                                        Stress: "var(--accent-violet)",
                                    };
                                    return (
                                        <div
                                            key={regime}
                                            style={{
                                                width: `${pct}%`,
                                                background: colors[regime] || "var(--text-muted)",
                                                opacity: 0.7,
                                                display: "flex",
                                                alignItems: "center",
                                                justifyContent: "center",
                                                fontSize: "0.7rem",
                                                fontWeight: 700,
                                                color: "#fff",
                                                minWidth: pct > 8 ? "auto" : 0,
                                            }}
                                            title={`${regime}: ${pct.toFixed(1)}%`}
                                        >
                                            {pct > 12 ? `${regime} ${pct.toFixed(0)}%` : ""}
                                        </div>
                                    );
                                })}
                            </div>
                            <div style={{ display: "flex", gap: 20, marginTop: 12, flexWrap: "wrap" }}>
                                {Object.entries(data.regime_distribution).map(([regime, pct]) => {
                                    const colors: Record<string, string> = {
                                        Bull: "var(--accent-emerald)",
                                        Bear: "var(--accent-rose)",
                                        Recovery: "var(--accent-amber)",
                                        Stress: "var(--accent-violet)",
                                    };
                                    return (
                                        <div key={regime} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                                            <div style={{ width: 10, height: 10, borderRadius: 3, background: colors[regime] }} />
                                            {regime}: {pct.toFixed(1)}%
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
}
