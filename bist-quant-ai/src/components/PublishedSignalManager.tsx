"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { Loader2, RefreshCw, ToggleLeft, ToggleRight, Trash2 } from "lucide-react";

interface PublishedSignal {
    id: string;
    name: string;
    enabled: boolean;
    updated_at: string;
    holdings: string[];
    signal: {
        name: string;
        cagr: number;
        sharpe: number;
        ytd: number;
    };
}

interface Props {
    onChanged?: () => void | Promise<void>;
}

function formatName(value: string): string {
    return value.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

function formatFixed(value: number | null | undefined, decimals = 2): string {
    if (typeof value !== "number" || !Number.isFinite(value)) return "0.00";
    return value.toFixed(decimals);
}

export default function PublishedSignalManager({ onChanged }: Props) {
    const [signals, setSignals] = useState<PublishedSignal[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [busyId, setBusyId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [status, setStatus] = useState<string | null>(null);

    const activeCount = useMemo(() => signals.filter((item) => item.enabled).length, [signals]);

    const loadSignals = async () => {
        setError(null);
        setLoading(true);
        try {
            const response = await fetch("/api/signal-construction/publish", { cache: "no-store" });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Failed (${response.status})`);
            }
            setSignals(Array.isArray(data.signals) ? (data.signals as PublishedSignal[]) : []);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load published signals.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void loadSignals();
    }, []);

    const toggleSignal = async (item: PublishedSignal) => {
        setBusyId(item.id);
        setError(null);
        setStatus(null);

        try {
            const response = await fetch("/api/signal-construction/publish", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: item.id, enabled: !item.enabled }),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Failed (${response.status})`);
            }

            await loadSignals();
            if (onChanged) await onChanged();
            setStatus(`${formatName(item.name)} ${item.enabled ? "disabled" : "enabled"}.`);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to update signal state.");
        } finally {
            setBusyId(null);
        }
    };

    const removeSignal = async (item: PublishedSignal) => {
        if (!window.confirm(`Remove published signal \"${formatName(item.name)}\" from dashboard?`)) {
            return;
        }

        setBusyId(item.id);
        setError(null);
        setStatus(null);

        try {
            const response = await fetch("/api/signal-construction/publish", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: item.id }),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Failed (${response.status})`);
            }

            await loadSignals();
            if (onChanged) await onChanged();
            setStatus(`${formatName(item.name)} removed from dashboard.`);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to remove signal.");
        } finally {
            setBusyId(null);
        }
    };

    return (
        <div className="glass-card" style={{ padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 10 }}>
                <div>
                    <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 700 }}>Manage Published Signals</h3>
                    <p style={{ margin: "4px 0 0", color: "var(--text-muted)", fontSize: "0.76rem" }}>
                        {activeCount} active / {signals.length} published
                    </p>
                </div>
                <button
                    onClick={() => void loadSignals()}
                    disabled={loading}
                    style={buttonStyle}
                    title="Refresh published signals"
                >
                    <RefreshCw size={14} />
                </button>
            </div>

            {error && (
                <div style={{ color: "var(--accent-rose)", fontSize: "0.76rem", marginBottom: 8 }}>
                    {error}
                </div>
            )}

            {status && (
                <div style={{ color: "var(--accent-emerald)", fontSize: "0.76rem", marginBottom: 8 }}>
                    {status}
                </div>
            )}

            {loading ? (
                <div style={{ color: "var(--text-muted)", fontSize: "0.82rem", display: "flex", alignItems: "center", gap: 6 }}>
                    <Loader2 size={14} className="spin-icon" /> Loading...
                </div>
            ) : signals.length === 0 ? (
                <div style={{ color: "var(--text-muted)", fontSize: "0.82rem" }}>
                    No published builder signals yet.
                </div>
            ) : (
                <div style={{ display: "grid", gap: 8, maxHeight: 260, overflowY: "auto", paddingRight: 4 }}>
                    {signals.map((item) => {
                        const isBusy = busyId === item.id;
                        return (
                            <div
                                key={item.id}
                                style={{
                                    border: "1px solid var(--border-subtle)",
                                    borderRadius: "var(--radius-sm)",
                                    padding: "9px 10px",
                                    background: item.enabled ? "rgba(16,185,129,0.06)" : "rgba(100,116,139,0.08)",
                                    opacity: isBusy ? 0.7 : 1,
                                }}
                            >
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                                    <div>
                                        <div style={{ fontSize: "0.82rem", fontWeight: 700 }}>{formatName(item.name)}</div>
                                        <div style={{ color: "var(--text-muted)", fontSize: "0.72rem" }}>
                                            {item.holdings.length} holdings · CAGR {formatFixed(item.signal?.cagr)}%
                                        </div>
                                    </div>
                                    <span
                                        className="badge"
                                        style={{
                                            background: item.enabled ? "var(--accent-emerald-dim)" : "var(--bg-secondary)",
                                            color: item.enabled ? "var(--accent-emerald)" : "var(--text-muted)",
                                        }}
                                    >
                                        {item.enabled ? "Enabled" : "Disabled"}
                                    </span>
                                </div>

                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 8, gap: 8 }}>
                                    <span style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}>
                                        Updated {item.updated_at.slice(0, 10)}
                                    </span>
                                    <div style={{ display: "flex", gap: 6 }}>
                                        <button
                                            onClick={() => void toggleSignal(item)}
                                            disabled={isBusy}
                                            style={buttonStyle}
                                            title={item.enabled ? "Disable signal" : "Enable signal"}
                                        >
                                            {item.enabled ? <ToggleRight size={14} /> : <ToggleLeft size={14} />}
                                        </button>
                                        <button
                                            onClick={() => void removeSignal(item)}
                                            disabled={isBusy}
                                            style={dangerButtonStyle}
                                            title="Remove signal"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            <style>{`.spin-icon { animation: spin 1s linear infinite; } @keyframes spin { to { transform: rotate(360deg); } }`}</style>
        </div>
    );
}

const buttonStyle: CSSProperties = {
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-secondary)",
    color: "var(--text-primary)",
    borderRadius: "var(--radius-sm)",
    padding: "6px 8px",
    cursor: "pointer",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
};

const dangerButtonStyle: CSSProperties = {
    border: "1px solid rgba(244,63,94,0.35)",
    background: "rgba(244,63,94,0.12)",
    color: "var(--accent-rose)",
    borderRadius: "var(--radius-sm)",
    padding: "6px 8px",
    cursor: "pointer",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
};
