"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { FlaskConical, PlayCircle, Plus, Trash2, Upload } from "lucide-react";

interface FactorParamSchema {
    key: string;
    label: string;
    type: "int" | "float" | "string" | "multi_select";
    default: number | string | string[];
    options?: Array<{ value: string; label: string }>;
    min?: number;
    max?: number;
}

interface FactorCatalogEntry {
    name: string;
    label: string;
    description: string;
    rebalance_frequency: string;
    timeline: {
        start_date?: string;
        end_date?: string;
    };
    parameter_schema: FactorParamSchema[];
}

interface FactorSelection {
    id: string;
    name: string;
    enabled: boolean;
    weight: number;
    signal_params: Record<string, unknown>;
}

interface FactorPortfolioOptions {
    use_regime_filter: boolean;
    use_vol_targeting: boolean;
    target_downside_vol: number;
    use_inverse_vol_sizing: boolean;
    use_stop_loss: boolean;
    stop_loss_threshold: number;
    use_liquidity_filter: boolean;
    liquidity_quantile: number;
    use_slippage: boolean;
    slippage_bps: number;
    use_mcap_slippage: boolean;
    mid_cap_slippage_bps: number;
    small_cap_slippage_bps: number;
}

const DEFAULT_PORTFOLIO_OPTIONS: FactorPortfolioOptions = {
    use_regime_filter: true,
    use_vol_targeting: false,
    target_downside_vol: 0.2,
    use_inverse_vol_sizing: false,
    use_stop_loss: false,
    stop_loss_threshold: 0.15,
    use_liquidity_filter: true,
    liquidity_quantile: 0.25,
    use_slippage: true,
    slippage_bps: 5,
    use_mcap_slippage: true,
    mid_cap_slippage_bps: 10,
    small_cap_slippage_bps: 20,
};

interface FactorLabResult {
    meta: {
        mode: string;
        as_of: string;
        start_date: string;
        end_date: string;
        rebalance_frequency: string;
        top_n: number;
        symbols_used: number;
        rows_used: number;
        execution_ms: number;
        factors: Array<{
            name: string;
            weight: number;
            signal_params: Record<string, unknown>;
        }>;
    };
    metrics: {
        cagr: number;
        sharpe: number;
        sortino: number;
        max_dd: number;
        total_return: number;
        win_rate: number;
        beta: number | null;
        rebalance_count: number;
        trade_count: number;
    };
    composite_top: Array<{ symbol: string; score: number }>;
    factor_top_symbols: Record<string, Array<{ symbol: string; score: number }>>;
    current_holdings: string[];
    equity_curve: Array<{ date: string; value: number }>;
    benchmark_curve: Array<{ date: string; value: number }>;
    error?: string;
}

function formatPercent(value: number | null | undefined, digits: number = 2): string {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(digits)}%`;
}

function formatNumber(value: number | null | undefined, digits: number = 3): string {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(digits);
}

function getNested(source: Record<string, unknown>, path: string): unknown {
    const keys = path.split(".");
    let cursor: unknown = source;
    for (const key of keys) {
        if (!cursor || typeof cursor !== "object" || Array.isArray(cursor)) {
            return undefined;
        }
        cursor = (cursor as Record<string, unknown>)[key];
    }
    return cursor;
}

function setNested(source: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
    const keys = path.split(".");
    const out: Record<string, unknown> = { ...source };
    let cursor: Record<string, unknown> = out;

    for (let i = 0; i < keys.length - 1; i += 1) {
        const key = keys[i];
        const current = cursor[key];
        if (!current || typeof current !== "object" || Array.isArray(current)) {
            cursor[key] = {};
        }
        cursor = cursor[key] as Record<string, unknown>;
    }

    cursor[keys[keys.length - 1]] = value;
    return out;
}

function nextId(): string {
    return `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function buildDefaultSignalParams(schema: FactorParamSchema[]): Record<string, unknown> {
    return schema.reduce((acc, param) => {
        const value = Array.isArray(param.default) ? [...param.default] : param.default;
        return setNested(acc, param.key, value);
    }, {} as Record<string, unknown>);
}

function normalizePortfolioOptions(raw: unknown): FactorPortfolioOptions {
    const merged: FactorPortfolioOptions = { ...DEFAULT_PORTFOLIO_OPTIONS };
    const mergedRecord = merged as unknown as Record<string, unknown>;
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        return merged;
    }

    const source = raw as Record<string, unknown>;
    const boolKeys: Array<keyof FactorPortfolioOptions> = [
        "use_regime_filter",
        "use_vol_targeting",
        "use_inverse_vol_sizing",
        "use_stop_loss",
        "use_liquidity_filter",
        "use_slippage",
        "use_mcap_slippage",
    ];
    const numKeys: Array<keyof FactorPortfolioOptions> = [
        "target_downside_vol",
        "stop_loss_threshold",
        "liquidity_quantile",
        "slippage_bps",
        "mid_cap_slippage_bps",
        "small_cap_slippage_bps",
    ];

    for (const key of boolKeys) {
        if (typeof source[key] === "boolean") {
            mergedRecord[key] = source[key] as boolean;
        }
    }
    for (const key of numKeys) {
        const val = source[key];
        if (typeof val === "number" && Number.isFinite(val)) {
            mergedRecord[key] = val;
        }
    }

    return merged;
}

export default function FactorLabPage() {
    const [catalog, setCatalog] = useState<FactorCatalogEntry[]>([]);
    const [loadingCatalog, setLoadingCatalog] = useState<boolean>(true);
    const [catalogError, setCatalogError] = useState<string | null>(null);

    const [rows, setRows] = useState<FactorSelection[]>([]);
    const [startDate, setStartDate] = useState<string>("2018-01-01");
    const [endDate, setEndDate] = useState<string>("2026-12-31");
    const [rebalanceFrequency, setRebalanceFrequency] = useState<string>("monthly");
    const [topN, setTopN] = useState<number>(20);
    const [portfolioOptions, setPortfolioOptions] = useState<FactorPortfolioOptions>(DEFAULT_PORTFOLIO_OPTIONS);

    const [runLoading, setRunLoading] = useState<boolean>(false);
    const [runError, setRunError] = useState<string | null>(null);
    const [result, setResult] = useState<FactorLabResult | null>(null);

    const [publishName, setPublishName] = useState<string>("custom_factor_lab");
    const [publishLoading, setPublishLoading] = useState<boolean>(false);
    const [publishStatus, setPublishStatus] = useState<string | null>(null);
    const [publishError, setPublishError] = useState<string | null>(null);

    useEffect(() => {
        const loadCatalog = async () => {
            setLoadingCatalog(true);
            setCatalogError(null);
            try {
                const response = await fetch("/api/factor-lab", { cache: "no-store" });
                const data = await response.json();
                if (!response.ok || data.error) {
                    throw new Error(data.error || `Failed (${response.status})`);
                }

                const factors = Array.isArray(data.factors) ? (data.factors as FactorCatalogEntry[]) : [];
                setCatalog(factors);
                setPortfolioOptions(normalizePortfolioOptions(data.default_portfolio_options));

                if (factors.length > 0) {
                    const defaults = factors
                        .filter((factor) => ["momentum", "value"].includes(factor.name))
                        .slice(0, 2)
                        .map((factor) => ({
                            id: nextId(),
                            name: factor.name,
                            enabled: true,
                            weight: 1,
                            signal_params: buildDefaultSignalParams(factor.parameter_schema),
                        }));

                    setRows(defaults.length > 0 ? defaults : [{
                        id: nextId(),
                        name: factors[0].name,
                        enabled: true,
                        weight: 1,
                        signal_params: buildDefaultSignalParams(factors[0].parameter_schema ?? []),
                    }]);
                }
            } catch (err) {
                setCatalogError(err instanceof Error ? err.message : "Failed to load factor catalog.");
            } finally {
                setLoadingCatalog(false);
            }
        };

        void loadCatalog();
    }, []);

    const catalogByName = useMemo(() => {
        const map = new Map<string, FactorCatalogEntry>();
        for (const entry of catalog) map.set(entry.name, entry);
        return map;
    }, [catalog]);

    const enabledRows = useMemo(() => rows.filter((row) => row.enabled && row.weight > 0), [rows]);

    const addFactorRow = () => {
        if (catalog.length === 0) return;
        setRows((prev) => [
            ...prev,
            {
                id: nextId(),
                name: catalog[0].name,
                enabled: true,
                weight: 1,
                signal_params: buildDefaultSignalParams(catalog[0].parameter_schema ?? []),
            },
        ]);
    };

    const removeFactorRow = (id: string) => {
        setRows((prev) => prev.filter((row) => row.id !== id));
    };

    const updateRow = (id: string, updater: (row: FactorSelection) => FactorSelection) => {
        setRows((prev) => prev.map((row) => (row.id === id ? updater(row) : row)));
    };

    const changeFactor = (id: string, name: string) => {
        const selected = catalogByName.get(name);
        updateRow(id, (row) => {
            let params = row.signal_params;
            if (selected?.parameter_schema?.length) {
                params = buildDefaultSignalParams(selected.parameter_schema);
            }
            return {
                ...row,
                name,
                signal_params: params,
            };
        });
    };

    const updateParam = (id: string, key: string, value: string, kind: "int" | "float" | "string") => {
        updateRow(id, (row) => {
            let parsed: unknown = value;
            if (kind === "int") {
                parsed = Number.parseInt(value, 10);
                if (!Number.isFinite(parsed as number)) parsed = 0;
            } else if (kind === "float") {
                parsed = Number.parseFloat(value);
                if (!Number.isFinite(parsed as number)) parsed = 0;
            }

            return {
                ...row,
                signal_params: setNested(row.signal_params, key, parsed),
            };
        });
    };

    const updateMultiSelectParam = (id: string, key: string, option: string, enabled: boolean) => {
        updateRow(id, (row) => {
            const current = getNested(row.signal_params, key);
            const currentValues = Array.isArray(current)
                ? current.filter((item): item is string => typeof item === "string")
                : [];
            const nextValues = enabled
                ? Array.from(new Set([...currentValues, option]))
                : currentValues.filter((item) => item !== option);
            return {
                ...row,
                signal_params: setNested(row.signal_params, key, nextValues),
            };
        });
    };

    const runFactorLab = async () => {
        setRunLoading(true);
        setRunError(null);
        setPublishStatus(null);
        setPublishError(null);

        try {
            const payload = {
                start_date: startDate,
                end_date: endDate,
                rebalance_frequency: rebalanceFrequency,
                top_n: topN,
                portfolio_options: portfolioOptions,
                factors: rows.map((row) => ({
                    name: row.name,
                    enabled: row.enabled,
                    weight: row.weight,
                    signal_params: row.signal_params,
                })),
            };

            const response = await fetch("/api/factor-lab", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const data: FactorLabResult = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Backtest failed (${response.status})`);
            }

            setResult(data);
            setPublishName(`factor_lab_${data.meta.factors.map((factor) => factor.name).slice(0, 2).join("_") || "custom"}`);
        } catch (err) {
            setRunError(err instanceof Error ? err.message : "Unknown error");
        } finally {
            setRunLoading(false);
        }
    };

    const publishToDashboard = async () => {
        setPublishLoading(true);
        setPublishStatus(null);
        setPublishError(null);

        try {
            if (!result) {
                throw new Error("Run a factor backtest first.");
            }
            if (!publishName.trim()) {
                throw new Error("Signal name is required.");
            }
            if (!result.current_holdings.length) {
                throw new Error("No holdings generated to publish.");
            }

            const payload = {
                name: publishName,
                holdings: result.current_holdings,
                config: {
                    source: "factor_lab",
                    start_date: startDate,
                    end_date: endDate,
                    rebalance_frequency: rebalanceFrequency,
                    top_n: topN,
                    portfolio_options: portfolioOptions,
                    factors: rows,
                },
                backtest: {
                    cagr: result.metrics.cagr,
                    sharpe: result.metrics.sharpe,
                    beta: result.metrics.beta,
                    max_dd: result.metrics.max_dd,
                    ytd: 0,
                    total_return: result.metrics.total_return,
                    win_rate: result.metrics.win_rate,
                    last_rebalance: result.meta.as_of.slice(0, 10),
                },
            };

            const response = await fetch("/api/signal-construction/publish", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Publish failed (${response.status})`);
            }

            setPublishStatus(`Published as ${data.published?.signal?.name || publishName}.`);
        } catch (err) {
            setPublishError(err instanceof Error ? err.message : "Failed to publish.");
        } finally {
            setPublishLoading(false);
        }
    };

    const eqEnd = result?.equity_curve[result.equity_curve.length - 1]?.value;
    const benchEnd = result?.benchmark_curve[result.benchmark_curve.length - 1]?.value;

    return (
        <>
            <Navbar />
            <main style={{ paddingTop: 92, paddingBottom: 64, background: "var(--bg-primary)", minHeight: "100vh" }}>
                <div style={{ maxWidth: 1420, margin: "0 auto", padding: "0 24px" }}>
                    <div style={{ marginBottom: 20, display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                        <div>
                            <h1 style={{ margin: 0, fontSize: "1.8rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
                                Factor Lab
                            </h1>
                            <p style={{ margin: "8px 0 0", color: "var(--text-secondary)" }}>
                                Build custom model strategies from your <code>Models</code> factors with parameter overrides and weighted combinations.
                            </p>
                        </div>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <Link href="/signal-construction" style={secondaryBtn}>
                                Technical Builder
                            </Link>
                            <button className="btn-primary" disabled={runLoading || enabledRows.length === 0 || loadingCatalog} onClick={runFactorLab}>
                                <PlayCircle size={16} />
                                {runLoading ? "Running..." : "Run Factor Backtest"}
                            </button>
                        </div>
                    </div>

                    {catalogError && (
                        <div className="glass-card" style={{ padding: 14, marginBottom: 14, borderColor: "rgba(244,63,94,0.4)" }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Catalog error:</strong> {catalogError}
                        </div>
                    )}

                    <div className="glass-card" style={{ padding: 18, marginBottom: 18 }}>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Start Date</span>
                                <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={inputStyle} />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">End Date</span>
                                <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={inputStyle} />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Rebalance</span>
                                <select value={rebalanceFrequency} onChange={(e) => setRebalanceFrequency(e.target.value)} style={inputStyle}>
                                    <option value="monthly">monthly</option>
                                    <option value="quarterly">quarterly</option>
                                </select>
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Top N</span>
                                <input type="number" min={5} max={200} value={topN} onChange={(e) => setTopN(Number(e.target.value))} style={inputStyle} />
                            </label>
                        </div>
                    </div>

                    <div className="glass-card" style={{ padding: 18, marginBottom: 18 }}>
                        <h2 style={{ margin: "0 0 10px", fontSize: "1.03rem" }}>Portfolio Engine Settings</h2>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 10 }}>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_regime_filter}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_regime_filter: e.target.checked }))}
                                />
                                <span className="metric-label">Regime Filter</span>
                            </label>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_vol_targeting}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_vol_targeting: e.target.checked }))}
                                />
                                <span className="metric-label">Vol Targeting</span>
                            </label>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_inverse_vol_sizing}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_inverse_vol_sizing: e.target.checked }))}
                                />
                                <span className="metric-label">Inverse Vol Sizing</span>
                            </label>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_stop_loss}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_stop_loss: e.target.checked }))}
                                />
                                <span className="metric-label">Stop Loss</span>
                            </label>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_liquidity_filter}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_liquidity_filter: e.target.checked }))}
                                />
                                <span className="metric-label">Liquidity Filter</span>
                            </label>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_slippage}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_slippage: e.target.checked }))}
                                />
                                <span className="metric-label">Slippage</span>
                            </label>
                            <label style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <input
                                    type="checkbox"
                                    checked={portfolioOptions.use_mcap_slippage}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, use_mcap_slippage: e.target.checked }))}
                                />
                                <span className="metric-label">MCAP Slippage</span>
                            </label>
                        </div>
                        <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 10 }}>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Target Downside Vol</span>
                                <input
                                    type="number"
                                    min={0.01}
                                    max={1}
                                    step={0.01}
                                    value={portfolioOptions.target_downside_vol}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, target_downside_vol: Number(e.target.value) }))}
                                    style={inputStyle}
                                />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Stop Loss Threshold</span>
                                <input
                                    type="number"
                                    min={0.01}
                                    max={0.9}
                                    step={0.01}
                                    value={portfolioOptions.stop_loss_threshold}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, stop_loss_threshold: Number(e.target.value) }))}
                                    style={inputStyle}
                                />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Liquidity Quantile</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={0.9}
                                    step={0.01}
                                    value={portfolioOptions.liquidity_quantile}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, liquidity_quantile: Number(e.target.value) }))}
                                    style={inputStyle}
                                />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Slippage (bps)</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={100}
                                    step={0.5}
                                    value={portfolioOptions.slippage_bps}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, slippage_bps: Number(e.target.value) }))}
                                    style={inputStyle}
                                />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Mid Cap Slippage</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={200}
                                    step={0.5}
                                    value={portfolioOptions.mid_cap_slippage_bps}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, mid_cap_slippage_bps: Number(e.target.value) }))}
                                    style={inputStyle}
                                />
                            </label>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Small Cap Slippage</span>
                                <input
                                    type="number"
                                    min={0}
                                    max={300}
                                    step={0.5}
                                    value={portfolioOptions.small_cap_slippage_bps}
                                    onChange={(e) => setPortfolioOptions((prev) => ({ ...prev, small_cap_slippage_bps: Number(e.target.value) }))}
                                    style={inputStyle}
                                />
                            </label>
                        </div>
                    </div>

                    <div className="glass-card" style={{ padding: 18, marginBottom: 18 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                            <h2 style={{ margin: 0, fontSize: "1.03rem", display: "inline-flex", gap: 8, alignItems: "center" }}>
                                <FlaskConical size={16} /> Factor Mix
                            </h2>
                            <button onClick={addFactorRow} style={secondaryBtn} disabled={loadingCatalog || catalog.length === 0}>
                                <Plus size={14} /> Add Factor
                            </button>
                        </div>

                        {loadingCatalog ? (
                            <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "0.86rem" }}>Loading factor catalog...</p>
                        ) : rows.length === 0 ? (
                            <p style={{ margin: 0, color: "var(--text-muted)", fontSize: "0.86rem" }}>No factors selected.</p>
                        ) : (
                            <div style={{ display: "grid", gap: 14 }}>
                                {rows.map((row) => {
                                    const meta = catalogByName.get(row.name);
                                    const params = meta?.parameter_schema ?? [];

                                    return (
                                        <div key={row.id} style={{ border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-md)", padding: 12 }}>
                                            <div style={{ display: "grid", gridTemplateColumns: "32px minmax(220px,1fr) 120px auto", gap: 8, alignItems: "center" }}>
                                                <label style={{ display: "inline-flex", justifyContent: "center" }}>
                                                    <input
                                                        type="checkbox"
                                                        checked={row.enabled}
                                                        onChange={(e) => updateRow(row.id, (prev) => ({ ...prev, enabled: e.target.checked }))}
                                                    />
                                                </label>
                                                <select value={row.name} onChange={(e) => changeFactor(row.id, e.target.value)} style={inputStyle}>
                                                    {catalog.map((factor) => (
                                                        <option key={factor.name} value={factor.name}>
                                                            {factor.label}
                                                        </option>
                                                    ))}
                                                </select>
                                                <input
                                                    type="number"
                                                    value={row.weight}
                                                    min={0}
                                                    step={0.1}
                                                    onChange={(e) => updateRow(row.id, (prev) => ({ ...prev, weight: Number(e.target.value) }))}
                                                    style={inputStyle}
                                                    title="Factor weight"
                                                />
                                                <button onClick={() => removeFactorRow(row.id)} style={dangerBtn}>
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>

                                            {meta?.description && (
                                                <p style={{ margin: "6px 0 0", color: "var(--text-muted)", fontSize: "0.8rem" }}>
                                                    {meta.description}
                                                </p>
                                            )}

                                            {params.length > 0 && (
                                                <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 8 }}>
                                                    {params.map((param) => {
                                                        const value = getNested(row.signal_params, param.key);
                                                        if (param.type === "multi_select") {
                                                            const selectedValues = Array.isArray(value)
                                                                ? value.filter((item): item is string => typeof item === "string")
                                                                : Array.isArray(param.default)
                                                                    ? param.default
                                                                    : [];
                                                            const options = param.options ?? [];
                                                            return (
                                                                <div key={param.key} style={{ display: "grid", gap: 4 }}>
                                                                    <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{param.label}</span>
                                                                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                                                                        {options.map((option) => (
                                                                            <label key={`${param.key}-${option.value}`} style={{ display: "inline-flex", gap: 5, alignItems: "center", fontSize: "0.75rem", color: "var(--text-secondary)", border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", padding: "4px 8px" }}>
                                                                                <input
                                                                                    type="checkbox"
                                                                                    checked={selectedValues.includes(option.value)}
                                                                                    onChange={(e) => updateMultiSelectParam(row.id, param.key, option.value, e.target.checked)}
                                                                                />
                                                                                <span>{option.label}</span>
                                                                            </label>
                                                                        ))}
                                                                    </div>
                                                                </div>
                                                            );
                                                        }

                                                        return (
                                                            <label key={param.key} style={{ display: "grid", gap: 4 }}>
                                                                <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{param.label}</span>
                                                                <input
                                                                    type="number"
                                                                    value={typeof value === "number" ? value : Number(param.default)}
                                                                    min={param.min}
                                                                    max={param.max}
                                                                    step={param.type === "int" ? 1 : 0.1}
                                                                    onChange={(e) => {
                                                                        const nextType = param.type === "int" || param.type === "float" ? param.type : "string";
                                                                        updateParam(row.id, param.key, e.target.value, nextType);
                                                                    }}
                                                                    style={inputStyle}
                                                                />
                                                            </label>
                                                        );
                                                    })}
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        )}

                        <div style={{ marginTop: 10, fontSize: "0.82rem", color: "var(--text-muted)" }}>
                            Enabled factors: <strong style={{ color: "var(--text-primary)" }}>{enabledRows.length}</strong>
                        </div>
                    </div>

                    {runError && (
                        <div className="glass-card" style={{ padding: 14, marginBottom: 14, borderColor: "rgba(244,63,94,0.4)" }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Run failed:</strong> {runError}
                        </div>
                    )}

                    {publishError && (
                        <div className="glass-card" style={{ padding: 14, marginBottom: 14, borderColor: "rgba(244,63,94,0.4)" }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Publish failed:</strong> {publishError}
                        </div>
                    )}

                    {publishStatus && (
                        <div className="glass-card" style={{ padding: 14, marginBottom: 14, borderColor: "rgba(16,185,129,0.4)" }}>
                            <strong style={{ color: "var(--accent-emerald)" }}>Published:</strong> {publishStatus}
                        </div>
                    )}

                    {result && (
                        <div className="glass-card" style={{ padding: 18 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: "1.05rem" }}>Factor Lab Backtest</h2>
                                    <p style={{ margin: "6px 0 0", color: "var(--text-muted)", fontSize: "0.8rem" }}>
                                        {result.meta.start_date} → {result.meta.end_date} · {result.meta.execution_ms}ms
                                    </p>
                                </div>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                                    <input value={publishName} onChange={(e) => setPublishName(e.target.value)} style={{ ...inputStyle, minWidth: 210 }} />
                                    <button className="btn-primary" onClick={publishToDashboard} disabled={publishLoading}>
                                        <Upload size={16} />
                                        {publishLoading ? "Publishing..." : "Publish to Dashboard"}
                                    </button>
                                </div>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10, marginBottom: 14 }}>
                                <MetricCard label="CAGR" value={formatPercent(result.metrics.cagr)} />
                                <MetricCard label="Sharpe" value={formatNumber(result.metrics.sharpe, 2)} />
                                <MetricCard label="Sortino" value={formatNumber(result.metrics.sortino, 2)} />
                                <MetricCard label="Max DD" value={formatPercent(result.metrics.max_dd)} />
                                <MetricCard label="Total Return" value={formatPercent(result.metrics.total_return)} />
                                <MetricCard label="Win Rate" value={formatPercent(result.metrics.win_rate)} />
                                <MetricCard label="Beta" value={formatNumber(result.metrics.beta, 2)} />
                                <MetricCard label="Rebalances" value={String(result.metrics.rebalance_count)} />
                            </div>

                            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 14 }}>
                                <span className="badge" style={{ background: "var(--bg-secondary)" }}>
                                    Holdings: {result.current_holdings.length}
                                </span>
                                <span className="badge" style={{ background: "var(--bg-secondary)" }}>
                                    Equity End: {formatNumber(eqEnd, 3)}x
                                </span>
                                <span className="badge" style={{ background: "var(--bg-secondary)" }}>
                                    Benchmark End: {formatNumber(benchEnd, 3)}x
                                </span>
                            </div>

                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
                                <div>
                                    <h3 style={{ margin: "0 0 8px", fontSize: "0.9rem" }}>Composite Top Signals</h3>
                                    <div style={{ overflowX: "auto" }}>
                                        <table className="data-table">
                                            <thead>
                                                <tr><th>#</th><th>Symbol</th><th>Score</th></tr>
                                            </thead>
                                            <tbody>
                                                {result.composite_top.map((row, idx) => (
                                                    <tr key={`${row.symbol}-${idx}`}>
                                                        <td>{idx + 1}</td>
                                                        <td style={{ fontWeight: 700 }}>{row.symbol}</td>
                                                        <td style={{ fontFamily: "var(--font-mono)" }}>{formatNumber(row.score, 3)}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                                <div>
                                    <h3 style={{ margin: "0 0 8px", fontSize: "0.9rem" }}>Factor Breakdown</h3>
                                    <div style={{ display: "grid", gap: 8 }}>
                                        {result.meta.factors.map((factor) => (
                                            <div key={factor.name} style={{ border: "1px solid var(--border-subtle)", borderRadius: "var(--radius-sm)", padding: 8 }}>
                                                <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                                                    <strong style={{ fontSize: "0.84rem" }}>{factor.name.replace(/_/g, " ")}</strong>
                                                    <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                                                        weight {formatNumber(factor.weight, 3)}
                                                    </span>
                                                </div>
                                                <div style={{ marginTop: 6, display: "flex", gap: 6, flexWrap: "wrap" }}>
                                                    {(result.factor_top_symbols[factor.name] || []).slice(0, 6).map((s) => (
                                                        <span key={`${factor.name}-${s.symbol}`} className="badge" style={{ background: "var(--accent-cyan-dim)", color: "var(--accent-cyan)" }}>
                                                            {s.symbol}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </>
    );
}

function MetricCard({ label, value }: { label: string; value: string }) {
    return (
        <div className="glass-card" style={{ padding: 10 }}>
            <div className="metric-label">{label}</div>
            <div className="metric-value" style={{ fontSize: "1rem" }}>{value}</div>
        </div>
    );
}

const inputStyle: CSSProperties = {
    width: "100%",
    border: "1px solid var(--border-subtle)",
    background: "rgba(0,0,0,0.3)",
    color: "var(--text-primary)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 10px",
    fontSize: "0.85rem",
    outline: "none",
};

const secondaryBtn: CSSProperties = {
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-secondary)",
    color: "var(--text-primary)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 12px",
    fontSize: "0.8rem",
    fontWeight: 600,
    cursor: "pointer",
    display: "inline-flex",
    gap: 6,
    alignItems: "center",
    textDecoration: "none",
};

const dangerBtn: CSSProperties = {
    border: "1px solid rgba(244,63,94,0.35)",
    background: "rgba(244,63,94,0.12)",
    color: "var(--accent-rose)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 10px",
    fontSize: "0.8rem",
    fontWeight: 600,
    cursor: "pointer",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
};
