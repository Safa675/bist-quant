"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import Navbar from "@/components/Navbar";
import { Settings2, PlayCircle, SlidersHorizontal, TrendingUp, TrendingDown, Minus } from "lucide-react";

type Universe = "XU030" | "XU100" | "XUTUM" | "CUSTOM";
type IndicatorKey = "rsi" | "macd" | "bollinger" | "atr" | "stochastic" | "adx" | "supertrend";

interface IndicatorParamDef {
    key: string;
    label: string;
    step: number;
    min?: number;
    max?: number;
}

interface IndicatorDef {
    label: string;
    description: string;
    params: IndicatorParamDef[];
}

interface IndicatorConfigState {
    enabled: boolean;
    params: Record<string, number>;
}

type IndicatorState = Record<IndicatorKey, IndicatorConfigState>;

interface SignalRow {
    symbol: string;
    action: string;
    combined_score: number | null;
    buy_votes: number | null;
    sell_votes: number | null;
    hold_votes: number | null;
    indicator_values: Record<string, number | null>;
    indicator_signals: Record<string, number | null>;
}

interface IndicatorSummary {
    name: string;
    buy_count: number;
    sell_count: number;
    hold_count: number;
}

interface BuilderResult {
    meta: {
        mode?: string;
        universe: string;
        period: string;
        interval: string;
        symbols_used: number;
        rows_used: number;
        as_of: string;
        indicators: string[];
        execution_ms: number;
    };
    indicator_summaries: IndicatorSummary[];
    signals: SignalRow[];
    error?: string;
}

interface BacktestResult {
    meta: {
        mode: string;
        universe: string;
        period: string;
        interval: string;
        symbols_used: number;
        rows_used: number;
        as_of: string;
        indicators: string[];
        buy_threshold: number;
        sell_threshold: number;
        max_positions: number;
        execution_ms: number;
    };
    metrics: {
        cagr: number;
        sharpe: number;
        max_dd: number;
        ytd: number;
        total_return: number;
        volatility: number;
        win_rate: number;
        beta: number | null;
        last_rebalance: string;
    };
    current_holdings: string[];
    equity_curve: Array<{ date: string; value: number }>;
    benchmark_curve: Array<{ date: string; value: number }>;
    error?: string;
}

interface PresetRecord {
    id: string;
    name: string;
    config: BuilderPayload;
    created_at: string;
    updated_at: string;
}

interface BuilderPayload {
    universe: Universe;
    symbols: string | string[];
    period: string;
    interval: string;
    max_symbols: number;
    top_n: number;
    buy_threshold: number;
    sell_threshold: number;
    indicators: IndicatorState;
}

const INDICATOR_DEFS: Record<IndicatorKey, IndicatorDef> = {
    rsi: {
        label: "RSI",
        description: "Mean-reversion thresholds with oversold/overbought bands.",
        params: [
            { key: "period", label: "Period", step: 1, min: 2, max: 200 },
            { key: "oversold", label: "Oversold", step: 1, min: 0, max: 100 },
            { key: "overbought", label: "Overbought", step: 1, min: 0, max: 100 },
        ],
    },
    macd: {
        label: "MACD Histogram",
        description: "Directional signal from MACD histogram sign and threshold.",
        params: [
            { key: "fast", label: "Fast EMA", step: 1, min: 2, max: 200 },
            { key: "slow", label: "Slow EMA", step: 1, min: 2, max: 300 },
            { key: "signal", label: "Signal EMA", step: 1, min: 2, max: 200 },
            { key: "threshold", label: "Threshold", step: 0.01, min: 0, max: 10 },
        ],
    },
    bollinger: {
        label: "Bollinger %B",
        description: "Band-position based mean-reversion signal on %B.",
        params: [
            { key: "period", label: "Period", step: 1, min: 2, max: 300 },
            { key: "std_dev", label: "Std Dev", step: 0.1, min: 0.1, max: 5 },
            { key: "lower", label: "Lower %B", step: 0.05, min: -1, max: 1 },
            { key: "upper", label: "Upper %B", step: 0.05, min: 0, max: 2 },
        ],
    },
    atr: {
        label: "ATR",
        description: "Cross-sectional volatility preference (lower ATR -> long bias).",
        params: [
            { key: "period", label: "Period", step: 1, min: 2, max: 300 },
            { key: "lower_pct", label: "Lower Pct", step: 0.05, min: 0, max: 1 },
            { key: "upper_pct", label: "Upper Pct", step: 0.05, min: 0, max: 1 },
        ],
    },
    stochastic: {
        label: "Stochastic %K",
        description: "Mean-reversion thresholds on stochastic oscillator.",
        params: [
            { key: "k_period", label: "%K Period", step: 1, min: 2, max: 200 },
            { key: "d_period", label: "%D Period", step: 1, min: 1, max: 100 },
            { key: "oversold", label: "Oversold", step: 1, min: 0, max: 100 },
            { key: "overbought", label: "Overbought", step: 1, min: 0, max: 100 },
        ],
    },
    adx: {
        label: "ADX (+DI/-DI)",
        description: "Trend-strength filter with DI directional confirmation.",
        params: [
            { key: "period", label: "Period", step: 1, min: 2, max: 300 },
            { key: "trend_threshold", label: "Trend Threshold", step: 1, min: 5, max: 60 },
        ],
    },
    supertrend: {
        label: "Supertrend Direction",
        description: "Direct directional regime from supertrend state.",
        params: [
            { key: "period", label: "Period", step: 1, min: 2, max: 200 },
            { key: "multiplier", label: "Multiplier", step: 0.1, min: 0.5, max: 10 },
        ],
    },
};

const DEFAULT_INDICATORS: IndicatorState = {
    rsi: { enabled: true, params: { period: 14, oversold: 30, overbought: 70 } },
    macd: { enabled: true, params: { fast: 12, slow: 26, signal: 9, threshold: 0 } },
    bollinger: { enabled: false, params: { period: 20, std_dev: 2.0, lower: 0.2, upper: 0.8 } },
    atr: { enabled: false, params: { period: 14, lower_pct: 0.3, upper_pct: 0.7 } },
    stochastic: { enabled: false, params: { k_period: 14, d_period: 3, oversold: 20, overbought: 80 } },
    adx: { enabled: false, params: { period: 14, trend_threshold: 25 } },
    supertrend: { enabled: true, params: { period: 10, multiplier: 3.0 } },
};

const PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y"];

function cloneDefaultIndicators(): IndicatorState {
    return (Object.keys(DEFAULT_INDICATORS) as IndicatorKey[]).reduce((acc, key) => {
        acc[key] = {
            enabled: DEFAULT_INDICATORS[key].enabled,
            params: { ...DEFAULT_INDICATORS[key].params },
        };
        return acc;
    }, {} as IndicatorState);
}

function normalizeIndicators(raw: unknown): IndicatorState {
    const next = cloneDefaultIndicators();
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        return next;
    }

    for (const key of Object.keys(next) as IndicatorKey[]) {
        const incoming = (raw as Record<string, unknown>)[key];
        if (!incoming || typeof incoming !== "object" || Array.isArray(incoming)) {
            continue;
        }

        const incomingObj = incoming as Record<string, unknown>;
        next[key] = {
            enabled: Boolean(incomingObj.enabled),
            params: {
                ...next[key].params,
                ...(incomingObj.params && typeof incomingObj.params === "object" && !Array.isArray(incomingObj.params)
                    ? (incomingObj.params as Record<string, number>)
                    : {}),
            },
        };
    }

    return next;
}

function formatNumber(value: number | null, decimals: number = 2): string {
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    return value.toFixed(decimals);
}

function formatPercent(value: number | null, decimals: number = 2): string {
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    const sign = value > 0 ? "+" : "";
    return `${sign}${value.toFixed(decimals)}%`;
}

function ActionBadge({ action }: { action: string }) {
    if (action === "BUY") {
        return (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--accent-emerald)", fontWeight: 700 }}>
                <TrendingUp size={14} />
                BUY
            </span>
        );
    }
    if (action === "SELL") {
        return (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--accent-rose)", fontWeight: 700 }}>
                <TrendingDown size={14} />
                SELL
            </span>
        );
    }
    return (
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: "var(--text-muted)", fontWeight: 700 }}>
            <Minus size={14} />
            HOLD
        </span>
    );
}

export default function SignalConstructionPage() {
    const [universe, setUniverse] = useState<Universe>("XU100");
    const [customSymbols, setCustomSymbols] = useState<string>("THYAO,AKBNK,GARAN,EREGL,TUPRS");
    const [period, setPeriod] = useState<string>("6mo");
    const [maxSymbols, setMaxSymbols] = useState<number>(50);
    const [topN, setTopN] = useState<number>(30);
    const [buyThreshold, setBuyThreshold] = useState<number>(0.2);
    const [sellThreshold, setSellThreshold] = useState<number>(-0.2);
    const [indicators, setIndicators] = useState<IndicatorState>(cloneDefaultIndicators());

    const [presetName, setPresetName] = useState<string>("my_signal_setup");
    const [selectedPresetId, setSelectedPresetId] = useState<string>("");
    const [presets, setPresets] = useState<PresetRecord[]>([]);
    const [presetStatus, setPresetStatus] = useState<string | null>(null);

    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<BuilderResult | null>(null);

    const [selectedSymbols, setSelectedSymbols] = useState<Set<string>>(new Set());
    const [publishName, setPublishName] = useState<string>("constructed_signal");
    const [publishLoading, setPublishLoading] = useState<boolean>(false);
    const [publishStatus, setPublishStatus] = useState<string | null>(null);
    const [publishError, setPublishError] = useState<string | null>(null);

    const [backtestLoading, setBacktestLoading] = useState<boolean>(false);
    const [backtestError, setBacktestError] = useState<string | null>(null);
    const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);

    const enabledIndicators = useMemo(
        () => (Object.keys(indicators) as IndicatorKey[]).filter((key) => indicators[key].enabled),
        [indicators]
    );

    const allVisibleSelected = useMemo(() => {
        if (!result || result.signals.length === 0) return false;
        return result.signals.every((row) => selectedSymbols.has(row.symbol));
    }, [result, selectedSymbols]);

    const buildPayload = (): BuilderPayload => ({
        universe,
        symbols: universe === "CUSTOM" ? customSymbols : "",
        period,
        interval: "1d",
        max_symbols: maxSymbols,
        top_n: topN,
        buy_threshold: buyThreshold,
        sell_threshold: sellThreshold,
        indicators,
    });

    const loadPresets = async () => {
        try {
            const response = await fetch("/api/signal-construction/presets", { cache: "no-store" });
            const data = await response.json();
            if (!response.ok || !Array.isArray(data.presets)) {
                throw new Error(data.error || "Failed to load presets.");
            }
            setPresets(data.presets as PresetRecord[]);
        } catch (err) {
            setPresetStatus(err instanceof Error ? err.message : "Failed to load presets.");
        }
    };

    useEffect(() => {
        void loadPresets();
    }, []);

    const applyPresetConfig = (config: BuilderPayload) => {
        const rawUniverse = typeof config.universe === "string" ? config.universe.toUpperCase() : "XU100";
        const parsedUniverse: Universe =
            rawUniverse === "XU030" || rawUniverse === "XUTUM" || rawUniverse === "CUSTOM" ? rawUniverse : "XU100";

        setUniverse(parsedUniverse);
        if (typeof config.symbols === "string") {
            setCustomSymbols(config.symbols);
        } else if (Array.isArray(config.symbols)) {
            setCustomSymbols(config.symbols.join(","));
        }

        setPeriod(typeof config.period === "string" ? config.period : "6mo");
        setMaxSymbols(typeof config.max_symbols === "number" ? config.max_symbols : 50);
        setTopN(typeof config.top_n === "number" ? config.top_n : 30);
        setBuyThreshold(typeof config.buy_threshold === "number" ? config.buy_threshold : 0.2);
        setSellThreshold(typeof config.sell_threshold === "number" ? config.sell_threshold : -0.2);
        setIndicators(normalizeIndicators(config.indicators));
    };

    const handleToggleIndicator = (key: IndicatorKey) => {
        setIndicators((prev) => ({
            ...prev,
            [key]: {
                ...prev[key],
                enabled: !prev[key].enabled,
            },
        }));
    };

    const handleParamChange = (indicator: IndicatorKey, paramKey: string, value: number) => {
        setIndicators((prev) => ({
            ...prev,
            [indicator]: {
                ...prev[indicator],
                params: {
                    ...prev[indicator].params,
                    [paramKey]: value,
                },
            },
        }));
    };

    const runConstruction = async () => {
        setLoading(true);
        setError(null);
        setPublishStatus(null);

        try {
            const response = await fetch("/api/signal-construction", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildPayload()),
            });
            const data: BuilderResult = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || `Request failed (${response.status})`);
            }

            setResult(data);
            setBacktestResult(null);
            setBacktestError(null);

            const buySymbols = data.signals.filter((row) => row.action === "BUY").map((row) => row.symbol);
            setSelectedSymbols(new Set(buySymbols));
            if (!publishName || publishName === "constructed_signal") {
                setPublishName(`${data.meta.universe.toLowerCase()}_constructed`);
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : "Unknown error";
            setError(message);
        } finally {
            setLoading(false);
        }
    };

    const runBacktest = async () => {
        setBacktestLoading(true);
        setBacktestError(null);
        setPublishStatus(null);

        try {
            const response = await fetch("/api/signal-construction/backtest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildPayload()),
            });
            const data: BacktestResult = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || `Backtest failed (${response.status})`);
            }

            setBacktestResult(data);

            if (Array.isArray(data.current_holdings) && data.current_holdings.length > 0) {
                setSelectedSymbols(new Set(data.current_holdings));
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : "Unknown error";
            setBacktestError(message);
        } finally {
            setBacktestLoading(false);
        }
    };

    const saveCurrentPreset = async () => {
        setPresetStatus(null);
        try {
            const response = await fetch("/api/signal-construction/presets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: presetName,
                    config: buildPayload(),
                }),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Failed (${response.status})`);
            }

            const preset = data.preset as PresetRecord;
            setSelectedPresetId(preset.id);
            setPresetStatus(`Saved preset: ${preset.name}`);
            await loadPresets();
        } catch (err) {
            setPresetStatus(err instanceof Error ? err.message : "Failed to save preset.");
        }
    };

    const loadSelectedPreset = () => {
        const preset = presets.find((item) => item.id === selectedPresetId);
        if (!preset) {
            setPresetStatus("Select a preset to load.");
            return;
        }

        applyPresetConfig(preset.config);
        setPresetName(preset.name);
        setPublishName(`${preset.name.toLowerCase().replace(/\s+/g, "_")}_signal`);
        setPresetStatus(`Loaded preset: ${preset.name}`);
    };

    const deleteSelectedPreset = async () => {
        const preset = presets.find((item) => item.id === selectedPresetId);
        if (!preset) {
            setPresetStatus("Select a preset to delete.");
            return;
        }

        try {
            const response = await fetch("/api/signal-construction/presets", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: preset.id }),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.error || `Delete failed (${response.status})`);
            }

            setSelectedPresetId("");
            setPresetStatus(`Deleted preset: ${preset.name}`);
            await loadPresets();
        } catch (err) {
            setPresetStatus(err instanceof Error ? err.message : "Failed to delete preset.");
        }
    };

    const toggleSymbolSelection = (symbol: string) => {
        setSelectedSymbols((prev) => {
            const next = new Set(prev);
            if (next.has(symbol)) {
                next.delete(symbol);
            } else {
                next.add(symbol);
            }
            return next;
        });
    };

    const selectBuyRows = () => {
        if (!result) return;
        setSelectedSymbols(new Set(result.signals.filter((row) => row.action === "BUY").map((row) => row.symbol)));
    };

    const toggleSelectVisible = () => {
        if (!result) return;
        setSelectedSymbols((prev) => {
            if (allVisibleSelected) {
                const next = new Set(prev);
                for (const row of result.signals) {
                    next.delete(row.symbol);
                }
                return next;
            }

            const next = new Set(prev);
            for (const row of result.signals) {
                next.add(row.symbol);
            }
            return next;
        });
    };

    const clearSelection = () => {
        setSelectedSymbols(new Set());
    };

    const publishSelectedSignals = async () => {
        setPublishError(null);
        setPublishStatus(null);
        setPublishLoading(true);

        try {
            if (!result) {
                throw new Error("Build signals first before publishing.");
            }

            const holdings = Array.from(selectedSymbols);
            if (!holdings.length) {
                throw new Error("Select at least one constructed signal row to publish.");
            }

            const publishPayload = {
                name: publishName,
                holdings,
                config: buildPayload(),
                backtest: backtestResult?.metrics,
            };

            const response = await fetch("/api/signal-construction/publish", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(publishPayload),
            });
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || `Publish failed (${response.status})`);
            }

            setPublishStatus(`Published to dashboard as: ${data.published?.signal?.name || publishName}`);
        } catch (err) {
            setPublishError(err instanceof Error ? err.message : "Failed to publish signal.");
        } finally {
            setPublishLoading(false);
        }
    };

    const backtestEquityEnd = backtestResult?.equity_curve[backtestResult.equity_curve.length - 1]?.value ?? null;
    const backtestBenchEnd = backtestResult?.benchmark_curve[backtestResult.benchmark_curve.length - 1]?.value ?? null;

    return (
        <>
            <Navbar />
            <main style={{ paddingTop: 92, paddingBottom: 64, background: "var(--bg-primary)", minHeight: "100vh" }}>
                <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 24px" }}>
                    <div style={{ marginBottom: 22 }}>
                        <h1 style={{ margin: 0, fontSize: "1.8rem", fontWeight: 800, letterSpacing: "-0.02em" }}>
                            Signal Construction
                        </h1>
                        <p style={{ marginTop: 8, marginBottom: 0, color: "var(--text-secondary)" }}>
                            Build, backtest, and publish technical-indicator signals from your production borsapy stack.
                        </p>
                    </div>

                    <div className="glass-card" style={{ padding: 20, marginBottom: 20 }}>
                        <h2 style={{ margin: 0, fontSize: "1.05rem", marginBottom: 12 }}>Strategy Presets</h2>
                        <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(220px, 1fr) auto auto auto", gap: 8 }}>
                            <input
                                value={presetName}
                                onChange={(e) => setPresetName(e.target.value)}
                                placeholder="Preset name"
                                style={inputStyle}
                            />
                            <select
                                value={selectedPresetId}
                                onChange={(e) => setSelectedPresetId(e.target.value)}
                                style={inputStyle}
                            >
                                <option value="">Select preset</option>
                                {presets.map((preset) => (
                                    <option key={preset.id} value={preset.id}>
                                        {preset.name}
                                    </option>
                                ))}
                            </select>
                            <button onClick={saveCurrentPreset} style={secondaryButtonStyle}>Save Preset</button>
                            <button onClick={loadSelectedPreset} style={secondaryButtonStyle}>Load Preset</button>
                            <button onClick={deleteSelectedPreset} style={dangerButtonStyle}>Delete</button>
                        </div>
                        {presetStatus && (
                            <div style={{ marginTop: 10, fontSize: "0.82rem", color: "var(--text-muted)" }}>
                                {presetStatus}
                            </div>
                        )}
                    </div>

                    <div className="glass-card" style={{ padding: 20, marginBottom: 20 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                            <Settings2 size={18} />
                            <h2 style={{ margin: 0, fontSize: "1.05rem" }}>Construction Settings</h2>
                        </div>

                        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Universe</span>
                                <select
                                    value={universe}
                                    onChange={(e) => setUniverse(e.target.value as Universe)}
                                    style={inputStyle}
                                >
                                    <option value="XU030">XU030</option>
                                    <option value="XU100">XU100</option>
                                    <option value="XUTUM">XUTUM</option>
                                    <option value="CUSTOM">CUSTOM</option>
                                </select>
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Period</span>
                                <select value={period} onChange={(e) => setPeriod(e.target.value)} style={inputStyle}>
                                    {PERIOD_OPTIONS.map((p) => (
                                        <option key={p} value={p}>
                                            {p}
                                        </option>
                                    ))}
                                </select>
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Max Symbols</span>
                                <input
                                    type="number"
                                    value={maxSymbols}
                                    onChange={(e) => setMaxSymbols(Number(e.target.value))}
                                    min={5}
                                    max={200}
                                    style={inputStyle}
                                />
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Top Rows / Positions</span>
                                <input
                                    type="number"
                                    value={topN}
                                    onChange={(e) => setTopN(Number(e.target.value))}
                                    min={1}
                                    max={200}
                                    style={inputStyle}
                                />
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Buy Threshold</span>
                                <input
                                    type="number"
                                    value={buyThreshold}
                                    onChange={(e) => setBuyThreshold(Number(e.target.value))}
                                    step={0.05}
                                    style={inputStyle}
                                />
                            </label>

                            <label style={{ display: "grid", gap: 6 }}>
                                <span className="metric-label">Sell Threshold</span>
                                <input
                                    type="number"
                                    value={sellThreshold}
                                    onChange={(e) => setSellThreshold(Number(e.target.value))}
                                    step={0.05}
                                    style={inputStyle}
                                />
                            </label>
                        </div>

                        {universe === "CUSTOM" && (
                            <label style={{ display: "grid", gap: 6, marginTop: 12 }}>
                                <span className="metric-label">Custom Symbols (comma-separated)</span>
                                <textarea
                                    value={customSymbols}
                                    onChange={(e) => setCustomSymbols(e.target.value)}
                                    rows={2}
                                    style={{ ...inputStyle, resize: "vertical", minHeight: 72 }}
                                />
                            </label>
                        )}
                    </div>

                    <div className="glass-card" style={{ padding: 20, marginBottom: 20 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                            <SlidersHorizontal size={18} />
                            <h2 style={{ margin: 0, fontSize: "1.05rem" }}>Indicators</h2>
                        </div>

                        <div style={{ display: "grid", gap: 12 }}>
                            {(Object.keys(INDICATOR_DEFS) as IndicatorKey[]).map((key) => {
                                const def = INDICATOR_DEFS[key];
                                const state = indicators[key];
                                return (
                                    <div
                                        key={key}
                                        style={{
                                            border: "1px solid var(--border-subtle)",
                                            borderRadius: "var(--radius-md)",
                                            padding: 12,
                                            background: state.enabled ? "rgba(16,185,129,0.06)" : "var(--bg-secondary)",
                                        }}
                                    >
                                        <label style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer" }}>
                                            <input
                                                type="checkbox"
                                                checked={state.enabled}
                                                onChange={() => handleToggleIndicator(key)}
                                            />
                                            <span style={{ fontWeight: 700 }}>{def.label}</span>
                                        </label>
                                        <div style={{ marginTop: 4, color: "var(--text-muted)", fontSize: "0.84rem" }}>
                                            {def.description}
                                        </div>

                                        {state.enabled && (
                                            <div
                                                style={{
                                                    marginTop: 10,
                                                    display: "grid",
                                                    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                                                    gap: 10,
                                                }}
                                            >
                                                {def.params.map((param) => (
                                                    <label key={param.key} style={{ display: "grid", gap: 4 }}>
                                                        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{param.label}</span>
                                                        <input
                                                            type="number"
                                                            value={state.params[param.key]}
                                                            step={param.step}
                                                            min={param.min}
                                                            max={param.max}
                                                            onChange={(e) =>
                                                                handleParamChange(key, param.key, Number(e.target.value))
                                                            }
                                                            style={inputStyle}
                                                        />
                                                    </label>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>

                        <div style={{ marginTop: 14, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                            <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                                Enabled: <strong style={{ color: "var(--text-primary)" }}>{enabledIndicators.length}</strong>
                                {result && (
                                    <>
                                        {" "}
                                        · Selected for publish: <strong style={{ color: "var(--text-primary)" }}>{selectedSymbols.size}</strong>
                                    </>
                                )}
                            </div>
                            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                <button className="btn-primary" onClick={runConstruction} disabled={loading}>
                                    <PlayCircle size={16} />
                                    {loading ? "Building..." : "Build Signals"}
                                </button>
                                <button onClick={runBacktest} disabled={backtestLoading} style={secondaryButtonStyle}>
                                    {backtestLoading ? "Backtesting..." : "Backtest In-App"}
                                </button>
                            </div>
                        </div>
                    </div>

                    {error && (
                        <div className="glass-card" style={{ padding: 16, borderColor: "rgba(244,63,94,0.35)", marginBottom: 20 }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Signal construction failed:</strong>{" "}
                            <span style={{ color: "var(--text-secondary)" }}>{error}</span>
                        </div>
                    )}

                    {backtestError && (
                        <div className="glass-card" style={{ padding: 16, borderColor: "rgba(244,63,94,0.35)", marginBottom: 20 }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Backtest failed:</strong>{" "}
                            <span style={{ color: "var(--text-secondary)" }}>{backtestError}</span>
                        </div>
                    )}

                    {publishError && (
                        <div className="glass-card" style={{ padding: 16, borderColor: "rgba(244,63,94,0.35)", marginBottom: 20 }}>
                            <strong style={{ color: "var(--accent-rose)" }}>Publish failed:</strong>{" "}
                            <span style={{ color: "var(--text-secondary)" }}>{publishError}</span>
                        </div>
                    )}

                    {publishStatus && (
                        <div className="glass-card" style={{ padding: 16, borderColor: "rgba(16,185,129,0.35)", marginBottom: 20 }}>
                            <strong style={{ color: "var(--accent-emerald)" }}>Published:</strong>{" "}
                            <span style={{ color: "var(--text-secondary)" }}>{publishStatus}</span>
                        </div>
                    )}

                    {result && (
                        <div className="glass-card" style={{ padding: 20, marginBottom: 20 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
                                <div>
                                    <h2 style={{ margin: 0, fontSize: "1.05rem" }}>Constructed Signals</h2>
                                    <p style={{ margin: "6px 0 0", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                                        As of {result.meta.as_of} · {result.meta.execution_ms}ms
                                    </p>
                                </div>
                                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                    <span className="badge badge-bull">{result.meta.universe}</span>
                                    <span className="badge" style={{ background: "var(--accent-cyan-dim)", color: "var(--accent-cyan)" }}>
                                        {result.meta.period}
                                    </span>
                                    <span className="badge" style={{ background: "var(--bg-secondary)" }}>
                                        {result.meta.symbols_used} symbols
                                    </span>
                                </div>
                            </div>

                            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                                    <button onClick={toggleSelectVisible} style={secondaryButtonStyle}>
                                        {allVisibleSelected ? "Unselect Visible" : "Select Visible"}
                                    </button>
                                    <button onClick={selectBuyRows} style={secondaryButtonStyle}>Select BUY</button>
                                    <button onClick={clearSelection} style={secondaryButtonStyle}>Clear</button>
                                    <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>
                                        {selectedSymbols.size} selected
                                    </span>
                                </div>
                                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                                    <input
                                        value={publishName}
                                        onChange={(e) => setPublishName(e.target.value)}
                                        placeholder="Dashboard signal name"
                                        style={{ ...inputStyle, minWidth: 220 }}
                                    />
                                    <button onClick={publishSelectedSignals} disabled={publishLoading} className="btn-primary">
                                        {publishLoading ? "Publishing..." : "Push To Main Dashboard"}
                                    </button>
                                </div>
                            </div>

                            <div style={{ overflowX: "auto" }}>
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            <th>
                                                <input
                                                    type="checkbox"
                                                    checked={allVisibleSelected}
                                                    onChange={toggleSelectVisible}
                                                    aria-label="Select all visible"
                                                />
                                            </th>
                                            <th>#</th>
                                            <th>Symbol</th>
                                            <th>Action</th>
                                            <th>Score</th>
                                            <th>Votes (B/S/H)</th>
                                            {result.meta.indicators.map((name) => (
                                                <th key={name}>{INDICATOR_DEFS[name as IndicatorKey]?.label ?? name}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {result.signals.map((row, idx) => (
                                            <tr key={`${row.symbol}-${idx}`}>
                                                <td>
                                                    <input
                                                        type="checkbox"
                                                        checked={selectedSymbols.has(row.symbol)}
                                                        onChange={() => toggleSymbolSelection(row.symbol)}
                                                        aria-label={`Select ${row.symbol}`}
                                                    />
                                                </td>
                                                <td>{idx + 1}</td>
                                                <td style={{ fontWeight: 700 }}>{row.symbol}</td>
                                                <td>
                                                    <ActionBadge action={row.action} />
                                                </td>
                                                <td style={{ fontFamily: "var(--font-mono)" }}>{formatNumber(row.combined_score, 3)}</td>
                                                <td style={{ fontFamily: "var(--font-mono)" }}>
                                                    {row.buy_votes}/{row.sell_votes}/{row.hold_votes}
                                                </td>
                                                {result.meta.indicators.map((name) => {
                                                    const v = row.indicator_values[name];
                                                    const s = row.indicator_signals[name];
                                                    const signalColor =
                                                        s === 1
                                                            ? "var(--accent-emerald)"
                                                            : s === -1
                                                                ? "var(--accent-rose)"
                                                                : "var(--text-muted)";
                                                    return (
                                                        <td key={`${row.symbol}-${name}`}>
                                                            <span style={{ fontFamily: "var(--font-mono)" }}>{formatNumber(v)}</span>
                                                            <span style={{ color: signalColor, marginLeft: 6, fontSize: "0.78rem" }}>
                                                                {s === 1 ? "▲" : s === -1 ? "▼" : "•"}
                                                            </span>
                                                        </td>
                                                    );
                                                })}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            <div style={{ marginTop: 14, display: "flex", gap: 8, flexWrap: "wrap" }}>
                                {result.indicator_summaries.map((summary) => (
                                    <span
                                        key={summary.name}
                                        className="badge"
                                        style={{ background: "var(--bg-secondary)", color: "var(--text-secondary)" }}
                                    >
                                        {summary.name.toUpperCase()}: {summary.buy_count}B / {summary.sell_count}S / {summary.hold_count}H
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}

                    {backtestResult && (
                        <div className="glass-card" style={{ padding: 20 }}>
                            <h2 style={{ margin: 0, fontSize: "1.05rem", marginBottom: 12 }}>Backtest Result</h2>

                            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 14 }}>
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <div className="metric-label">CAGR</div>
                                    <div className="metric-value" style={{ fontSize: "1.1rem" }}>{formatPercent(backtestResult.metrics.cagr)}</div>
                                </div>
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <div className="metric-label">Sharpe</div>
                                    <div className="metric-value" style={{ fontSize: "1.1rem" }}>{formatNumber(backtestResult.metrics.sharpe, 2)}</div>
                                </div>
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <div className="metric-label">Max DD</div>
                                    <div className="metric-value" style={{ fontSize: "1.1rem" }}>{formatPercent(backtestResult.metrics.max_dd)}</div>
                                </div>
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <div className="metric-label">YTD</div>
                                    <div className="metric-value" style={{ fontSize: "1.1rem" }}>{formatPercent(backtestResult.metrics.ytd)}</div>
                                </div>
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <div className="metric-label">Win Rate</div>
                                    <div className="metric-value" style={{ fontSize: "1.1rem" }}>{formatPercent(backtestResult.metrics.win_rate)}</div>
                                </div>
                                <div className="glass-card" style={{ padding: 12 }}>
                                    <div className="metric-label">Volatility</div>
                                    <div className="metric-value" style={{ fontSize: "1.1rem" }}>{formatPercent(backtestResult.metrics.volatility)}</div>
                                </div>
                            </div>

                            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", color: "var(--text-muted)", fontSize: "0.84rem", marginBottom: 10 }}>
                                <span>Equity End: {formatNumber(backtestEquityEnd, 3)}x</span>
                                <span>Benchmark End: {formatNumber(backtestBenchEnd, 3)}x</span>
                                <span>Curve Points: {backtestResult.equity_curve.length}</span>
                                <span>Last Rebalance: {backtestResult.metrics.last_rebalance}</span>
                            </div>

                            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                {backtestResult.current_holdings.slice(0, 20).map((symbol) => (
                                    <span key={symbol} className="badge" style={{ background: "var(--accent-emerald-dim)", color: "var(--accent-emerald)" }}>
                                        {symbol}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </>
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

const secondaryButtonStyle: CSSProperties = {
    border: "1px solid var(--border-subtle)",
    background: "var(--bg-secondary)",
    color: "var(--text-primary)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 12px",
    fontSize: "0.8rem",
    fontWeight: 600,
    cursor: "pointer",
};

const dangerButtonStyle: CSSProperties = {
    border: "1px solid rgba(244,63,94,0.35)",
    background: "rgba(244,63,94,0.12)",
    color: "var(--accent-rose)",
    borderRadius: "var(--radius-sm)",
    padding: "8px 12px",
    fontSize: "0.8rem",
    fontWeight: 600,
    cursor: "pointer",
};
