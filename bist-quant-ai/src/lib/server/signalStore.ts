import { existsSync } from "fs";
import { mkdir, readFile, writeFile } from "fs/promises";
import { join } from "path";

const DATA_DIR = process.env.VERCEL
    ? "/tmp/bist-quant-ai-data"
    : join(process.cwd(), "data");
const STORE_PATH = join(DATA_DIR, "signal_store.json");

export interface IndicatorPayload {
    enabled?: boolean;
    params?: Record<string, number>;
}

export interface SignalConstructionConfig {
    universe?: string;
    symbols?: string[] | string;
    period?: string;
    interval?: string;
    max_symbols?: number;
    top_n?: number;
    buy_threshold?: number;
    sell_threshold?: number;
    indicators?: Record<string, IndicatorPayload>;
}

export interface BacktestSummary {
    cagr: number;
    sharpe: number;
    beta: number | null;
    max_dd: number;
    ytd: number;
    total_return?: number;
    volatility?: number;
    win_rate?: number;
    last_rebalance?: string;
}

export interface StoredPreset {
    id: string;
    name: string;
    config: SignalConstructionConfig;
    created_at: string;
    updated_at: string;
}

export interface StoredPublishedSignal {
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
    enabled: boolean;
    holdings: string[];
    config: SignalConstructionConfig;
    backtest?: BacktestSummary;
    signal: {
        name: string;
        enabled: boolean;
        cagr: number;
        sharpe: number;
        beta: number | null;
        max_dd: number;
        ytd: number;
        last_rebalance: string;
        source: string;
    };
}

interface SignalStore {
    presets: StoredPreset[];
    published_signals: StoredPublishedSignal[];
}

const DEFAULT_STORE: SignalStore = {
    presets: [],
    published_signals: [],
};

function toIsoNow(): string {
    return new Date().toISOString();
}

function slugify(value: string): string {
    return value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .slice(0, 64) || "signal";
}

function asFiniteNumber(value: unknown, fallback = 0): number {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) return parsed;
    }
    return fallback;
}

function parseStore(raw: unknown): SignalStore {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
        return { ...DEFAULT_STORE };
    }

    const root = raw as Record<string, unknown>;
    const presets = Array.isArray(root.presets) ? (root.presets as StoredPreset[]) : [];
    const publishedSignals = Array.isArray(root.published_signals)
        ? (root.published_signals as StoredPublishedSignal[])
        : [];

    return {
        presets,
        published_signals: publishedSignals,
    };
}

async function ensureStoreDir(): Promise<void> {
    await mkdir(DATA_DIR, { recursive: true });
}

async function readStore(): Promise<SignalStore> {
    await ensureStoreDir();
    if (!existsSync(STORE_PATH)) {
        return { ...DEFAULT_STORE };
    }

    const raw = await readFile(STORE_PATH, "utf-8");
    try {
        const parsed: unknown = JSON.parse(raw);
        return parseStore(parsed);
    } catch {
        return { ...DEFAULT_STORE };
    }
}

async function writeStore(store: SignalStore): Promise<void> {
    await ensureStoreDir();
    await writeFile(STORE_PATH, JSON.stringify(store, null, 2), "utf-8");
}

export async function listPresets(): Promise<StoredPreset[]> {
    const store = await readStore();
    return [...store.presets].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

export async function savePreset(input: {
    name: string;
    config: SignalConstructionConfig;
}): Promise<StoredPreset> {
    const name = input.name.trim();
    if (!name) {
        throw new Error("Preset name is required.");
    }

    const store = await readStore();
    const now = toIsoNow();
    const existingIndex = store.presets.findIndex((preset) => preset.name.toLowerCase() === name.toLowerCase());

    if (existingIndex >= 0) {
        const updated: StoredPreset = {
            ...store.presets[existingIndex],
            name,
            config: input.config,
            updated_at: now,
        };
        store.presets[existingIndex] = updated;
        await writeStore(store);
        return updated;
    }

    const created: StoredPreset = {
        id: `${slugify(name)}_${Date.now()}`,
        name,
        config: input.config,
        created_at: now,
        updated_at: now,
    };

    store.presets.push(created);
    await writeStore(store);
    return created;
}

export async function deletePreset(input: { id?: string; name?: string }): Promise<boolean> {
    const store = await readStore();
    const initialLength = store.presets.length;

    const targetId = input.id?.trim();
    const targetName = input.name?.trim().toLowerCase();

    store.presets = store.presets.filter((preset) => {
        if (targetId && preset.id === targetId) return false;
        if (targetName && preset.name.toLowerCase() === targetName) return false;
        return true;
    });

    const changed = store.presets.length !== initialLength;
    if (changed) {
        await writeStore(store);
    }
    return changed;
}

function coerceHoldings(values: unknown): string[] {
    if (!Array.isArray(values)) return [];
    const seen = new Set<string>();
    const out: string[] = [];

    for (const item of values) {
        const normalized = String(item ?? "")
            .trim()
            .toUpperCase()
            .replace(/\.IS$/, "");
        if (!normalized || seen.has(normalized)) continue;
        seen.add(normalized);
        out.push(normalized);
    }

    return out;
}

function coerceBacktest(input: unknown): BacktestSummary | undefined {
    if (!input || typeof input !== "object" || Array.isArray(input)) return undefined;
    const raw = input as Record<string, unknown>;

    return {
        cagr: asFiniteNumber(raw.cagr),
        sharpe: asFiniteNumber(raw.sharpe),
        beta: raw.beta === null ? null : asFiniteNumber(raw.beta, 0),
        max_dd: asFiniteNumber(raw.max_dd),
        ytd: asFiniteNumber(raw.ytd),
        total_return: asFiniteNumber(raw.total_return),
        volatility: asFiniteNumber(raw.volatility),
        win_rate: asFiniteNumber(raw.win_rate),
        last_rebalance: typeof raw.last_rebalance === "string" ? raw.last_rebalance : undefined,
    };
}

export async function listPublishedSignals(): Promise<StoredPublishedSignal[]> {
    const store = await readStore();
    return [...store.published_signals].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

export async function savePublishedSignal(input: {
    name: string;
    holdings: string[];
    config: SignalConstructionConfig;
    backtest?: BacktestSummary | Record<string, unknown>;
}): Promise<StoredPublishedSignal> {
    const name = input.name.trim();
    if (!name) {
        throw new Error("Signal name is required.");
    }

    const holdings = coerceHoldings(input.holdings);
    if (holdings.length === 0) {
        throw new Error("At least one selected symbol is required to publish.");
    }

    const backtest = coerceBacktest(input.backtest);
    const now = toIsoNow();
    const signalName = slugify(name);

    const signalPayload = {
        name: signalName,
        enabled: true,
        cagr: backtest?.cagr ?? 0,
        sharpe: backtest?.sharpe ?? 0,
        beta: backtest?.beta ?? null,
        max_dd: backtest?.max_dd ?? 0,
        ytd: backtest?.ytd ?? 0,
        last_rebalance: (backtest?.last_rebalance || now).slice(0, 10),
        source: "signal_builder",
    };

    const store = await readStore();
    const existingIndex = store.published_signals.findIndex((item) => item.name.toLowerCase() === name.toLowerCase());

    if (existingIndex >= 0) {
        const existing = store.published_signals[existingIndex];
        const updated: StoredPublishedSignal = {
            ...existing,
            name,
            holdings,
            config: input.config,
            backtest,
            signal: signalPayload,
            enabled: true,
            updated_at: now,
        };
        store.published_signals[existingIndex] = updated;
        await writeStore(store);
        return updated;
    }

    const created: StoredPublishedSignal = {
        id: `${signalName}_${Date.now()}`,
        name,
        holdings,
        config: input.config,
        backtest,
        signal: signalPayload,
        enabled: true,
        created_at: now,
        updated_at: now,
    };

    store.published_signals.push(created);
    await writeStore(store);
    return created;
}

export async function setPublishedSignalEnabled(input: {
    id?: string;
    name?: string;
    enabled: boolean;
}): Promise<StoredPublishedSignal | null> {
    const store = await readStore();
    const targetId = input.id?.trim();
    const targetName = input.name?.trim().toLowerCase();

    const index = store.published_signals.findIndex((item) => {
        if (targetId && item.id === targetId) return true;
        if (targetName && item.name.toLowerCase() === targetName) return true;
        return false;
    });

    if (index < 0) {
        return null;
    }

    const now = toIsoNow();
    const updated: StoredPublishedSignal = {
        ...store.published_signals[index],
        enabled: input.enabled,
        signal: {
            ...store.published_signals[index].signal,
            enabled: input.enabled,
        },
        updated_at: now,
    };

    store.published_signals[index] = updated;
    await writeStore(store);
    return updated;
}

export async function deletePublishedSignal(input: { id?: string; name?: string }): Promise<boolean> {
    const store = await readStore();
    const initialLength = store.published_signals.length;

    const targetId = input.id?.trim();
    const targetName = input.name?.trim().toLowerCase();

    store.published_signals = store.published_signals.filter((item) => {
        if (targetId && item.id === targetId) return false;
        if (targetName && item.name.toLowerCase() === targetName) return false;
        return true;
    });

    const changed = store.published_signals.length !== initialLength;
    if (changed) {
        await writeStore(store);
    }
    return changed;
}
