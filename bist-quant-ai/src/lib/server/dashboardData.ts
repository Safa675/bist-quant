import { execFile } from "child_process";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { promisify } from "util";
import { listPublishedSignals, type StoredPublishedSignal } from "@/lib/server/signalStore";

const execFileAsync = promisify(execFile);

const PROJECT_ROOT = process.cwd();
const GENERATOR_PATH = join(PROJECT_ROOT, "dashboard", "generate_dashboard_data.py");
const SNAPSHOT_PATH = join(PROJECT_ROOT, "public", "data", "dashboard_data.json");
const PYTHON_CANDIDATES = ["python3", "python"] as const;
const REFRESH_THROTTLE_MS = 10_000;

let lastRefreshAtMs = 0;

function mergePublishedSignals(
    snapshot: Record<string, unknown>,
    publishedSignals: StoredPublishedSignal[]
): Record<string, unknown> {
    if (!publishedSignals.length) {
        return snapshot;
    }

    const signals = Array.isArray(snapshot.signals)
        ? [...(snapshot.signals as Array<Record<string, unknown>>)]
        : [];
    const holdingsRoot =
        snapshot.holdings && typeof snapshot.holdings === "object" && !Array.isArray(snapshot.holdings)
            ? (snapshot.holdings as Record<string, string[]>)
            : {};
    const holdings: Record<string, string[]> = { ...holdingsRoot };
    const allPublishedNames = new Set<string>(
        publishedSignals
            .map((item) => item.signal?.name)
            .filter((name): name is string => typeof name === "string" && name.length > 0)
    );

    const filteredSignals = signals.filter((item) => !allPublishedNames.has(String(item?.name || "")));
    for (const name of allPublishedNames) {
        delete holdings[name];
    }

    const enabledPublishedSignals = publishedSignals.filter(
        (item) => item.enabled !== false && item.signal?.enabled !== false
    );
    for (const published of enabledPublishedSignals) {
        const signalName = published.signal.name;
        filteredSignals.push({ ...published.signal, enabled: true });
        holdings[signalName] = [...published.holdings];
    }

    filteredSignals.sort((a, b) => {
        const aCagr = typeof a.cagr === "number" ? a.cagr : Number.NEGATIVE_INFINITY;
        const bCagr = typeof b.cagr === "number" ? b.cagr : Number.NEGATIVE_INFINITY;
        return bCagr - aCagr;
    });

    return {
        ...snapshot,
        signals: filteredSignals,
        holdings,
        active_signals: filteredSignals.filter((item) => item?.enabled !== false).length,
        displayed_signals: filteredSignals.length,
        published_signals_count: publishedSignals.length,
        published_signals_enabled_count: enabledPublishedSignals.length,
    };
}

function readSnapshot(): Record<string, unknown> {
    if (!existsSync(SNAPSHOT_PATH)) {
        throw new Error(`Dashboard snapshot not found: ${SNAPSHOT_PATH}`);
    }

    const raw = readFileSync(SNAPSHOT_PATH, "utf-8");
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Dashboard snapshot root must be a JSON object.");
    }
    return parsed as Record<string, unknown>;
}

async function runGenerator(): Promise<void> {
    if (!existsSync(GENERATOR_PATH)) {
        throw new Error(`Dashboard generator not found: ${GENERATOR_PATH}`);
    }

    let lastError: unknown = null;
    for (const pythonCmd of PYTHON_CANDIDATES) {
        try {
            await execFileAsync(pythonCmd, [GENERATOR_PATH], {
                cwd: PROJECT_ROOT,
                timeout: 120_000,
                maxBuffer: 10 * 1024 * 1024,
            });
            return;
        } catch (error) {
            lastError = error;
        }
    }

    const detail = lastError instanceof Error ? lastError.message : String(lastError);
    throw new Error(`Failed to run dashboard generator with python3/python: ${detail}`);
}

export async function loadDashboardData(options?: {
    refresh?: boolean;
    force?: boolean;
}): Promise<{
    data: Record<string, unknown>;
    refreshed: boolean;
    refreshError?: string;
}> {
    const wantsRefresh = options?.refresh === true;
    const forceRefresh = options?.force === true;
    let refreshed = false;
    let refreshError: string | undefined;

    if (wantsRefresh) {
        const now = Date.now();
        const canRefresh = forceRefresh || now - lastRefreshAtMs >= REFRESH_THROTTLE_MS;
        if (canRefresh) {
            try {
                await runGenerator();
                refreshed = true;
                lastRefreshAtMs = now;
            } catch (error) {
                refreshError = error instanceof Error ? error.message : String(error);
            }
        }
    }

    const snapshot = readSnapshot();
    const publishedSignals = await listPublishedSignals().catch(() => []);
    const data = mergePublishedSignals(snapshot, publishedSignals);

    return {
        data,
        refreshed,
        refreshError,
    };
}
