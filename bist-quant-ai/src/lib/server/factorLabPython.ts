import { spawn } from "child_process";
import path from "path";
import fallbackCatalog from "@/lib/server/factorLabFallbackCatalog.json";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "factor_lab_api.py");
const REMOTE_ENGINE_URL = (process.env.FACTOR_ENGINE_URL || "").trim();

// Local Python API endpoint (same Vercel deployment)
const LOCAL_PY_API = "/py-api/api/factor_lab";

function getApiUrl(): string {
    // Priority: external engine > local Python API
    if (REMOTE_ENGINE_URL) {
        return REMOTE_ENGINE_URL;
    }
    // Use local Python API on Vercel
    const baseUrl = process.env.VERCEL_URL
        ? `https://${process.env.VERCEL_URL}`
        : process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000";
    return `${baseUrl}${LOCAL_PY_API}`;
}

export interface FactorLabPayload {
    _mode?: "catalog" | "run";
    start_date?: string;
    end_date?: string;
    rebalance_frequency?: string;
    top_n?: number;
    factors?: Array<{
        name: string;
        enabled?: boolean;
        weight?: number;
        signal_params?: Record<string, unknown>;
    }>;
    portfolio_options?: Record<string, unknown>;
}

function cloneFallbackCatalog(): Record<string, unknown> {
    return JSON.parse(JSON.stringify(fallbackCatalog)) as Record<string, unknown>;
}

function parseJsonFromStdout(stdout: string): Record<string, unknown> {
    const trimmed = stdout.trim();
    if (!trimmed) {
        throw new Error("Python script returned empty output.");
    }

    try {
        const parsed: unknown = JSON.parse(trimmed);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
        }
    } catch {
        // Ignore; try line fallback.
    }

    const lines = trimmed
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .reverse();

    for (const line of lines) {
        try {
            const parsed: unknown = JSON.parse(line);
            if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                return parsed as Record<string, unknown>;
            }
        } catch {
            continue;
        }
    }

    throw new Error(`Failed to parse Python output as JSON: ${trimmed}`);
}

async function executeRemote(url: string, payload: FactorLabPayload): Promise<Record<string, unknown>> {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        cache: "no-store",
    });

    let data: unknown;
    try {
        data = await response.json();
    } catch {
        throw new Error(`Remote factor engine returned non-JSON response (${response.status}).`);
    }

    if (!data || typeof data !== "object" || Array.isArray(data)) {
        throw new Error("Remote factor engine response must be a JSON object.");
    }

    const result = data as Record<string, unknown>;
    if (!response.ok) {
        const msg = typeof result.error === "string" ? result.error : `Remote factor engine failed (${response.status})`;
        throw new Error(msg);
    }

    return result;
}

export async function executeFactorLabPython(payload: FactorLabPayload): Promise<Record<string, unknown>> {
    // On Vercel, always use HTTP API (either remote or local Python function)
    if (process.env.VERCEL || REMOTE_ENGINE_URL) {
        const url = getApiUrl();
        return executeRemote(url, payload);
    }

    return new Promise((resolve, reject) => {
        const child = spawn("python3", [PYTHON_SCRIPT], {
            cwd: path.dirname(PYTHON_SCRIPT),
        });

        let stdout = "";
        let stderr = "";

        const timeout = setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error("Factor lab timeout (300s)."));
        }, 300_000);

        child.stdout.on("data", (chunk: Buffer) => {
            stdout += chunk.toString();
        });

        child.stderr.on("data", (chunk: Buffer) => {
            stderr += chunk.toString();
        });

        child.on("error", (err) => {
            clearTimeout(timeout);
            reject(new Error(`Failed to spawn Python process: ${err.message}`));
        });

        child.on("close", (code) => {
            clearTimeout(timeout);
            if (code !== 0) {
                reject(new Error(`Python process failed (${code}): ${stderr || "unknown error"}`));
                return;
            }

            try {
                resolve(parseJsonFromStdout(stdout));
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                reject(new Error(`${msg}\nStderr: ${stderr || "(none)"}`));
            }
        });

        child.stdin.write(JSON.stringify(payload));
        child.stdin.end();
    });
}
