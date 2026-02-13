import { spawn } from "child_process";
import path from "path";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "signal_construction_api.py");
const REMOTE_ENGINE_URL = (process.env.SIGNAL_ENGINE_URL || "").trim();

// Local Python API endpoint (same Vercel deployment)
const LOCAL_PY_API = "/py-api/api/signal_construction";

function getSignalApiUrl(): string {
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

export interface SignalConstructionPayload {
    universe?: string;
    symbols?: string[] | string;
    period?: string;
    interval?: string;
    max_symbols?: number;
    top_n?: number;
    buy_threshold?: number;
    sell_threshold?: number;
    indicators?: Record<string, { enabled?: boolean; params?: Record<string, number> }>;
    _mode?: "construct" | "backtest";
}

async function executeRemoteSignalEngine(
    url: string,
    payload: SignalConstructionPayload
): Promise<Record<string, unknown>> {
    const response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        cache: "no-store",
    });

    let parsed: unknown = null;
    try {
        parsed = await response.json();
    } catch {
        parsed = { error: `Remote signal engine returned non-JSON response (${response.status}).` };
    }

    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Remote signal engine response must be a JSON object.");
    }

    const result = parsed as Record<string, unknown>;
    if (!response.ok) {
        const remoteError = typeof result.error === "string" ? result.error : `Remote engine failed (${response.status})`;
        throw new Error(remoteError);
    }
    return result;
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
        // Ignore; try line-by-line fallback.
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

export async function executeSignalPython(payload: SignalConstructionPayload): Promise<Record<string, unknown>> {
    // On Vercel, always use HTTP API (either remote or local Python function)
    if (process.env.VERCEL || REMOTE_ENGINE_URL) {
        const url = getSignalApiUrl();
        return executeRemoteSignalEngine(url, payload);
    }

    return new Promise((resolve, reject) => {
        const child = spawn("python3", [PYTHON_SCRIPT], {
            cwd: path.dirname(PYTHON_SCRIPT),
        });

        let stdout = "";
        let stderr = "";

        const timeoutHandle = setTimeout(() => {
            child.kill("SIGKILL");
            reject(new Error("Signal API timeout (180s)."));
        }, 180_000);

        child.stdout.on("data", (data: Buffer) => {
            stdout += data.toString();
        });

        child.stderr.on("data", (data: Buffer) => {
            stderr += data.toString();
        });

        child.on("error", (err) => {
            clearTimeout(timeoutHandle);
            reject(new Error(`Failed to spawn Python process: ${err.message}`));
        });

        child.on("close", (code) => {
            clearTimeout(timeoutHandle);
            if (code !== 0) {
                reject(new Error(`Python process failed (${code}): ${stderr || "unknown error"}`));
                return;
            }

            try {
                resolve(parseJsonFromStdout(stdout));
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err);
                reject(new Error(`${message}\nStderr: ${stderr || "(none)"}`));
            }
        });

        try {
            child.stdin.write(JSON.stringify(payload));
            child.stdin.end();
        } catch (err) {
            clearTimeout(timeoutHandle);
            reject(new Error(`Failed to send payload to Python process: ${String(err)}`));
        }
    });
}
