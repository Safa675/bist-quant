import { spawn } from "child_process";
import path from "path";

const PYTHON_SCRIPT = path.resolve(process.cwd(), "dashboard", "signal_construction_api.py");

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
