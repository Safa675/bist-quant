import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Path to the Python realtime API script
const PYTHON_SCRIPT = path.resolve(
    process.cwd(),
    "..",
    "data",
    "Fetcher-Scrapper",
    "realtime_api.py"
);

interface QuoteData {
    symbol: string;
    last_price?: number;
    change?: number;
    change_pct?: number;
    volume?: number;
    bid?: number;
    ask?: number;
    high?: number;
    low?: number;
    open?: number;
    prev_close?: number;
    market_cap?: number;
    timestamp?: string;
    error?: string;
}

interface QuotesResponse {
    quotes: Record<string, QuoteData>;
    count: number;
    timestamp?: string;
    error?: string;
}

/**
 * Execute Python script and return JSON result
 */
async function executePythonScript(
    command: string,
    args: string = ""
): Promise<Record<string, unknown>> {
    return new Promise((resolve, reject) => {
        const pythonArgs = [PYTHON_SCRIPT, command];
        if (args) {
            pythonArgs.push(args);
        }

        const process = spawn("python3", pythonArgs, {
            cwd: path.dirname(PYTHON_SCRIPT),
        });

        let stdout = "";
        let stderr = "";

        process.stdout.on("data", (data) => {
            stdout += data.toString();
        });

        process.stderr.on("data", (data) => {
            stderr += data.toString();
        });

        process.on("close", (code) => {
            if (code !== 0) {
                reject(new Error(`Python script failed: ${stderr || "Unknown error"}`));
                return;
            }

            try {
                const result = JSON.parse(stdout);
                resolve(result);
            } catch {
                reject(new Error(`Failed to parse Python output: ${stdout}`));
            }
        });

        process.on("error", (err) => {
            reject(new Error(`Failed to spawn Python: ${err.message}`));
        });

        // Timeout after 30 seconds
        setTimeout(() => {
            process.kill();
            reject(new Error("Python script timeout"));
        }, 30000);
    });
}

/**
 * GET /api/realtime
 *
 * Query parameters:
 * - type: "quote" | "quotes" | "index" | "market" (default: "quotes")
 * - symbols: Comma-separated list of symbols (for "quote" and "quotes")
 * - index: Index name for "index" type (default: "XU100")
 *
 * Examples:
 * - /api/realtime?type=quote&symbols=THYAO
 * - /api/realtime?type=quotes&symbols=THYAO,AKBNK,GARAN
 * - /api/realtime?type=index&index=XU030
 * - /api/realtime?type=market
 */
export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const type = searchParams.get("type") || "quotes";
    const symbols = searchParams.get("symbols") || "";
    const index = searchParams.get("index") || "XU100";

    try {
        let result: Record<string, unknown>;

        switch (type) {
            case "quote":
                if (!symbols) {
                    return NextResponse.json(
                        { error: "Symbol required" },
                        { status: 400 }
                    );
                }
                result = await executePythonScript("quote", symbols.split(",")[0]);
                break;

            case "quotes":
                if (!symbols) {
                    return NextResponse.json(
                        { error: "Symbols required" },
                        { status: 400 }
                    );
                }
                result = await executePythonScript("quotes", symbols);
                break;

            case "index":
                result = await executePythonScript("index", index);
                break;

            case "market":
                result = await executePythonScript("market");
                break;

            default:
                return NextResponse.json(
                    { error: `Invalid type: ${type}` },
                    { status: 400 }
                );
        }

        return NextResponse.json(result, {
            headers: {
                "Cache-Control": "no-store, max-age=0",
                "X-Realtime-Type": type,
            },
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        console.error("Realtime API error:", message);

        return NextResponse.json(
            { error: message },
            {
                status: 500,
                headers: { "Cache-Control": "no-store" },
            }
        );
    }
}

/**
 * POST /api/realtime
 *
 * For portfolio snapshots with holdings data.
 *
 * Body:
 * {
 *   "holdings": { "THYAO": 100, "AKBNK": 200 },
 *   "cost_basis": { "THYAO": 250.0, "AKBNK": 45.0 }  // optional
 * }
 */
export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { holdings, cost_basis } = body;

        if (!holdings || typeof holdings !== "object") {
            return NextResponse.json(
                { error: "Holdings object required" },
                { status: 400 }
            );
        }

        // Combine holdings and cost_basis into a single JSON string for Python
        const portfolioData = JSON.stringify({
            holdings,
            cost_basis: cost_basis || {},
        });

        const result = await executePythonScript("portfolio", JSON.stringify(holdings));

        return NextResponse.json(result, {
            headers: {
                "Cache-Control": "no-store, max-age=0",
                "X-Realtime-Type": "portfolio",
            },
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        console.error("Realtime API error:", message);

        return NextResponse.json(
            { error: message },
            {
                status: 500,
                headers: { "Cache-Control": "no-store" },
            }
        );
    }
}
