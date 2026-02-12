import { NextRequest, NextResponse } from "next/server";
import { executeSignalPython, type SignalConstructionPayload } from "@/lib/server/signalConstructionPython";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
    return NextResponse.json({
        universes: ["XU030", "XU100", "XUTUM", "CUSTOM"],
        periods: ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        intervals: ["1d"],
        indicators: [
            { key: "rsi", label: "RSI" },
            { key: "macd", label: "MACD Histogram" },
            { key: "bollinger", label: "Bollinger %B" },
            { key: "atr", label: "ATR (Cross-Sectional)" },
            { key: "stochastic", label: "Stochastic %K" },
            { key: "adx", label: "ADX (+DI/-DI trend)" },
            { key: "supertrend", label: "Supertrend Direction" },
        ],
    });
}

export async function POST(request: NextRequest) {
    try {
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            return NextResponse.json({ error: "Request body must be a JSON object." }, { status: 400 });
        }

        const payload = body as SignalConstructionPayload;
        const result = await executeSignalPython({ ...payload, _mode: "construct" });

        if (typeof result.error === "string" && result.error.length > 0) {
            return NextResponse.json(result, { status: 400 });
        }

        return NextResponse.json(result, {
            headers: {
                "Cache-Control": "no-store, max-age=0",
            },
        });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json(
            { error: message },
            {
                status: 500,
                headers: {
                    "Cache-Control": "no-store",
                },
            }
        );
    }
}
