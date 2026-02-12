import { NextRequest, NextResponse } from "next/server";
import {
    deletePublishedSignal,
    listPublishedSignals,
    setPublishedSignalEnabled,
    savePublishedSignal,
    type BacktestSummary,
    type SignalConstructionConfig,
} from "@/lib/server/signalStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
    try {
        const signals = await listPublishedSignals();
        return NextResponse.json({ signals }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to load published signals.";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}

export async function POST(request: NextRequest) {
    try {
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            return NextResponse.json({ error: "Request body must be a JSON object." }, { status: 400 });
        }

        const payload = body as Record<string, unknown>;
        const name = typeof payload.name === "string" ? payload.name : "";
        const holdings = Array.isArray(payload.holdings) ? (payload.holdings as string[]) : [];
        const config = (payload.config || {}) as SignalConstructionConfig;
        const backtest = (payload.backtest || undefined) as BacktestSummary | undefined;

        const published = await savePublishedSignal({
            name,
            holdings,
            config,
            backtest,
        });

        return NextResponse.json({ published }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to publish signal.";
        return NextResponse.json({ error: message }, { status: 400 });
    }
}

export async function PATCH(request: NextRequest) {
    try {
        const body: unknown = await request.json();
        const payload = body && typeof body === "object" && !Array.isArray(body)
            ? (body as Record<string, unknown>)
            : {};

        const id = typeof payload.id === "string" ? payload.id : undefined;
        const name = typeof payload.name === "string" ? payload.name : undefined;
        const enabled = payload.enabled;

        if (!id && !name) {
            return NextResponse.json({ error: "Either `id` or `name` is required." }, { status: 400 });
        }
        if (typeof enabled !== "boolean") {
            return NextResponse.json({ error: "`enabled` must be a boolean." }, { status: 400 });
        }

        const updated = await setPublishedSignalEnabled({ id, name, enabled });
        if (!updated) {
            return NextResponse.json({ error: "Published signal not found." }, { status: 404 });
        }

        return NextResponse.json({ updated }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to update published signal.";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}

export async function DELETE(request: NextRequest) {
    try {
        const body: unknown = await request.json();
        const payload = body && typeof body === "object" && !Array.isArray(body)
            ? (body as Record<string, unknown>)
            : {};

        const id = typeof payload.id === "string" ? payload.id : undefined;
        const name = typeof payload.name === "string" ? payload.name : undefined;
        if (!id && !name) {
            return NextResponse.json({ error: "Either `id` or `name` is required." }, { status: 400 });
        }

        const removed = await deletePublishedSignal({ id, name });
        return NextResponse.json({ removed }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to unpublish signal.";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
