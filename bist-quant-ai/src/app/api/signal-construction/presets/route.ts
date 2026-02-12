import { NextRequest, NextResponse } from "next/server";
import {
    deletePreset,
    listPresets,
    savePreset,
    type SignalConstructionConfig,
} from "@/lib/server/signalStore";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
    try {
        const presets = await listPresets();
        return NextResponse.json({ presets }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to load presets.";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}

export async function POST(request: NextRequest) {
    try {
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            return NextResponse.json({ error: "Request body must be a JSON object." }, { status: 400 });
        }

        const name = typeof (body as Record<string, unknown>).name === "string"
            ? ((body as Record<string, unknown>).name as string)
            : "";
        const config = ((body as Record<string, unknown>).config || {}) as SignalConstructionConfig;

        const preset = await savePreset({ name, config });
        return NextResponse.json({ preset }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to save preset.";
        return NextResponse.json({ error: message }, { status: 400 });
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

        const removed = await deletePreset({ id, name });
        return NextResponse.json({ removed }, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to delete preset.";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
