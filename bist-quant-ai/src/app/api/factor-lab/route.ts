import { NextRequest, NextResponse } from "next/server";
import { executeFactorLabPython, type FactorLabPayload } from "@/lib/server/factorLabPython";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
    try {
        const result = await executeFactorLabPython({ _mode: "catalog" });
        if (typeof result.error === "string" && result.error.length > 0) {
            return NextResponse.json(result, { status: 400 });
        }
        return NextResponse.json(result, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}

export async function POST(request: NextRequest) {
    try {
        const body: unknown = await request.json();
        if (!body || typeof body !== "object" || Array.isArray(body)) {
            return NextResponse.json({ error: "Request body must be a JSON object." }, { status: 400 });
        }

        const payload = body as FactorLabPayload;
        const result = await executeFactorLabPython({ ...payload, _mode: "run" });

        if (typeof result.error === "string" && result.error.length > 0) {
            return NextResponse.json(result, { status: 400 });
        }

        return NextResponse.json(result, { headers: { "Cache-Control": "no-store" } });
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error";
        return NextResponse.json({ error: message }, { status: 500 });
    }
}
