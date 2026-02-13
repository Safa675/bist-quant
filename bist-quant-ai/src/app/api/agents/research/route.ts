import { NextRequest, NextResponse } from "next/server";
import { createRequestId, logError, logInfo, safeErrorMessage } from "@/lib/agents/logging";
import { generateResearchResponse } from "@/lib/agents/tool-orchestrator";
import { buildFallbackContext, loadDashboardData, resolveContext } from "../_shared";

export async function POST(request: NextRequest) {
    const requestId = createRequestId();
    const startedAt = Date.now();

    try {
        let body: unknown = null;
        try {
            body = await request.json();
        } catch {
            return NextResponse.json(
                {
                    error: "Request body must be valid JSON.",
                    request_id: requestId,
                },
                { status: 400 }
            );
        }

        const payload = (body || {}) as { query?: unknown; context?: unknown };
        const query = typeof payload.query === "string" ? payload.query.trim() : "";

        if (!query) {
            return NextResponse.json(
                {
                    error: "Missing 'query' in request body.",
                    request_id: requestId,
                },
                { status: 400 }
            );
        }

        const data = await loadDashboardData();
        const fallbackContext = buildFallbackContext(data);
        const context = resolveContext(payload.context, fallbackContext);

        logInfo("research.request.received", {
            requestId,
            agent: "research",
            path: request.nextUrl.pathname,
            queryChars: query.length,
            regime: context.regime,
            signalCount: context.signals.length,
            holdingsFactorCount: Object.keys(context.holdings).length,
        });

        const result = await generateResearchResponse(query, context, { requestId });
        const latencyMs = Date.now() - startedAt;

        logInfo("research.request.completed", {
            requestId,
            agent: "research",
            path: request.nextUrl.pathname,
            latencyMs,
            responseChars: result.response.length,
            toolsUsed: result.toolsUsed,
            toolCount: result.toolsUsed.length,
        });

        return NextResponse.json({
            mode: "llm_live",
            request_id: requestId,
            agent: "research",
            query,
            response: result.response,
            tools_used: result.toolsUsed,
            tool_results: result.toolResults,
            context: {
                regime: context.regime,
                signalCount: context.signals.length,
                holdingsFactorCount: Object.keys(context.holdings).length,
            },
            usage: result.usage,
            timestamp: new Date().toISOString(),
        });
    } catch (error) {
        const message = safeErrorMessage(error);
        const latencyMs = Date.now() - startedAt;

        logError("research.request.failed", {
            requestId,
            agent: "research",
            path: request.nextUrl.pathname,
            latencyMs,
            message,
        });

        return NextResponse.json(
            {
                error: `Research agent failed: ${message}`,
                request_id: requestId,
            },
            { status: 500 }
        );
    }
}
