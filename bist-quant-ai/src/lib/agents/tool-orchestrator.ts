/**
 * Tool-Enabled Agent Orchestrator
 *
 * Extends the base orchestrator with function calling capabilities
 * for Borsa MCP integration. Enables LLM agents to:
 * - Screen stocks with fundamental/technical filters
 * - Fetch financial statements and ratios
 * - Run technical scans
 * - Access fund and macro data
 */

import { logError, logInfo } from "./logging";
import {
    BORSA_MCP_TOOL_DEFINITIONS,
    executeMCPTool,
    type MCPToolCall,
    type MCPToolResult,
} from "./borsa-mcp-client";

// Re-export types from base orchestrator
export type { AgentContext, AgentMessage } from "./orchestrator";
import type { AgentContext } from "./orchestrator";
import { buildContext, getAzureOpenAIConfig, validateQuery } from "./orchestrator";

export type ResearchAgentType = "research";

interface ToolCall {
    id: string;
    type: "function";
    function: {
        name: string;
        arguments: string;
    };
}

interface AzureUsage {
    promptTokens: number | null;
    completionTokens: number | null;
    totalTokens: number | null;
}

interface ToolOrchestrationResult {
    response: string;
    toolsUsed: string[];
    toolResults: Record<string, MCPToolResult>;
    usage: AzureUsage;
}

interface ToolOrchestrationOptions {
    requestId?: string;
    maxToolCalls?: number;
    maxRetries?: number;
}

const RESEARCH_SYSTEM_PROMPT = `You are a Research Analyst AI agent for Quant AI Platform.
You have access to real-time BIST market data via function calling tools.

Your capabilities:
- Screen stocks using fundamental filters (P/E, P/B, dividend yield, market cap)
- Screen stocks using technical filters (RSI oversold/overbought, momentum)
- Fetch detailed financial statements (balance sheet, income statement, cash flow)
- Get financial ratios (valuation, profitability, liquidity metrics)
- Run technical scans (MACD signals, Supertrend, golden/death crosses)
- Compare sectors by various metrics
- Access TEFAS fund data (836+ funds)
- Get KAP news and announcements

When answering questions:
1. Use the appropriate tools to fetch real data
2. Analyze the data and provide insights
3. Be specific with numbers and metrics
4. Highlight key findings and actionable insights
5. Compare results when relevant (vs index, vs sector)

Available screening presets: value_stocks, growth_stocks, high_dividend, low_pe, high_momentum, oversold, overbought, breakout, undervalued, quality, small_cap, large_cap

Available scan types: oversold, overbought, macd_bullish, macd_bearish, supertrend_buy, supertrend_sell, golden_cross, death_cross

Respond in a clear, analytical style suitable for retail investors.`;

const MAX_TOOL_ITERATIONS = 5;
const DEFAULT_MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function _extractUsage(payload: unknown): AzureUsage {
    const usage = (payload as { usage?: { prompt_tokens?: unknown; completion_tokens?: unknown; total_tokens?: unknown } })?.usage;
    const toNumOrNull = (value: unknown) => {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    };
    return {
        promptTokens: toNumOrNull(usage?.prompt_tokens),
        completionTokens: toNumOrNull(usage?.completion_tokens),
        totalTokens: toNumOrNull(usage?.total_tokens),
    };
}

function _extractToolCalls(payload: unknown): ToolCall[] {
    const choices = (payload as { choices?: Array<{ message?: { tool_calls?: ToolCall[] } }> })?.choices;
    return choices?.[0]?.message?.tool_calls || [];
}

function _extractAssistantText(payload: unknown): string {
    const choices = (payload as { choices?: Array<{ message?: { content?: string | null } }> })?.choices;
    const content = choices?.[0]?.message?.content;
    return typeof content === "string" ? content.trim() : "";
}

function _extractFinishReason(payload: unknown): string {
    const choices = (payload as { choices?: Array<{ finish_reason?: string }> })?.choices;
    return choices?.[0]?.finish_reason || "";
}

/**
 * Generate a research response with tool calling
 */
export async function generateResearchResponse(
    query: string,
    context: AgentContext,
    options: ToolOrchestrationOptions = {}
): Promise<ToolOrchestrationResult> {
    const validation = validateQuery(query);
    if (!validation.valid) {
        throw new Error(validation.error || "Invalid query.");
    }
    const sanitizedQuery = validation.sanitized!;

    const cfg = getAzureOpenAIConfig();
    const url = `${cfg.endpoint}/openai/deployments/${encodeURIComponent(cfg.deployment)}/chat/completions` +
        `?api-version=${encodeURIComponent(cfg.apiVersion)}`;

    const contextStr = buildContext(context);
    const maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;

    const toolsUsed: string[] = [];
    const toolResults: Record<string, MCPToolResult> = {};

    // Build initial messages
    const messages: Array<{ role: string; content?: string; tool_call_id?: string; tool_calls?: ToolCall[] }> = [
        { role: "system", content: RESEARCH_SYSTEM_PROMPT },
        {
            role: "user",
            content: `Portfolio Context:\n${contextStr}\n\nUser Question:\n${sanitizedQuery}`,
        },
    ];

    let totalUsage: AzureUsage = {
        promptTokens: null,
        completionTokens: null,
        totalTokens: null,
    };

    logInfo("research.agent.start", {
        requestId: options.requestId || null,
        queryChars: sanitizedQuery.length,
        toolCount: BORSA_MCP_TOOL_DEFINITIONS.length,
    });

    // Tool calling loop
    for (let iteration = 0; iteration < MAX_TOOL_ITERATIONS; iteration++) {
        const startedAt = Date.now();

        let parsed: unknown = null;
        let lastError: Error | null = null;

        // Retry loop for API calls
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                const res = await fetch(url, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "api-key": cfg.apiKey,
                    },
                    body: JSON.stringify({
                        messages,
                        tools: BORSA_MCP_TOOL_DEFINITIONS,
                        tool_choice: "auto",
                        max_completion_tokens: 1500,
                    }),
                    cache: "no-store",
                });

                const raw = await res.text();
                try {
                    parsed = JSON.parse(raw);
                } catch {
                    throw new Error(`Invalid JSON response: ${raw.slice(0, 200)}`);
                }

                if (!res.ok) {
                    const errPayload = parsed as { error?: { message?: string } } | null;
                    const upstreamMessage = errPayload?.error?.message || raw.slice(0, 400);
                    throw new Error(`Azure OpenAI error (${res.status}): ${upstreamMessage}`);
                }

                lastError = null;
                break;
            } catch (error) {
                lastError = error instanceof Error ? error : new Error(String(error));
                if (attempt < maxRetries) {
                    await sleep(RETRY_DELAY_MS * attempt);
                }
            }
        }

        if (lastError) {
            throw lastError;
        }

        const latencyMs = Date.now() - startedAt;
        const usage = _extractUsage(parsed);
        const finishReason = _extractFinishReason(parsed);
        const toolCalls = _extractToolCalls(parsed);

        // Accumulate usage
        if (usage.promptTokens !== null) {
            totalUsage.promptTokens = (totalUsage.promptTokens || 0) + usage.promptTokens;
        }
        if (usage.completionTokens !== null) {
            totalUsage.completionTokens = (totalUsage.completionTokens || 0) + usage.completionTokens;
        }
        if (usage.totalTokens !== null) {
            totalUsage.totalTokens = (totalUsage.totalTokens || 0) + usage.totalTokens;
        }

        logInfo("research.agent.iteration", {
            requestId: options.requestId || null,
            iteration,
            latencyMs,
            finishReason,
            toolCallCount: toolCalls.length,
            promptTokens: usage.promptTokens,
            completionTokens: usage.completionTokens,
        });

        // If no tool calls, we're done
        if (finishReason === "stop" || toolCalls.length === 0) {
            const responseText = _extractAssistantText(parsed);

            logInfo("research.agent.complete", {
                requestId: options.requestId || null,
                iterations: iteration + 1,
                toolsUsed,
                responseChars: responseText.length,
                totalPromptTokens: totalUsage.promptTokens,
                totalCompletionTokens: totalUsage.completionTokens,
            });

            return {
                response: responseText || "I couldn't generate a response. Please try again.",
                toolsUsed,
                toolResults,
                usage: totalUsage,
            };
        }

        // Process tool calls
        const assistantMessage: { role: string; content: string | null; tool_calls: ToolCall[] } = {
            role: "assistant",
            content: null,
            tool_calls: toolCalls,
        };
        messages.push(assistantMessage);

        // Execute tool calls in parallel
        const toolPromises = toolCalls.map(async (tc) => {
            const toolName = tc.function.name;
            let params: Record<string, unknown> = {};

            try {
                params = JSON.parse(tc.function.arguments || "{}");
            } catch {
                params = {};
            }

            toolsUsed.push(toolName);

            logInfo("research.agent.tool_call", {
                requestId: options.requestId || null,
                tool: toolName,
                params,
                iteration,
            });

            const mcpCall: MCPToolCall = {
                tool: toolName,
                params,
            };

            const result = await executeMCPTool(mcpCall, options.requestId);
            toolResults[`${toolName}_${tc.id}`] = result;

            return {
                id: tc.id,
                toolName,
                result,
            };
        });

        const toolResponses = await Promise.all(toolPromises);

        // Add tool results to messages
        for (const tr of toolResponses) {
            const content = tr.result.success
                ? JSON.stringify(tr.result.data, null, 2)
                : `Error: ${tr.result.error}`;

            messages.push({
                role: "tool",
                tool_call_id: tr.id,
                content,
            });

            logInfo("research.agent.tool_result", {
                requestId: options.requestId || null,
                tool: tr.toolName,
                success: tr.result.success,
                latencyMs: tr.result.latencyMs,
                iteration,
            });
        }
    }

    // Max iterations reached
    logError("research.agent.max_iterations", {
        requestId: options.requestId || null,
        maxIterations: MAX_TOOL_ITERATIONS,
        toolsUsed,
    });

    return {
        response: "I reached the maximum number of tool calls. Here's what I found with the data gathered so far.",
        toolsUsed,
        toolResults,
        usage: totalUsage,
    };
}

/**
 * Simple tool execution without full conversation
 * Useful for one-off data fetches
 */
export async function executeToolDirect(
    toolName: string,
    params: Record<string, unknown>,
    requestId?: string
): Promise<MCPToolResult> {
    const mcpCall: MCPToolCall = {
        tool: toolName,
        params,
    };

    return executeMCPTool(mcpCall, requestId);
}
