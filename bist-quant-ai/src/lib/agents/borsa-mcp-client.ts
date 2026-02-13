/**
 * Borsa MCP Client for LLM Agent Tool Calling
 *
 * Integrates with borsamcp.fastmcp.app for:
 * - Stock screening (23 presets + custom filters)
 * - Financial statements & ratios
 * - Technical analysis
 * - Fund data (TEFAS)
 * - Crypto & FX data
 */

import { logError, logInfo } from "./logging";

const MCP_ENDPOINT = "https://borsamcp.fastmcp.app/mcp";
const MCP_TIMEOUT_MS = 30000;

export interface MCPToolCall {
    tool: string;
    params: Record<string, unknown>;
}

export interface MCPToolResult {
    success: boolean;
    data?: unknown;
    error?: string;
    latencyMs: number;
}

/**
 * Available Borsa MCP Tools for LLM Agents
 */
export const BORSA_MCP_TOOLS = {
    // Stock fundamentals
    GET_PROFILE: "get_profile",
    GET_QUICK_INFO: "get_quick_info",
    GET_FINANCIAL_STATEMENTS: "get_financial_statements",
    GET_FINANCIAL_RATIOS: "get_financial_ratios",
    GET_DIVIDENDS: "get_dividends",
    GET_EARNINGS: "get_earnings",
    GET_CORPORATE_ACTIONS: "get_corporate_actions",
    GET_ANALYST_DATA: "get_analyst_data",

    // Screening & scanning
    SCREEN_SECURITIES: "screen_securities",
    SCAN_STOCKS: "scan_stocks",
    SEARCH_SYMBOL: "search_symbol",

    // Technical analysis
    GET_HISTORICAL_DATA: "get_historical_data",
    GET_TECHNICAL_ANALYSIS: "get_technical_analysis",
    GET_PIVOT_POINTS: "get_pivot_points",

    // News
    GET_NEWS: "get_news",

    // Funds & indices
    GET_FUND_DATA: "get_fund_data",
    GET_INDEX_DATA: "get_index_data",

    // Macro & FX
    GET_FX_DATA: "get_fx_data",
    GET_ECONOMIC_CALENDAR: "get_economic_calendar",
    GET_BOND_YIELDS: "get_bond_yields",
    GET_MACRO_DATA: "get_macro_data",
    GET_SECTOR_COMPARISON: "get_sector_comparison",

    // Crypto
    GET_CRYPTO_MARKET: "get_crypto_market",

    // Help
    GET_SCREENER_HELP: "get_screener_help",
    GET_SCANNER_HELP: "get_scanner_help",
} as const;

/**
 * Tool definitions for Azure OpenAI function calling
 */
export const BORSA_MCP_TOOL_DEFINITIONS = [
    {
        type: "function",
        function: {
            name: "screen_securities",
            description: "Screen BIST stocks using fundamental and technical filters. Returns matching stocks with key metrics. Use presets like 'value_stocks', 'growth_stocks', 'high_dividend', 'low_pe', 'momentum' or custom filters.",
            parameters: {
                type: "object",
                properties: {
                    preset: {
                        type: "string",
                        description: "Screening preset: value_stocks, growth_stocks, high_dividend, low_pe, high_momentum, oversold, overbought, breakout, undervalued, quality, small_cap, large_cap, etc.",
                    },
                    index: {
                        type: "string",
                        description: "Filter by index: XU100, XU030, XBANK, XUSIN, etc. Default: all BIST stocks",
                    },
                    pe_max: {
                        type: "number",
                        description: "Maximum P/E ratio filter",
                    },
                    pe_min: {
                        type: "number",
                        description: "Minimum P/E ratio filter",
                    },
                    pb_max: {
                        type: "number",
                        description: "Maximum P/B ratio filter",
                    },
                    div_yield_min: {
                        type: "number",
                        description: "Minimum dividend yield % filter",
                    },
                    market_cap_min: {
                        type: "number",
                        description: "Minimum market cap in TRY",
                    },
                    rsi_max: {
                        type: "number",
                        description: "Maximum RSI (for oversold stocks, use 30)",
                    },
                    rsi_min: {
                        type: "number",
                        description: "Minimum RSI (for overbought stocks, use 70)",
                    },
                },
                required: [],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_financial_statements",
            description: "Get detailed financial statements for a BIST stock including balance sheet, income statement, and cash flow statement.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker symbol (e.g., THYAO, AKBNK, GARAN)",
                    },
                    statement_type: {
                        type: "string",
                        enum: ["balance_sheet", "income_statement", "cash_flow", "all"],
                        description: "Type of financial statement to retrieve",
                    },
                    period: {
                        type: "string",
                        enum: ["annual", "quarterly"],
                        description: "Reporting period. Default: annual",
                    },
                },
                required: ["symbol"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_financial_ratios",
            description: "Get comprehensive financial ratios for a BIST stock including valuation, profitability, liquidity, and health metrics.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker symbol (e.g., THYAO, AKBNK, GARAN)",
                    },
                },
                required: ["symbol"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_quick_info",
            description: "Get quick snapshot of a stock including current price, P/E, P/B, ROE, 52-week range, market cap, and volume.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker symbol (e.g., THYAO, AKBNK, GARAN)",
                    },
                },
                required: ["symbol"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_profile",
            description: "Get company profile including sector, industry, description, website, and key executives.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker symbol (e.g., THYAO, AKBNK, GARAN)",
                    },
                },
                required: ["symbol"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_dividends",
            description: "Get dividend history, current yield, payout ratio, and upcoming dividend dates for a stock.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker symbol (e.g., THYAO, AKBNK, TUPRS)",
                    },
                },
                required: ["symbol"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_technical_analysis",
            description: "Get technical analysis including RSI, MACD, Bollinger Bands, moving averages, and trend signals.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker symbol",
                    },
                    interval: {
                        type: "string",
                        enum: ["1d", "1h", "4h", "1w"],
                        description: "Chart interval. Default: 1d",
                    },
                },
                required: ["symbol"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "scan_stocks",
            description: "Technical scanner for BIST stocks using indicators like RSI, MACD, Supertrend. Find oversold/overbought stocks, trend reversals, breakouts.",
            parameters: {
                type: "object",
                properties: {
                    scan_type: {
                        type: "string",
                        enum: ["oversold", "overbought", "macd_bullish", "macd_bearish", "supertrend_buy", "supertrend_sell", "golden_cross", "death_cross"],
                        description: "Type of technical scan to run",
                    },
                    index: {
                        type: "string",
                        description: "Limit scan to index: XU100, XU030, etc.",
                    },
                },
                required: ["scan_type"],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_sector_comparison",
            description: "Compare sectors in BIST by average P/E, P/B, ROE, and performance metrics.",
            parameters: {
                type: "object",
                properties: {
                    metric: {
                        type: "string",
                        enum: ["pe", "pb", "roe", "performance", "all"],
                        description: "Metric to compare. Default: all",
                    },
                },
                required: [],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_fund_data",
            description: "Get TEFAS fund data including NAV, returns, expense ratio, and portfolio composition. Access to 836+ Turkish investment funds.",
            parameters: {
                type: "object",
                properties: {
                    fund_code: {
                        type: "string",
                        description: "TEFAS fund code (e.g., IPB, AK1, YAF)",
                    },
                    category: {
                        type: "string",
                        description: "Fund category filter: equity, bond, money_market, mixed, etc.",
                    },
                    sort_by: {
                        type: "string",
                        enum: ["return_1m", "return_3m", "return_1y", "nav", "expense_ratio"],
                        description: "Sort funds by metric",
                    },
                },
                required: [],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "get_news",
            description: "Get latest KAP (Public Disclosure Platform) news and announcements for a stock or the market.",
            parameters: {
                type: "object",
                properties: {
                    symbol: {
                        type: "string",
                        description: "Stock ticker to filter news. Leave empty for market-wide news.",
                    },
                    limit: {
                        type: "number",
                        description: "Number of news items to return. Default: 10",
                    },
                },
                required: [],
            },
        },
    },
    {
        type: "function",
        function: {
            name: "search_symbol",
            description: "Search for stocks, indices, funds, or crypto by name or ticker.",
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "Search query (company name, ticker, or partial match)",
                    },
                    type: {
                        type: "string",
                        enum: ["stock", "index", "fund", "crypto", "all"],
                        description: "Filter by security type. Default: all",
                    },
                },
                required: ["query"],
            },
        },
    },
];

/**
 * Execute a tool call against Borsa MCP
 */
export async function executeMCPTool(
    toolCall: MCPToolCall,
    requestId?: string
): Promise<MCPToolResult> {
    const startedAt = Date.now();

    logInfo("mcp.tool.request", {
        requestId: requestId || null,
        tool: toolCall.tool,
        params: toolCall.params,
    });

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), MCP_TIMEOUT_MS);

        const response = await fetch(MCP_ENDPOINT, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                jsonrpc: "2.0",
                id: requestId || crypto.randomUUID(),
                method: "tools/call",
                params: {
                    name: toolCall.tool,
                    arguments: toolCall.params,
                },
            }),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        const latencyMs = Date.now() - startedAt;

        if (!response.ok) {
            const errorText = await response.text();
            logError("mcp.tool.error", {
                requestId: requestId || null,
                tool: toolCall.tool,
                status: response.status,
                latencyMs,
                error: errorText.slice(0, 500),
            });

            return {
                success: false,
                error: `MCP request failed (${response.status}): ${errorText.slice(0, 200)}`,
                latencyMs,
            };
        }

        const result = await response.json();

        // Handle JSON-RPC error response
        if (result.error) {
            logError("mcp.tool.rpc_error", {
                requestId: requestId || null,
                tool: toolCall.tool,
                latencyMs,
                error: result.error,
            });

            return {
                success: false,
                error: result.error.message || JSON.stringify(result.error),
                latencyMs,
            };
        }

        logInfo("mcp.tool.response", {
            requestId: requestId || null,
            tool: toolCall.tool,
            latencyMs,
            hasData: !!result.result,
        });

        return {
            success: true,
            data: result.result,
            latencyMs,
        };
    } catch (error) {
        const latencyMs = Date.now() - startedAt;
        const message = error instanceof Error ? error.message : String(error);

        logError("mcp.tool.exception", {
            requestId: requestId || null,
            tool: toolCall.tool,
            latencyMs,
            error: message,
        });

        return {
            success: false,
            error: message,
            latencyMs,
        };
    }
}

/**
 * Execute multiple tool calls in parallel
 */
export async function executeMCPToolsBatch(
    toolCalls: MCPToolCall[],
    requestId?: string
): Promise<MCPToolResult[]> {
    return Promise.all(
        toolCalls.map((tc) => executeMCPTool(tc, requestId))
    );
}

/**
 * High-level helpers for common operations
 */
export const BorsaMCP = {
    /**
     * Screen stocks with preset or custom filters
     */
    async screenStocks(params: {
        preset?: string;
        index?: string;
        pe_max?: number;
        pb_max?: number;
        div_yield_min?: number;
        rsi_max?: number;
        rsi_min?: number;
    }): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.SCREEN_SECURITIES,
            params,
        });
    },

    /**
     * Get financial statements for a stock
     */
    async getFinancials(symbol: string, statementType: string = "all"): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.GET_FINANCIAL_STATEMENTS,
            params: { symbol, statement_type: statementType },
        });
    },

    /**
     * Get financial ratios for a stock
     */
    async getRatios(symbol: string): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.GET_FINANCIAL_RATIOS,
            params: { symbol },
        });
    },

    /**
     * Get quick info snapshot
     */
    async getQuickInfo(symbol: string): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.GET_QUICK_INFO,
            params: { symbol },
        });
    },

    /**
     * Technical scan for stocks
     */
    async scanStocks(scanType: string, index?: string): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.SCAN_STOCKS,
            params: { scan_type: scanType, index },
        });
    },

    /**
     * Get sector comparison
     */
    async compareSectors(metric: string = "all"): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.GET_SECTOR_COMPARISON,
            params: { metric },
        });
    },

    /**
     * Get fund data
     */
    async getFundData(fundCode?: string, category?: string): Promise<MCPToolResult> {
        return executeMCPTool({
            tool: BORSA_MCP_TOOLS.GET_FUND_DATA,
            params: { fund_code: fundCode, category },
        });
    },
};
