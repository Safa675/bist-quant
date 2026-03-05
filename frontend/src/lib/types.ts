// ─── UI ──────────────────────────────────────────────────────────────────────

export type UiDensity = "comfortable" | "compact";

// ─── Dashboard ───────────────────────────────────────────────────────────────

export interface DashboardKpi {
    xu100_last: number | null;
    xu100_daily_pct: number | null;
    usdtry_last: number | null;
    usdtry_daily_pct: number | null;
    xau_try_last: number | null;
    xau_try_daily_pct: number | null;
}

export interface TimelinePoint { date: string; close: number; }
export interface MacroPoint { date: string; value: number; }

export interface RegimePoint { date: string; regime: string; }

export interface RegimeDistributionItem {
    regime: string;
    count: number;
    percent: number;
}

export interface RegimePayload {
    label: string;
    current: {
        date?: string | null;
        above_ma?: boolean;
        allocation?: number | null;
        realized_vol?: number | null;
        vol_percentile?: number | null;
    };
    series: RegimePoint[];
    distribution: RegimeDistributionItem[];
}

export interface MacroChange {
    asset: string;
    current: number;
    d1_pct: number | null;
    w1_pct: number | null;
    m1_pct: number | null;
}

export interface MacroPayload {
    series: { usdtry: MacroPoint[]; xau_try: MacroPoint[] };
    changes: MacroChange[];
}

export interface DashboardOverview {
    kpi: DashboardKpi;
    regime: RegimePayload;
    timeline: TimelinePoint[];
    macro: MacroPayload;
    lookback: number;
    date_range?: { start: string; end: string };
    defaults?: { start_date: string; end_date: string };
    error?: string;
}

// ─── Backtest ────────────────────────────────────────────────────────────────

export interface BacktestRequest {
    factor_name: string;
    start_date: string;
    end_date: string;
    rebalance_frequency?: string;
    top_n?: number;
    max_position_weight?: number;
}

export interface EquityCurvePoint {
    date: string;
    strategy: number;
    benchmark?: number | null;
    drawdown?: number | null;
}

/** Contiguous regime band for equity curve shading */
export interface RegimeBand {
    start: string;
    end: string;
    regime: string;
}

export interface BacktestMetrics {
    cagr?: number | null;
    sharpe?: number | null;
    sortino?: number | null;
    max_drawdown?: number | null;
    annualized_volatility?: number | null;
    alpha?: number | null;
    beta?: number | null;
    calmar?: number | null;
    win_rate?: number | null;
    total_return?: number | null;
    [key: string]: unknown;
}

export interface BacktestRiskMetrics {
    tail_risk?: {
        var_95?: number | null;
        cvar_95?: number | null;
        var_99?: number | null;
        cvar_99?: number | null;
    };
    mae_mfe?: {
        mae_1d?: number | null;
        mfe_1d?: number | null;
        worst_5d?: number | null;
        best_5d?: number | null;
    };
    [key: string]: unknown;
}

export interface BacktestScenarioAnalysis {
    best_day?: number | null;
    worst_day?: number | null;
    stress_1d_minus_2sigma?: number | null;
    stress_1d_minus_3sigma?: number | null;
    [key: string]: unknown;
}

export interface Holding {
    symbol: string;
    weight: number;
    [key: string]: unknown;
}

export interface BacktestRawResult {
    metrics: BacktestMetrics;
    equity_curve?: Array<{
        date: string;
        strategy?: number | null;
        value?: number | null;
        benchmark?: number | null;
        drawdown?: number | null;
    }>;
    drawdown_curve?: Array<{ date: string; drawdown?: number | null; value?: number | null }>;
    monthly_returns?:
        | Record<string, Record<string, number>>
        | Array<{
              month: string;
              strategy_return?: number | null;
              benchmark_return?: number | null;
              excess_return?: number | null;
          }>;
    holdings?: Holding[];
    top_holdings?: Array<{ ticker: string; weight: number }>;
    rolling_metrics?: Array<{
        date: string;
        rolling_sharpe_63d?: number | null;
        rolling_volatility_63d?: number | null;
        rolling_max_drawdown_126d?: number | null;
    }>;
    risk_metrics?: BacktestRiskMetrics;
    scenario_analysis?: BacktestScenarioAnalysis;
    sector_exposure?: Record<string, number>;
    summary?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface BacktestUiResult {
    metrics: BacktestMetrics;
    equity_curve: EquityCurvePoint[];
    drawdown_curve: { date: string; drawdown: number }[];
    monthly_returns: Record<string, Record<string, number>>;
    holdings: Holding[];
    rolling_metrics: Array<{
        date: string;
        rolling_sharpe_63d: number | null;
        rolling_volatility_63d: number | null;
        rolling_max_drawdown_126d: number | null;
    }>;
    risk_metrics?: BacktestRiskMetrics;
    scenario_analysis?: BacktestScenarioAnalysis;
    sector_exposure?: Record<string, number>;
    summary?: Record<string, unknown>;
    raw?: Record<string, unknown>;
}

export type BacktestResult = BacktestUiResult;

// ─── Jobs ─────────────────────────────────────────────────────────────────────

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface JobPayload {
    id: string;
    kind: string;
    status: JobStatus;
    created_at: string;
    updated_at: string;
    error?: string | null;
    meta?: Record<string, unknown>;
    request?: Record<string, unknown>;
    result?: Record<string, unknown>;
}

// ─── Analytics ───────────────────────────────────────────────────────────────

export interface AnalyticsRawResult {
    methods?: string[];
    performance?: Record<string, unknown>;
    risk?: Record<string, unknown>;
    rolling?: Array<Record<string, unknown>>;
    stress?: Array<Record<string, unknown>>;
    transaction_costs?: Record<string, unknown>;
    monte_carlo?: Record<string, unknown>;
    walk_forward?: Array<Record<string, unknown>>;
    attribution?: Record<string, unknown> | null;
    benchmark?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface AnalyticsUiResult {
    methods: string[];
    metrics: Record<string, number | null>;
    rolling: Array<{ date: string; rolling_sharpe: number | null }>;
    performance?: Record<string, unknown>;
    risk?: Record<string, unknown>;
    stress?: Array<Record<string, unknown>>;
    transaction_costs?: Record<string, unknown>;
    monte_carlo?: Record<string, unknown>;
    walk_forward?: Array<Record<string, unknown>>;
    attribution?: Record<string, unknown> | null;
    benchmark?: Record<string, unknown>;
}

// ─── Optimization ────────────────────────────────────────────────────────────

export interface OptimizationTrial {
    trial_id: number;
    params: Record<string, number>;
    metrics: Record<string, unknown>;
    score?: number | null;
    feasible?: boolean;
}

export interface OptimizationRawResult {
    status?: string;
    method?: string;
    best_trial?: OptimizationTrial & { backtest?: BacktestRawResult };
    trials?: OptimizationTrial[];
    pareto_front?: OptimizationTrial[];
    walk_forward?: Array<Record<string, unknown>>;
    scenario_analysis?: Record<string, unknown>;
    constraints?: Record<string, unknown>;
    objective?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface OptimizationUiResult {
    best_params: Record<string, number>;
    best_metric: number | null;
    sweep_results: Array<{
        params: Record<string, number>;
        metric: number;
    }>;
    best_trial?: OptimizationTrial;
    trials: OptimizationTrial[];
    pareto_front?: OptimizationTrial[];
    walk_forward?: Array<Record<string, unknown>>;
    scenario_analysis?: Record<string, unknown>;
    constraints?: Record<string, unknown>;
    objective?: Record<string, unknown>;
}

// ─── Factors ─────────────────────────────────────────────────────────────────

export interface SignalInfo {
    name: string;
    category?: string;
    description?: string;
    parameters?: Record<string, unknown>;
}

export interface FactorCatalog {
    count: number;
    signals: string[];
    details?: SignalInfo[];
}

// ─── Screener ─────────────────────────────────────────────────────────────────

export interface ScreenerFilters {
    index?: string;
    sector?: string;
    sectors?: string[];
    template?: string;
    recommendation?: string;
    recommendations?: string[];
    filters?: Record<string, { min?: number; max?: number }>;
    fields?: string[];
    limit?: number;
    sort_desc?: boolean;
    min_pe?: number | null;
    max_pe?: number | null;
    min_pb?: number | null;
    max_pb?: number | null;
    min_market_cap_usd?: number | null;
    max_market_cap_usd?: number | null;
    min_rsi_14?: number | null;
    max_rsi_14?: number | null;
    min_return_1m?: number | null;
    max_return_1m?: number | null;
    buy_recommendation?: boolean;
    sell_recommendation?: boolean;
    high_return?: boolean;
    high_foreign_ownership?: boolean;
    sort_by?: string;
    sort_asc?: boolean;
    top_n?: number;
}

export interface ScreenerRow {
    symbol: string;
    sector?: string;
    market_cap_usd?: number | null;
    pe?: number | null;
    pb?: number | null;
    rsi_14?: number | null;
    return_1m?: number | null;
    return_1y?: number | null;
    upside_potential?: number | null;
    recommendation?: string | null;
    [key: string]: unknown;
}

export interface ScreenerResult {
    count: number;
    rows: ScreenerRow[];
}

export interface ScreenerRawResult {
    meta?: {
        total_matches?: number;
        returned_rows?: number;
        as_of?: string;
        data_source?: string;
        execution_ms?: number;
        [key: string]: unknown;
    };
    applied_filters?: Array<Record<string, unknown>>;
    rows?: Record<string, unknown>[];
    [key: string]: unknown;
}

export interface ScreenerUiResult {
    count: number;
    rows: ScreenerRow[];
    meta?: Record<string, unknown>;
    applied_filters?: Array<Record<string, unknown>>;
}

// ─── Professional ─────────────────────────────────────────────────────────────

export interface GreeksInput {
    option_type: "call" | "put";
    spot: number;
    strike: number;
    time_years: number;
    volatility: number;
    risk_free_rate: number;
}

export interface GreeksResult {
    delta: number;
    gamma: number;
    theta_per_day: number;
    vega_per_1pct: number;
    rho_per_1pct: number;
    theoretical_price: number;
}

export interface StressShock {
    factor: string;
    shock_pct: number;
    beta: number;
}

export interface StressResult {
    scenario_loss_pct: number;
    scenario_loss_value: number;
    by_factor: { factor: string; loss_pct: number; contribution_pct: number }[];
}

export interface CryptoSizingInput {
    pair: string;
    side: "long" | "short";
    entry_price: number;
    equity: number;
    risk_pct: number;
    leverage: number;
    stop_distance_pct: number;
}

export interface CryptoTradePlan {
    pair: string;
    side: string;
    notional: number;
    margin_required: number;
    quantity: number;
    liquidation_price: number;
    max_loss: number;
    estimated_fees: number;
}

export interface PipValueResult {
    pair: string;
    pip_size: number;
    pip_value_quote: number;
    pip_value_account: number;
}

// ─── Compliance ───────────────────────────────────────────────────────────────

export interface ComplianceRule {
    id: string;
    description: string;
    field: string;
    operator: string;
    threshold: number;
    severity: "warning" | "critical";
}

export interface ComplianceTransaction {
    id: string;
    timestamp: string;
    user_id: string;
    order_id: string;
    symbol: string;
    side: "buy" | "sell";
    quantity: number;
    price: number;
}

export interface ComplianceHit {
    rule_id: string;
    message: string;
    severity: string;
    field?: string;
    operator?: string;
    observed?: number | null;
    limit?: number | null;
}

export interface ComplianceResult {
    transaction_id: string;
    status: "PASS" | "FAIL";
    hits: ComplianceHit[];
}

export interface ComplianceRawRule {
    id: string;
    description?: string;
    message?: string;
    field?: string;
    comparator?: string;
    operator?: string;
    threshold?: number;
    severity?: "warning" | "critical";
    [key: string]: unknown;
}

export interface ComplianceRawResult {
    transaction_id?: string;
    status?: "PASS" | "FAIL";
    passed?: boolean;
    hits?: Array<Record<string, unknown>>;
    [key: string]: unknown;
}

export interface ComplianceUiResult {
    transaction_id: string;
    status: "PASS" | "FAIL";
    hits: ComplianceHit[];
}

export interface ActivityAnomalyResult {
    events_count: number;
    anomaly_count: number;
    anomalies: Array<{
        user_id: string;
        actions_per_hour: number;
        z_score: number;
    }>;
}

export interface AgentSessionEntry {
    id: string;
    timestamp: string;
    role: "system" | "user" | "agent";
    agent: string;
    content: string;
}
