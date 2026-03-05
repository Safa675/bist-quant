/**
 * Additional TypeScript type definitions beyond lib/types.ts.
 * Complex API response shapes, utility types, etc.
 */

// ─── Analytics ────────────────────────────────────────────────────────────────

export interface AnalyticsRequest {
  equity_curve: Array<{ date: string; strategy: number }>;
  methods?: string[];
}

export interface AnalyticsResult {
  basic_metrics?: Record<string, number | null>;
  advanced_metrics?: Record<string, number | null>;
  rolling_metrics?: Array<{ date: string; rolling_sharpe?: number | null }>;
  [key: string]: unknown;
}

// ─── Factor Lab ───────────────────────────────────────────────────────────────

export interface FactorSnapshotRequest {
  indicators: string[];
  universe?: string;
  date?: string;
}

export interface FactorScoreRow {
  symbol: string;
  score: number;
  rank?: number;
  [key: string]: unknown;
}

export interface FactorSnapshotResult {
  scores: FactorScoreRow[];
  date: string;
}

export interface FactorCombineRequest {
  signals: string[];
  weights?: number[];
  method?: "equal" | "custom";
  start_date?: string;
  end_date?: string;
}

export interface FactorCombineResult {
  combined_series: Array<{ date: string; value: number }>;
  attribution?: Record<string, number>;
}

// ─── Signal Construction ──────────────────────────────────────────────────────

export interface SignalConstructionRequest {
  factor_name: string;
  start_date: string;
  end_date: string;
  top_n?: number;
}

export interface SignalConstructionResult {
  series: Array<{ date: string; signal: number }>;
  stats?: Record<string, number | null>;
}

// ─── Optimization ─────────────────────────────────────────────────────────────

export interface OptimizationRequest {
  signal: string;
  params: Record<string, number[]>;
  method?: "grid" | "random";
  metric?: string;
  start_date?: string;
  end_date?: string;
}

export interface OptimizationResult {
  best_params: Record<string, number>;
  best_metric: number;
  sweep_results?: Array<{ params: Record<string, number>; metric: number }>;
}

// ─── Professional ─────────────────────────────────────────────────────────────

export interface StressShock {
  asset: string;
  shock_pct: number;
}

export interface StressResult {
  portfolio_pnl: number;
  positions: Array<{ symbol: string; weight: number; pnl: number }>;
}

export interface CryptoSizingInput {
  capital: number;
  risk_per_trade_pct: number;
  entry_price: number;
  stop_loss_price: number;
  leverage?: number;
}

export interface CryptoTradePlan {
  position_size_usd: number;
  quantity: number;
  risk_amount_usd: number;
  reward_risk_ratio?: number;
  [key: string]: unknown;
}

// ─── Compliance ───────────────────────────────────────────────────────────────

export interface ComplianceTransaction {
  symbol: string;
  quantity: number;
  price: number;
  direction: "buy" | "sell";
  portfolio_value?: number;
  [key: string]: unknown;
}

export interface ComplianceRule {
  id: string;
  description: string;
  enabled: boolean;
}

export interface ComplianceResult {
  passed: boolean;
  violations: Array<{ rule_id: string; message: string }>;
  warnings: Array<{ rule_id: string; message: string }>;
}

// ─── Utility ──────────────────────────────────────────────────────────────────

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type ApiResponse<T> = { data: T; error?: never } | { data?: never; error: string };
