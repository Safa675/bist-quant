import {
  toAnalyticsUiResult,
  toBacktestUiResult,
  toComplianceUiResult,
  toComplianceUiRules,
  toScreenerUiResult,
} from "@/lib/adapters";
import { ApiClientError, parseApiClientError } from "@/lib/api-error";
import type {
  ActivityAnomalyResult,
  AnalyticsUiResult,
  BacktestRawResult,
  BacktestRequest,
  BacktestResult,
  ComplianceRawResult,
  ComplianceResult,
  ComplianceRule,
  ComplianceTransaction,
  CryptoSizingInput,
  CryptoTradePlan,
  DashboardOverview,
  FactorCatalog,
  GreeksInput,
  GreeksResult,
  JobPayload,
  PipValueResult,
  ScreenerFilters,
  ScreenerRawResult,
  ScreenerResult,
  StressResult,
  StressShock,
} from "@/lib/types";

// ─── Base ─────────────────────────────────────────────────────────────────────

function getApiBase(): string {
  const url = process.env.NEXT_PUBLIC_API_URL?.trim();
  return url ? url.replace(/\/$/, "") : "http://127.0.0.1:8001";
}

const API_BASE = getApiBase();

async function fetchJson<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });

  if (!res.ok) {
    let fallbackDetail = res.statusText || "Request failed";
    let parsedError: ApiClientError | null = null;
    try {
      const body = (await res.json()) as unknown;
      parsedError = parseApiClientError(res.status, body);
    } catch {
      // Ignore parse errors and fall back to response text when available.
      const text = await res.text().catch(() => "");
      if (text) fallbackDetail = text;
    }

    if (parsedError) throw parsedError;
    throw new ApiClientError({ status: res.status, detail: fallbackDetail });
  }

  if (res.status === 204) {
    return {} as T;
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return fetchJson<T>(path, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export async function getDashboardOverview(lookback = 504): Promise<DashboardOverview> {
  return fetchJson<DashboardOverview>(`/api/dashboard/overview?lookback=${lookback}`);
}

export async function getDashboardRegimeHistory(lookback = 504): Promise<unknown> {
  return fetchJson(`/api/dashboard/regime-history?lookback=${lookback}`);
}

export async function getDashboardMacro(lookback = 252): Promise<unknown> {
  return fetchJson(`/api/dashboard/macro?lookback=${lookback}`);
}

// ─── Factors / Signals ────────────────────────────────────────────────────────

export async function getFactorCatalog(): Promise<FactorCatalog> {
  return fetchJson<FactorCatalog>("/api/meta/signals");
}

export async function getFactorDetail(name: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`/api/factors/${encodeURIComponent(name)}`);
}

export async function getMacroCalendar(params?: {
  period?: string;
  country?: string;
  importance?: string;
}): Promise<unknown> {
  const q = new URLSearchParams();
  if (params?.period) q.set("period", params.period);
  if (params?.country) q.set("country", params.country);
  if (params?.importance) q.set("importance", params.importance);
  return fetchJson(`/api/macro/calendar?${q.toString()}`);
}

// ─── Backtest ─────────────────────────────────────────────────────────────────

export async function runBacktest(req: BacktestRequest): Promise<BacktestResult> {
  const raw = await postJson<BacktestRawResult>("/api/backtest/run", req);
  return toBacktestUiResult(raw);
}

// ─── Jobs ─────────────────────────────────────────────────────────────────────

export async function createJob(kind: string, request: Record<string, unknown>): Promise<JobPayload> {
  return postJson<JobPayload>("/api/jobs", { kind, request });
}

export async function getJob(id: string): Promise<JobPayload> {
  const job = await fetchJson<JobPayload>(`/api/jobs/${id}`);
  if (job.status !== "completed" || !job.result) return job;

  if (job.kind === "backtest") {
    return {
      ...job,
      result: toBacktestUiResult(job.result) as unknown as Record<string, unknown>,
    };
  }

  return job;
}

export async function listJobs(limit = 20): Promise<{ count: number; jobs: JobPayload[] }> {
  return fetchJson(`/api/jobs?limit=${limit}`);
}

export async function cancelJob(id: string): Promise<{ id: string; cancelled: boolean }> {
  return fetchJson(`/api/jobs/${id}`, { method: "DELETE" });
}

export async function retryJob(id: string): Promise<JobPayload> {
  return postJson<JobPayload>(`/api/jobs/${id}/retry`, {});
}

/** Submit a job and poll until terminal state (completed/failed/cancelled) or timeout. */
export async function submitAndPollJob(
  kind: string,
  request: Record<string, unknown>,
  onStatus?: (job: JobPayload) => void,
  maxWaitSecs = 300,
): Promise<JobPayload> {
  const job = await createJob(kind, request);
  onStatus?.(job);
  const pollInterval = 2000;
  const maxPolls = (maxWaitSecs * 1000) / pollInterval;
  for (let i = 0; i < maxPolls; i++) {
    await new Promise((r) => setTimeout(r, pollInterval));
    const updated = await getJob(job.id);
    onStatus?.(updated);
    if (
      updated.status === "completed" ||
      updated.status === "failed" ||
      updated.status === "cancelled"
    ) {
      return updated;
    }
  }
  throw new Error("Job timed out");
}

// ─── Screener ─────────────────────────────────────────────────────────────────

function buildScreenerRequest(filters: ScreenerFilters): Record<string, unknown> {
  // If caller already provides canonical nested filters payload, preserve it.
  const canonicalFilters = filters.filters;
  const request: Record<string, unknown> = {
    limit: filters.limit ?? filters.top_n ?? 50,
    sort_by: filters.sort_by ?? "return_1m",
    sort_desc: filters.sort_desc ?? !(filters.sort_asc ?? false),
  };

  if (filters.index) request.index = filters.index;
  if (filters.sector) request.sector = filters.sector;
  if (filters.template) request.template = filters.template;
  if (filters.recommendation) request.recommendation = filters.recommendation;
  if (Array.isArray(filters.fields) && filters.fields.length > 0) {
    request.fields = filters.fields;
  }

  if (filters.buy_recommendation) {
    request.recommendation = "AL";
  } else if (filters.sell_recommendation) {
    request.recommendation = "SAT";
  } else if (filters.high_return) {
    request.template = "high_return";
  } else if (filters.high_foreign_ownership) {
    request.template = "high_foreign_ownership";
  }

  const nestedFilters: Record<string, { min?: number; max?: number }> = {};
  const addRange = (key: string, min?: number | null, max?: number | null) => {
    const row: { min?: number; max?: number } = {};
    if (typeof min === "number" && Number.isFinite(min)) row.min = min;
    if (typeof max === "number" && Number.isFinite(max)) row.max = max;
    if (row.min !== undefined || row.max !== undefined) nestedFilters[key] = row;
  };

  addRange("pe", filters.min_pe, filters.max_pe);
  addRange("pb", filters.min_pb, filters.max_pb);
  addRange("market_cap_usd", filters.min_market_cap_usd, filters.max_market_cap_usd);
  addRange("rsi_14", filters.min_rsi_14, filters.max_rsi_14);
  addRange("return_1m", filters.min_return_1m, filters.max_return_1m);

  if (canonicalFilters && Object.keys(canonicalFilters).length > 0) {
    request.filters = canonicalFilters;
  } else if (Object.keys(nestedFilters).length > 0) {
    request.filters = nestedFilters;
  }

  return request;
}

export async function runScreener(filters: ScreenerFilters): Promise<ScreenerResult> {
  const raw = await postJson<ScreenerRawResult>("/api/screener/run", buildScreenerRequest(filters));
  return toScreenerUiResult(raw);
}

export async function getSparklines(symbols: string[]): Promise<Record<string, number[]>> {
  return postJson<Record<string, number[]>>("/api/screener/sparklines", { symbols });
}

export async function getScreenerMetadata(): Promise<Record<string, unknown>> {
  return fetchJson("/api/screener/metadata");
}

// ─── Factor Lab ───────────────────────────────────────────────────────────────

function normalizeFactorSignals(rawSignals: unknown): Array<{ name: string; weight: number }> {
  if (!Array.isArray(rawSignals)) return [];

  const asObjects = rawSignals
    .map((row) => {
      if (typeof row === "string") {
        return { name: row, weight: 0 };
      }
      if (row && typeof row === "object") {
        const item = row as Record<string, unknown>;
        const name = typeof item.name === "string" ? item.name : "";
        const weight = typeof item.weight === "number" ? item.weight : 0;
        if (!name) return null;
        return { name, weight };
      }
      return null;
    })
    .filter((row): row is NonNullable<typeof row> => row !== null);

  if (!asObjects.length) return [];

  const explicitWeightSum = asObjects.reduce((sum, row) => sum + row.weight, 0);
  if (explicitWeightSum > 0) {
    return asObjects.map((row) => ({ name: row.name, weight: row.weight / explicitWeightSum }));
  }

  const equal = 1 / asObjects.length;
  return asObjects.map((row) => ({ name: row.name, weight: equal }));
}

export async function combineFactors(payload: Record<string, unknown>): Promise<JobPayload> {
  const normalizedPayload: Record<string, unknown> = { ...payload };
  normalizedPayload.signals = normalizeFactorSignals(payload.signals);
  return createJob("factor_combine", normalizedPayload);
}

// ─── Analytics ────────────────────────────────────────────────────────────────

export async function runAnalytics(payload: Record<string, unknown>): Promise<AnalyticsUiResult> {
  const raw = await postJson<unknown>("/api/analytics/run", payload);
  return toAnalyticsUiResult(raw);
}

export async function getAnalyticsBenchmarkXU100(): Promise<{
  symbol: string;
  curve: Array<{ date: string; value: number }>;
}> {
  return fetchJson("/api/analytics/benchmark/xu100");
}

// ─── Optimization ─────────────────────────────────────────────────────────────

export async function runOptimization(payload: Record<string, unknown>): Promise<JobPayload> {
  return createJob("optimize", payload);
}

// ─── Signal Construction ─────────────────────────────────────────────────────

export async function runSignalConstructionSnapshot(
  payload: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return postJson("/api/signal-construction/snapshot", payload);
}

export async function runSignalConstructionBacktest(
  payload: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return postJson("/api/signal-construction/backtest", payload);
}

export async function runSignalConstructionFiveFactor(
  payload: Record<string, unknown>,
): Promise<BacktestResult> {
  const raw = await postJson<BacktestRawResult>("/api/signal-construction/five-factor", payload);
  return toBacktestUiResult(raw);
}

export async function runSignalConstructionOrthogonalization(
  payload: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  return postJson("/api/signal-construction/orthogonalization", payload);
}

// ─── Professional ─────────────────────────────────────────────────────────────

export async function calculateGreeks(input: GreeksInput): Promise<GreeksResult> {
  return postJson<GreeksResult>("/api/professional/greeks", input);
}

export async function runStressTest(input: {
  shocks: StressShock[];
  portfolio_value?: number;
}): Promise<StressResult> {
  return postJson<StressResult>("/api/professional/stress", input);
}

export async function calculateCryptoSizing(input: CryptoSizingInput): Promise<CryptoTradePlan> {
  return postJson<CryptoTradePlan>("/api/professional/crypto-sizing", input);
}

export async function calculatePipValue(input: {
  pair: string;
  lot_size: number;
  account_conversion_rate?: number;
}): Promise<PipValueResult> {
  return postJson<PipValueResult>("/api/professional/pip-value", input);
}

// ─── Compliance ───────────────────────────────────────────────────────────────

export async function runComplianceCheck(payload: {
  transaction: ComplianceTransaction;
  rules: ComplianceRule[];
}): Promise<ComplianceResult> {
  const rules = payload.rules.map((rule) => ({
    id: rule.id,
    field: rule.field,
    comparator: rule.operator,
    threshold: rule.threshold,
    message: rule.description,
    severity: rule.severity,
  }));
  const raw = await postJson<ComplianceRawResult>("/api/compliance/check", {
    transaction: payload.transaction,
    rules,
  });
  return toComplianceUiResult(raw, payload.rules);
}

export async function getDefaultComplianceRules(): Promise<{ rules: ComplianceRule[] }> {
  const raw = await fetchJson<unknown>("/api/compliance/rules");
  return toComplianceUiRules(raw);
}

export async function checkPositionLimits(
  positions: { symbol: string; value: number; limit: number }[],
): Promise<{ breaches: unknown[] }> {
  return postJson("/api/compliance/position-limits", { positions });
}

export async function checkActivityAnomalies(
  events: { user_id: string }[],
): Promise<ActivityAnomalyResult> {
  return postJson<ActivityAnomalyResult>("/api/compliance/activity-anomalies", { events });
}

// ─── System ───────────────────────────────────────────────────────────────────

export async function getSystemMeta(): Promise<unknown> {
  return fetchJson("/api/meta/system");
}

export async function getHealthStatus(): Promise<{ ok: boolean }> {
  return fetchJson("/api/health/live");
}
