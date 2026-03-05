import { describe, it, expect, afterEach, afterAll, beforeAll } from "vitest";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { ApiClientError } from "@/lib/api-error";
import {
  getDashboardOverview,
  getFactorCatalog,
  runBacktest,
  runAnalytics,
  getAnalyticsBenchmarkXU100,
  createJob,
  getJob,
  listJobs,
  cancelJob,
  runScreener,
  runSignalConstructionSnapshot,
  runSignalConstructionBacktest,
  runSignalConstructionOrthogonalization,
  runSignalConstructionFiveFactor,
  calculatePipValue,
  checkActivityAnomalies,
  getHealthStatus,
  getDefaultComplianceRules,
} from "@/lib/api";

// ─── MSW Server ───────────────────────────────────────────────────────────────

const API = "http://127.0.0.1:8001";

const handlers = [
  http.get(`${API}/api/dashboard/overview`, ({ request }) => {
    const url = new URL(request.url);
    return HttpResponse.json({
      total_return: 0.25,
      sharpe: 1.5,
      max_drawdown: -0.12,
      lookback: Number(url.searchParams.get("lookback") || 504),
    });
  }),

  http.get(`${API}/api/meta/signals`, () => {
    return HttpResponse.json({
      count: 2,
      signals: ["momentum", "value"],
    });
  }),

  http.post(`${API}/api/backtest/run`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      metrics: {
        cagr: body.factor_name ? 0.35 : 0.2,
        sharpe: 1.8,
        max_drawdown: -0.08,
      },
      equity_curve: [
        { date: "2024-01-02", value: 1.0, benchmark: 1.0, drawdown: 0.0 },
        { date: "2024-01-03", value: 1.01, benchmark: 1.005, drawdown: -0.01 },
      ],
      drawdown_curve: [
        { date: "2024-01-02", value: 0.0 },
        { date: "2024-01-03", value: -0.01 },
      ],
      monthly_returns: [{ month: "2024-01", strategy_return: 1.2 }],
      top_holdings: [{ ticker: "THYAO", weight: 0.2 }],
    });
  }),

  http.post(`${API}/api/jobs`, async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      id: "job-123",
      kind: body.kind,
      status: "queued",
      created_at: new Date().toISOString(),
    });
  }),

  http.get(`${API}/api/jobs/job-123`, () => {
    return HttpResponse.json({
      id: "job-123",
      kind: "backtest",
      status: "completed",
      result: { sharpe: 1.5 },
    });
  }),

  http.get(`${API}/api/jobs`, () => {
    return HttpResponse.json({
      count: 1,
      jobs: [{ id: "job-123", kind: "backtest", status: "completed" }],
    });
  }),

  http.delete(`${API}/api/jobs/job-123`, () => {
    return HttpResponse.json({ id: "job-123", cancelled: true });
  }),

  http.post(`${API}/api/screener/run`, () => {
    return HttpResponse.json({
      meta: { total_matches: 1 },
      rows: [{ symbol: "THYAO", return_1m: 12 }],
    });
  }),

  http.post(`${API}/api/analytics/run`, () => {
    return HttpResponse.json({
      methods: ["performance", "rolling", "walk_forward", "attribution"],
      performance: { cagr: 12.3, sharpe: 1.4 },
      rolling: [{ date: "2024-01-03", rolling_sharpe_63d: 1.1 }],
      walk_forward: [{ split: 1, train_cagr: 10.1, test_cagr: 8.4 }],
      attribution: { momentum: 0.3, value: 0.2 },
    });
  }),

  http.get(`${API}/api/analytics/benchmark/xu100`, () => {
    return HttpResponse.json({
      symbol: "XU100",
      curve: [
        { date: "2024-01-02", value: 100 },
        { date: "2024-01-03", value: 101 },
      ],
    });
  }),

  http.post(`${API}/api/signal-construction/snapshot`, () => {
    return HttpResponse.json({
      meta: { universe: "XU100", symbols_used: 3 },
      signals: [{ symbol: "THYAO", action: "BUY", combined_score: 0.8 }],
      indicator_summaries: [{ name: "rsi", buy_count: 1, hold_count: 2, sell_count: 0 }],
    });
  }),

  http.post(`${API}/api/signal-construction/backtest`, () => {
    return HttpResponse.json({
      metrics: { cagr: 0.12, sharpe: 1.3 },
      equity_curve: [
        { date: "2024-01-02", value: 1.0, benchmark: 1.0, drawdown: 0.0 },
        { date: "2024-01-03", value: 1.02, benchmark: 1.01, drawdown: -0.01 },
      ],
    });
  }),

  http.post(`${API}/api/signal-construction/five-factor`, () => {
    return HttpResponse.json({
      metrics: { cagr: 0.15, sharpe: 1.5 },
      equity_curve: [
        { date: "2024-01-02", value: 1.0, benchmark: 1.0, drawdown: 0.0 },
        { date: "2024-01-03", value: 1.03, benchmark: 1.01, drawdown: -0.01 },
      ],
    });
  }),

  http.post(`${API}/api/signal-construction/orthogonalization`, () => {
    return HttpResponse.json({ enabled: true, status: "configured", axes: ["momentum", "value"] });
  }),

  http.post(`${API}/api/professional/pip-value`, () => {
    return HttpResponse.json({
      pair: "EURUSD",
      pip_size: 0.0001,
      pip_value_quote: 10,
      pip_value_account: 10,
    });
  }),

  http.post(`${API}/api/compliance/activity-anomalies`, () => {
    return HttpResponse.json({
      events_count: 10,
      anomaly_count: 1,
      anomalies: [{ user_id: "USR-001", actions_per_hour: 19.2, z_score: 2.4 }],
    });
  }),

  http.get(`${API}/api/health/live`, () => {
    return HttpResponse.json({ ok: true });
  }),

  http.get(`${API}/api/compliance/rules`, () => {
    return HttpResponse.json({
      rules: [
        { id: "max-position", name: "Max Position", threshold: 0.1 },
      ],
    });
  }),
];

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("API client", () => {
  describe("getDashboardOverview", () => {
    it("fetches dashboard overview with default lookback", async () => {
      const data = await getDashboardOverview();
      expect(data).toHaveProperty("total_return", 0.25);
      expect(data).toHaveProperty("sharpe", 1.5);
    });

    it("passes custom lookback param", async () => {
      const data = await getDashboardOverview(252);
      expect(data).toHaveProperty("lookback", 252);
    });
  });

  describe("getFactorCatalog", () => {
    it("returns signal list", async () => {
      const data = await getFactorCatalog();
      expect(data.signals).toHaveLength(2);
      expect(data.signals[0]).toBe("momentum");
    });
  });

  describe("runBacktest", () => {
    it("submits backtest and returns result", async () => {
      const result = await runBacktest({
        factor_name: "momentum",
        start_date: "2020-01-01",
        end_date: "2020-12-31",
      } as never);
      expect(result.metrics).toHaveProperty("cagr", 0.35);
      expect(result.metrics).toHaveProperty("sharpe", 1.8);
      expect(result.equity_curve).toHaveLength(2);
      expect(result.monthly_returns["2024"].Jan).toBeCloseTo(0.012);
    });
  });

  describe("Jobs API", () => {
    it("creates a job", async () => {
      const job = await createJob("backtest", { strategy: "momentum" });
      expect(job.id).toBe("job-123");
      expect(job.status).toBe("queued");
    });

    it("gets a job by id", async () => {
      const job = await getJob("job-123");
      expect(job.status).toBe("completed");
    });

    it("lists jobs", async () => {
      const data = await listJobs();
      expect(data.count).toBe(1);
      expect(data.jobs).toHaveLength(1);
    });

    it("cancels a job", async () => {
      const result = await cancelJob("job-123");
      expect(result.cancelled).toBe(true);
    });
  });

  describe("runScreener", () => {
    it("fetches screener results", async () => {
      const data = await runScreener({} as never);
      expect(data.rows).toHaveLength(1);
      expect(data.rows[0]).toHaveProperty("symbol", "THYAO");
      expect(data.rows[0].return_1m).toBeCloseTo(0.12);
    });
  });

  describe("analytics", () => {
    it("runs analytics and normalizes key sections", async () => {
      const data = await runAnalytics({
        equity_curve: [
          { date: "2024-01-02", value: 100 },
          { date: "2024-01-03", value: 101 },
        ],
      });
      expect(data.methods).toContain("performance");
      expect(data.metrics.cagr).toBe(12.3);
      expect(data.walk_forward?.length).toBe(1);
    });

    it("loads XU100 benchmark curve", async () => {
      const data = await getAnalyticsBenchmarkXU100();
      expect(data.symbol).toBe("XU100");
      expect(data.curve).toHaveLength(2);
    });
  });

  describe("signal construction", () => {
    it("runs snapshot/backtest/five-factor/orthogonalization endpoints", async () => {
      const snap = await runSignalConstructionSnapshot({ universe: "XU100" });
      expect((snap.meta as { universe: string }).universe).toBe("XU100");

      const backtest = await runSignalConstructionBacktest({ universe: "XU100" });
      expect(backtest.metrics).toBeTruthy();

      const fiveFactor = await runSignalConstructionFiveFactor({ factor_name: "five_factor_rotation" });
      expect(fiveFactor.metrics.sharpe).toBeCloseTo(1.5);

      const orth = await runSignalConstructionOrthogonalization({ enabled: true, axes: ["momentum", "value"] });
      expect(orth.status).toBe("configured");
    });
  });

  describe("professional/compliance extras", () => {
    it("returns pip value result", async () => {
      const data = await calculatePipValue({ pair: "EURUSD", lot_size: 100_000 });
      expect(data.pip_size).toBeCloseTo(0.0001);
      expect(data.pip_value_quote).toBe(10);
    });

    it("returns compliance activity anomalies", async () => {
      const data = await checkActivityAnomalies([{ user_id: "USR-001" }]);
      expect(data.anomaly_count).toBe(1);
      expect(data.anomalies[0]?.user_id).toBe("USR-001");
    });
  });

  describe("getHealthStatus", () => {
    it("returns ok: true", async () => {
      const data = await getHealthStatus();
      expect(data.ok).toBe(true);
    });
  });

  describe("getDefaultComplianceRules", () => {
    it("returns compliance rules", async () => {
      const data = await getDefaultComplianceRules();
      expect(data.rules).toHaveLength(1);
      expect(data.rules[0]).toHaveProperty("id", "max-position");
    });
  });

  describe("error handling", () => {
    it("throws on non-ok response", async () => {
      server.use(
        http.get(`${API}/api/health/live`, () => {
          return new HttpResponse("service unavailable", { status: 503 });
        }),
      );
      await expect(getHealthStatus()).rejects.toThrow("API 503");
    });

    it("parses structured errors into ApiClientError", async () => {
      server.use(
        http.post(`${API}/api/jobs`, () => {
          return HttpResponse.json(
            {
              detail: {
                code: "job_validation_error",
                detail: "Invalid request payload for job kind 'backtest'.",
                hint: "Provide a non-empty factor_name.",
              },
            },
            { status: 422 },
          );
        }),
      );

      try {
        await createJob("backtest", {});
      } catch (error) {
        expect(error).toBeInstanceOf(ApiClientError);
        const apiError = error as ApiClientError;
        expect(apiError.status).toBe(422);
        expect(apiError.code).toBe("job_validation_error");
        expect(apiError.detail).toBe("Invalid request payload for job kind 'backtest'.");
        expect(apiError.hint).toBe("Provide a non-empty factor_name.");
        return;
      }

      throw new Error("Expected createJob to reject with ApiClientError");
    });
  });
});
