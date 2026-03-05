import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.PLAYWRIGHT_PORT ?? "3100");
const BACKEND_PORT = Number(process.env.PLAYWRIGHT_BACKEND_PORT ?? "8001");
const EXTERNAL_BASE_URL = process.env.PLAYWRIGHT_EXTERNAL_BASE_URL?.trim() || "";
const BASE_URL = EXTERNAL_BASE_URL || `http://127.0.0.1:${PORT}`;
const API_URL = `http://127.0.0.1:${BACKEND_PORT}`;
const IS_EXTERNAL = Boolean(EXTERNAL_BASE_URL);

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",

  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: IS_EXTERNAL
    ? undefined
    : [
        {
          command: `bash -lc "cd .. && python -m uvicorn bist_quant.api.main:app --host 127.0.0.1 --port ${BACKEND_PORT}"`,
          url: `${API_URL}/api/health/live`,
          reuseExistingServer: !process.env.CI,
          timeout: 90_000,
        },
        {
          command: `NEXT_PUBLIC_API_URL=${API_URL} npm run dev -- --port ${PORT}`,
          url: BASE_URL,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
      ],
});
