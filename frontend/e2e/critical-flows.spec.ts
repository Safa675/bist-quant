import { test, expect, type Locator, type Page } from "@playwright/test";

const LONG_TIMEOUT_MS = 180_000;

test.describe.configure({ mode: "serial" });

async function assertNoApiFailure(page: Page) {
  await expect(page.getByText(/API 500:/)).toHaveCount(0);
  await expect(page.getByText(/Request failed/i)).toHaveCount(0);
  await expect(page.getByText(/Failed to fetch/i)).toHaveCount(0);
  await expect(page.getByText(/Job failed/i)).toHaveCount(0);
}

async function waitForSuccessOrFailure(page: Page, success: Locator) {
  const failure = page.getByText(/API 500:|Job failed|Request failed|Failed to fetch/i).first();
  await Promise.race([
    expect(success).toBeVisible({ timeout: LONG_TIMEOUT_MS }),
    expect(failure)
      .toBeVisible({ timeout: LONG_TIMEOUT_MS })
      .then(async () => {
        const msg = (await failure.textContent()) ?? "unknown error";
        throw new Error(`workflow failed: ${msg}`);
      }),
  ]);
}

async function parseKpiValue(page: Page, title: string): Promise<number | null> {
  const label = await page
    .locator(`[aria-label^="${title}:"]`)
    .first()
    .getAttribute("aria-label");
  if (!label) return null;
  const valueText = label.split(":").slice(1).join(":").trim().replace(/,/g, "");
  const parsed = Number(valueText);
  return Number.isFinite(parsed) ? parsed : null;
}

test("backtest flow renders attribution payload without 500", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/backtest");
  await page.getByLabel("Start Date").fill("2021-01-01");
  await page.getByLabel("End Date").fill("2022-12-31");
  await page.getByTestId("backtest-run").click();

  await waitForSuccessOrFailure(page, page.getByRole("tab", { name: "Risk" }));
  await expect(page.getByRole("tab", { name: "Holdings" })).toBeVisible();

  await page.getByRole("tab", { name: "Holdings" }).click();
  await expect(page.getByText("Top Holdings")).toBeVisible();
  await assertNoApiFailure(page);
});

test("factor lab combine with 2+ signals returns tabs without 500", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/factor-lab");
  const rowChecks = page.getByRole("checkbox", { name: "Select row" });
  await expect(rowChecks.first()).toBeVisible({ timeout: 60_000 });
  await rowChecks.nth(0).check();
  await rowChecks.nth(1).check();

  await page.locator("#factor-lab-start").fill("2020-01-01");
  await page.locator("#factor-lab-end").fill("2022-12-31");
  await page.getByTestId("factor-lab-combine-run").click();

  await waitForSuccessOrFailure(page, page.getByRole("tab", { name: "Curve" }));
  await expect(page.getByRole("tab", { name: "Breakdown" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Correlation" })).toBeVisible();
  await assertNoApiFailure(page);
});

test("analytics paste workflow renders performance and rolling", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/analytics");
  await page
    .locator("#analytics-equity-csv")
    .fill("2024-01-02,100\n2024-01-03,101\n2024-01-04,102\n2024-01-05,101\n2024-01-08,103\n2024-01-09,104");
  await page.getByTestId("analytics-run").click();

  await waitForSuccessOrFailure(page, page.getByRole("tab", { name: "Walk-Forward" }));
  await expect(page.getByRole("tab", { name: "Rolling" })).toBeVisible();
  await page.getByRole("tab", { name: "Rolling" }).click();
  await expect(page.getByText("Rolling Sharpe")).toBeVisible();
  await assertNoApiFailure(page);
});

test("optimization submits sweep, polls job, and renders best trial", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/optimization");
  const sweepCheckbox = page
    .locator('fieldset:has(legend:has-text("Parameter Sweep")) input[type="checkbox"]')
    .first();
  await expect(sweepCheckbox).toBeVisible({ timeout: 60_000 });
  await sweepCheckbox.check();
  await page.locator("#optimization-max-trials").fill("4");
  await page.locator("#optimization-start").fill("2024-01-01");
  await page.locator("#optimization-end").fill("2024-06-30");

  await page.getByTestId("optimization-run").click();

  await waitForSuccessOrFailure(page, page.getByRole("tab", { name: "All Trials" }));
  await page.getByRole("tab", { name: "All Trials" }).click();
  await expect(page.getByText("Trial Table")).toBeVisible();
  await assertNoApiFailure(page);
});

test("signal construction renders snapshot and backtest visuals", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/signal-construction");
  await page.getByRole("tab", { name: "Indicator Builder" }).click();

  await page.getByTestId("signal-snapshot-run").click();
  await waitForSuccessOrFailure(page, page.getByText("Snapshot Outputs"));

  await page.getByTestId("signal-backtest-run").click();
  await waitForSuccessOrFailure(page, page.getByText("Indicator Backtest Equity Curve"));
  await assertNoApiFailure(page);
});

test("screener filters materially change result set", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/screener");
  await page.getByTestId("screener-run").click();
  await waitForSuccessOrFailure(page, page.getByRole("heading", { name: "Results" }));

  const baseCount = await parseKpiValue(page, "After Filters");

  await page.getByLabel("Min P/E").fill("50");
  await page.getByTestId("screener-run").click();
  await waitForSuccessOrFailure(page, page.getByRole("heading", { name: "Applied Filters" }));

  const filteredCount = await parseKpiValue(page, "After Filters");
  expect(baseCount).not.toBeNull();
  expect(filteredCount).not.toBeNull();
  expect(filteredCount).not.toEqual(baseCount);

  await assertNoApiFailure(page);
});

test("compliance PASS/FAIL follows backend truth", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/compliance");

  await page.getByTestId("compliance-run-check").click();
  await waitForSuccessOrFailure(page, page.getByText(/PASS|FAIL/).first());
  await expect(page.getByText(/PASS/i).first()).toBeVisible();

  await page.locator("#compliance-quantity").fill("250000");
  await page.getByTestId("compliance-run-check").click();
  await waitForSuccessOrFailure(page, page.getByText(/FAIL/i).first());

  await assertNoApiFailure(page);
});

test("professional and agents flows remain functional", async ({ page }) => {
  test.setTimeout(LONG_TIMEOUT_MS);

  await page.goto("/professional");
  await page.getByTestId("professional-run-greeks").click();
  await waitForSuccessOrFailure(page, page.getByText("Greek Sensitivity Overview"));

  await page.getByRole("tab", { name: "Pip Value" }).click();
  await page.getByTestId("professional-run-pip").click();
  await waitForSuccessOrFailure(page, page.getByText("Pip Value Result"));

  await page.goto("/agents");
  await page.locator("#agents-prompt").fill("Summarize current market regime and top momentum names.");
  await page.getByRole("button", { name: "Send Stub Prompt" }).click();
  await expect(page.getByRole("heading", { name: "Session Log" })).toBeVisible();

  await assertNoApiFailure(page);
});
