import { test, expect } from "@playwright/test";

const ROUTES = [
  "/dashboard",
  "/backtest",
  "/factor-lab",
  "/signal-construction",
  "/screener",
  "/analytics",
  "/optimization",
  "/professional",
  "/compliance",
  "/agents",
] as const;

test.describe("UI consistency", () => {
  test("all product routes use scaffold/header and shared controls only", async ({ page }) => {
    for (const route of ROUTES) {
      await test.step(route, async () => {
        await page.goto(route);

        await expect(page.locator("[data-ui-page-header] h1")).toHaveCount(1);
        await expect(page.locator("[data-ui-scaffold]").first()).toBeVisible({ timeout: 60_000 });
        await expect(page.locator("[data-ui-main]").first()).toBeVisible();

        await expect(page.locator("#main-content .page-content pre")).toHaveCount(0);
        await expect(
          page.locator("#main-content .page-content select:not([data-ui-shared-control='select'])")
        ).toHaveCount(0);
        await expect(
          page.locator("#main-content .page-content textarea:not([data-ui-shared-control='textarea'])")
        ).toHaveCount(0);
        await expect(
          page.locator("#main-content .page-content table:not([data-ui-shared-control='table'])")
        ).toHaveCount(0);

        const nav = page.locator("aside[aria-label='Main navigation']");
        await expect(nav).toBeVisible();
        const navLinks = nav.locator("nav a");
        await expect(navLinks.first()).toBeVisible();
        expect(await navLinks.count()).toBeGreaterThanOrEqual(10);
      });
    }
  });

  test("density toggle updates global density tokens", async ({ page }) => {
    await page.goto("/backtest");

    const densityToggle = page.locator("[data-ui-density-toggle]:visible").first();
    await expect(densityToggle).toBeVisible();

    const initialDensity = await page.evaluate(() =>
      document.documentElement.getAttribute("data-density")
    );
    const initialControlHeight = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue("--control-h").trim()
    );

    await densityToggle.click();

    await expect
      .poll(async () =>
        page.evaluate(() => document.documentElement.getAttribute("data-density"))
      )
      .not.toBe(initialDensity);

    const nextControlHeight = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue("--control-h").trim()
    );

    expect(nextControlHeight).not.toBe(initialControlHeight);
  });
});
