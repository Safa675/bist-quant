import { test, expect } from "@playwright/test";

test.describe("Navigation", () => {
  test("root redirects and sidebar drives route-level controls", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/dashboard/);

    const sidebar = page.getByRole("complementary", { name: "Main navigation" });
    await expect(sidebar).toBeVisible();

    await sidebar.getByRole("link", { name: "Backtest" }).first().click();
    await expect(page).toHaveURL(/\/backtest/);
    await expect(page.getByTestId("backtest-run")).toBeVisible();

    await sidebar.getByRole("link", { name: "Analytics" }).first().click();
    await expect(page).toHaveURL(/\/analytics/);
    await expect(page.getByTestId("analytics-run")).toBeVisible();
  });

  test("skip-to-content focuses main container", async ({ page }) => {
    await page.goto("/dashboard");
    const skipLink = page.locator('a[href="#main-content"]');
    await expect(skipLink).toBeAttached();
    await skipLink.focus();
    await skipLink.press("Enter");
    await expect(page.locator("#main-content")).toBeFocused();
  });
});

test.describe("Error handling", () => {
  test("404 page shows not found message", async ({ page }) => {
    await page.goto("/nonexistent-route-xyz");
    await expect(page.locator("body")).toContainText(/not found|404/i);
  });
});
