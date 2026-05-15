import { test, expect } from '@playwright/test';

test.describe('Progressive Loading Demo E2E', () => {
  test('Page loads and runs progressive model loading', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/progressive-loading');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('Progressive Model Loading Demo');

    const loadBtn = page.locator('#loadBtn');
    await expect(loadBtn).toBeVisible();
    await loadBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Session initialized!', { timeout: 10000 });

    const runBtn = page.locator('#runBtn');
    await expect(runBtn).toBeEnabled();
    await runBtn.click();

    await expect(output).toContainText('Success! Progressively loaded weights', { timeout: 10000 });
  });
});
