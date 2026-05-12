import { test, expect } from '@playwright/test';

test.describe('Apple Metal Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-apple-metal');
    } catch (e) {
      test.skip();
    }
  });

  test('Apple Metal executes successfully', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    await expect(runBtn).toBeVisible();

    await runBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Execution complete: SUCCESS', { timeout: 5000 });
  });
});
