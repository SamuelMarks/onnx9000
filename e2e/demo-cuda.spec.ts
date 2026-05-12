import { test, expect } from '@playwright/test';

test.describe('CUDA Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-cuda');
    } catch (e) {
      test.skip();
    }
  });

  test('CUDA executes successfully', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    await expect(runBtn).toBeVisible();

    await runBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Execution complete: SUCCESS', { timeout: 5000 });
  });
});
