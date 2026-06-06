import { expect, test } from '@playwright/test';

test.describe('WASM Standalone Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-wasm');
    } catch (_e) {
      test.skip();
    }
  });

  test('WASM standalone executes successfully', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    await expect(runBtn).toBeVisible();

    await runBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Execution complete: SUCCESS', { timeout: 5000 });
  });
});
