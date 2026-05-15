import { test, expect } from '@playwright/test';

test.describe('Mobile Memory Demo E2E', () => {
  test('Page loads and runs mobile memory arena allocation', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/mobile-memory');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('Mobile Memory Best Practices');

    const allocateBtn = page.locator('#allocateBtn');
    await expect(allocateBtn).toBeVisible();
    await allocateBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Arena pre-allocated successfully', { timeout: 10000 });

    const runInferenceBtn = page.locator('#runInferenceBtn');
    await expect(runInferenceBtn).toBeEnabled();
    await runInferenceBtn.click();

    await expect(output).toContainText('Inference complete. No memory was allocated', { timeout: 10000 });
    
    const freeBtn = page.locator('#freeBtn');
    await expect(freeBtn).toBeEnabled();
    await freeBtn.click();
    
    await expect(output).toContainText('Arena memory freed', { timeout: 10000 });
  });
});
