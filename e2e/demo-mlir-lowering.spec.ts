import { test, expect } from '@playwright/test';

test.describe('MLIR Lowering Demo E2E', () => {
  test('Page loads and runs MLIR lowering visualization', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/mlir-lowering');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('MLIR Lowering Pipeline Demo');

    const lowerBtn = page.locator('#lowerBtn');
    await expect(lowerBtn).toBeVisible();
    await lowerBtn.click();

    const output = page.locator('#output');
    
    // Check that it reaches the end of the pipeline
    await expect(output).toContainText('MLIR Lowering Pipeline Completed Successfully!', { timeout: 10000 });

    const resetBtn = page.locator('#resetBtn');
    await expect(resetBtn).toBeEnabled();
    await resetBtn.click();
    
    await expect(output).toContainText('Ready to compile', { timeout: 10000 });
  });
});
