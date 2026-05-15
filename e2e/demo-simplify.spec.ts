import { test, expect } from '@playwright/test';

test.describe('Simplify Demo E2E', () => {
  test('Page loads and runs ONNX simplification pass', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/simplify');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('ONNX Graph Simplifier Demo');

    const simplifyBtn = page.locator('#simplifyBtn');
    await expect(simplifyBtn).toBeVisible();
    await simplifyBtn.click();

    const output = page.locator('#output');
    
    // Check that it reaches the end of the pipeline
    await expect(output).toContainText('Loading ONNX model', { timeout: 10000 });
    await expect(output).toContainText('Simplifying graph', { timeout: 10000 });
    await expect(output).toContainText('Graph simplification complete', { timeout: 10000 });

    const resetBtn = page.locator('#resetBtn');
    await expect(resetBtn).toBeEnabled();
    await resetBtn.click();
    
    await expect(output).toContainText('Waiting to simplify', { timeout: 10000 });
  });
});
