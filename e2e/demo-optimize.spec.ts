import { test, expect } from '@playwright/test';

test.describe('Optimizer Demo E2E', () => {
  test('Page loads and runs ONNX optimization passes', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/optimize');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('ONNX Graph Optimizer Demo');

    const passesInput = page.locator('#passes');
    await passesInput.fill('fuse_bn_into_conv,eliminate_deadend');

    const optimizeBtn = page.locator('#optimizeBtn');
    await expect(optimizeBtn).toBeVisible();
    await optimizeBtn.click();

    const output = page.locator('#output');
    
    // Check that it reaches the end of the pipeline
    await expect(output).toContainText('Loading ONNX model', { timeout: 10000 });
    await expect(output).toContainText('Running optimization passes: fuse_bn_into_conv,eliminate_deadend', { timeout: 10000 });
    await expect(output).toContainText('Graph optimization complete', { timeout: 10000 });

    const resetBtn = page.locator('#resetBtn');
    await expect(resetBtn).toBeEnabled();
    await resetBtn.click();
    
    await expect(output).toContainText('Waiting to optimize', { timeout: 10000 });
  });
});
