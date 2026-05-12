import { test, expect } from '@playwright/test';

test.describe('Sparse Demo E2E', () => {
  test('Page loads and runs sparsification', async ({ page }) => {
    test.setTimeout(120000);
    await page.goto('/sparse');

    await expect(page.locator('h1')).toHaveText('ONNX9000: Model Sparsification & Pruning');

    const pruneBtn = page.locator('#prune-btn');
    await expect(pruneBtn).toBeVisible();

    await pruneBtn.click();

    const output = page.locator('#sparse-output');
    await expect(output).toContainText('Sparsification successful', { timeout: 10000 });
  });
});
