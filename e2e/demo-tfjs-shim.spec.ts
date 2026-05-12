import { test, expect } from '@playwright/test';

test.describe('TFJS Shim Demo E2E', () => {
  test('Page loads and runs tensor operations', async ({ page }) => {
    await page.goto('/tfjs-shim');

    await expect(page.locator('h1')).toHaveText('ONNX9000: TFJS Shim Replacement');
    await expect(page.locator('#run-btn')).toBeVisible();

    await page.click('#run-btn');

    const output = page.locator('#output');
    await expect(output).toContainText('Running operations...');
    await expect(output).toContainText('Tensor A:');
    await expect(output).toContainText('Tensor B:');
    await expect(output).toContainText('C = matMul(A, B):');
    await expect(output).toContainText('D = relu(sub(A, 2)):');
    await expect(output).toContainText('Operations completed inside tf.tidy scope.');
  });
});
