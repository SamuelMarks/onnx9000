import { test, expect } from '@playwright/test';

test.describe('Autograd Demo E2E', () => {
  test('Page loads and generates gradients', async ({ page }) => {
    test.setTimeout(120000);
    await page.goto('/autograd');

    await expect(page.locator('h1')).toHaveText('ONNX9000: Web Autograd Engine');

    const gradBtn = page.locator('#grad-btn');
    await expect(gradBtn).toBeVisible();

    await gradBtn.click();

    const output = page.locator('#autograd-output');
    await expect(output).toContainText(
      'Augmented ONNX graph now computes forward pass + gradients',
      { timeout: 10000 },
    );
  });
});
