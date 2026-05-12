import { test, expect } from '@playwright/test';

test.describe('Hummingbird Demo E2E', () => {
  test('Page loads and runs transpiler', async ({ page }) => {
    test.setTimeout(120000);
    await page.goto('/hummingbird');

    await expect(page.locator('h1')).toHaveText('ONNX9000: Hummingbird Web Transpiler');

    const transpileBtn = page.locator('#transpile-btn');
    await expect(transpileBtn).toBeVisible();

    await transpileBtn.click();

    const output = page.locator('#transpiler-output');
    await expect(output).toContainText('Transpilation successful!', { timeout: 10000 });
  });
});
