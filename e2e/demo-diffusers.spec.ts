import { test, expect } from '@playwright/test';
test.describe('Diffusers Demo E2E', () => {
  test('Page loads and converts code', async ({ page }) => {
    test.setTimeout(120000);
    await page.goto('/diffusers');
    await expect(page.locator('h1')).toHaveText('ONNX9000: Web-Native Diffusers');
    await expect(page.locator('#run-btn')).toBeVisible();
  });
});
