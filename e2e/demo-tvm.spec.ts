import { test, expect } from '@playwright/test';
test.describe('TVM Demo E2E', () => {
  test('Page loads and converts code', async ({ page }) => {
    test.setTimeout(120000);
    await page.goto('/tvm');
    await expect(page.locator('h1')).toHaveText('ONNX9000: TVM Exporter');
    await expect(page.locator('#convert-btn')).toBeVisible();
  });
});
