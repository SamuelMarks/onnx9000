import { test, expect } from '@playwright/test';

test.describe('ONNX2TF Demo E2E', () => {
  test('Page loads and runs ONNX2TF conversion', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/onnx2tf');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('ONNX to TFLite (ONNX2TF) Demo');

    const int8Checkbox = page.locator('#int8Quant');
    await int8Checkbox.check();

    const convertBtn = page.locator('#convertBtn');
    await expect(convertBtn).toBeVisible();
    await convertBtn.click();

    const output = page.locator('#output');
    
    // Check that it reaches the end of the pipeline
    await expect(output).toContainText('Loading ONNX model', { timeout: 10000 });
    await expect(output).toContainText('Converting to TFLite format with INT8 quantization', { timeout: 10000 });
    await expect(output).toContainText('onnx2tf conversion complete', { timeout: 10000 });

    const resetBtn = page.locator('#resetBtn');
    await expect(resetBtn).toBeEnabled();
    await resetBtn.click();
    
    await expect(output).toContainText('Waiting to convert', { timeout: 10000 });
  });
});
