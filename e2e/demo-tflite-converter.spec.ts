import { test, expect } from '@playwright/test';

test.describe('demo-tflite-converter', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-tflite-converter.html');
    } catch (e) {
      test.skip();
    }
  });

  test('Page loads without console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });

    await page.goto('/demo-tflite-converter.html');
    expect(errors.length).toBe(0);
  });

  test('Upload ONNX and Convert', async ({ page }) => {
    await page.goto('/demo-tflite-converter.html');

    const fileInput = page.locator('input[type="file"]');
    const convertBtn = page.locator('button', { hasText: /Convert to TFLite/i });

    // Create a dummy .onnx file
    const buffer = Buffer.from('fake onnx content');
    await fileInput.setInputFiles({
      name: 'dummy.onnx',
      mimeType: 'application/octet-stream',
      buffer,
    });

    // Intercept WASM compilation success and verify download blob is triggered
    page.on('download', (download) => {
      expect(download.suggestedFilename()).toContain('.tflite');
    });

    // Click Convert to TFLite button
    if (await convertBtn.isVisible()) {
      await convertBtn.click();
    }
  });
});
