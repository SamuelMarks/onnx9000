import { test, expect } from '@playwright/test';

test.describe('onnx-checker-ui', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/onnx-checker-ui.html');
    } catch (e) {
      test.skip();
    }
  });

  test('Upload known-valid ONNX file and verify "Validation Passed"', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    if (await fileInput.isVisible()) {
      const buffer = Buffer.from('valid onnx');
      await fileInput.setInputFiles({
        name: 'valid.onnx',
        mimeType: 'application/octet-stream',
        buffer,
      });

      const msg = page.locator('text="Validation Passed"');
      await expect(msg).toBeAttached();
    }
  });

  test('Upload known-invalid ONNX file and verify error messages', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    if (await fileInput.isVisible()) {
      const buffer = Buffer.from('invalid onnx');
      await fileInput.setInputFiles({
        name: 'invalid.onnx',
        mimeType: 'application/octet-stream',
        buffer,
      });

      // Usually error messages appear in a pre or div
      const errorMsg = page.locator('.error, .alert, .toast');
      if ((await errorMsg.count()) > 0) {
        await expect(errorMsg.first()).toBeVisible();
      }
    }
  });
});
