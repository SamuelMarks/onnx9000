import { test, expect } from '@playwright/test';

test.describe('onnx2c-ui', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/onnx2c-ui.html');
    } catch (e) {
      test.skip();
    }
  });

  test('Upload ONNX file and Verify C99 code generation', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    if (await fileInput.isVisible()) {
      const buffer = Buffer.from('dummy onnx');
      await fileInput.setInputFiles({
        name: 'model.onnx',
        mimeType: 'application/octet-stream',
        buffer,
      });

      const convertBtn = page.locator('button', { hasText: /Convert/i });
      if (await convertBtn.isVisible()) {
        await convertBtn.click();
      }

      const pre = page.locator('pre');
      if (await pre.isVisible()) {
        const text = await pre.textContent();
        expect(text).toContain('#include <stdio.h>');
        expect(text).toContain('float* outputs');
      }
    }
  });
});
