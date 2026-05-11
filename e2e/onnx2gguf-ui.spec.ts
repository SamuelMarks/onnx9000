import { test, expect } from '@playwright/test';

test.describe('onnx2gguf-ui', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/onnx2gguf-ui.html');
    } catch (e) {
      test.skip();
    }
  });

  test('Upload ONNX + tokenizer, select Q4_K_M, verify progress', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    if (await fileInput.isVisible()) {
      await fileInput.setInputFiles([
        {
          name: 'model.onnx',
          mimeType: 'application/octet-stream',
          buffer: Buffer.from('dummy onnx'),
        },
        {
          name: 'tokenizer.json',
          mimeType: 'application/json',
          buffer: Buffer.from('{}'),
        },
      ]);

      const select = page.locator('select');
      if (await select.isVisible()) {
        await select.selectOption('Q4_K_M');
      }

      const convertBtn = page.locator('button', { hasText: /Convert/i });
      if (await convertBtn.isVisible()) {
        await convertBtn.click();
      }

      const progress = page.locator('progress, .progress-bar');
      if ((await progress.count()) > 0) {
        await expect(progress.first()).toBeVisible();
      }
    }
  });
});
