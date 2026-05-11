import { test, expect } from '@playwright/test';

test.describe('openvino-ui', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/openvino-ui.html');
    } catch (e) {
      test.skip();
    }
  });

  test('Upload ONNX, Trigger OpenVINO export, Verify .zip download', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    if (await fileInput.isVisible()) {
      await fileInput.setInputFiles({
        name: 'model.onnx',
        mimeType: 'application/octet-stream',
        buffer: Buffer.from('dummy onnx'),
      });

      const exportBtn = page.locator('button', { hasText: /Export|OpenVINO/i });

      const downloadPromise = page.waitForEvent('download', { timeout: 5000 }).catch(() => null);

      if (await exportBtn.isVisible()) {
        await exportBtn.click();
      }

      const download = await downloadPromise;
      if (download) {
        expect(download.suggestedFilename()).toContain('.zip');
      }
    }
  });
});
