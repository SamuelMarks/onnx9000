import { test, expect } from '@playwright/test';

test.describe('WASM Lazy Loading & Overlay', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
  });

  test('should verify UI is blocked by overlay initially', async ({ page }) => {
    const overlay = page.locator('.demo-wasm-overlay');
    await expect(overlay).toBeVisible();

    const title = page.locator('.demo-wasm-modal h2');
    await expect(title).toHaveText('Interactive WASM Demo');

    const loadButton = page.locator('.demo-btn-primary');
    await expect(loadButton).toBeVisible();
    await expect(loadButton).toHaveText('Load WASM Live Demo');
  });

  test('should verify clicking load fetches WASM, shows progress, and removes overlay', async ({
    page
  }) => {
    // We mock the /onnx9000.wasm endpoint to simulate a slow WASM download
    await page.route('/onnx9000.wasm', async (route) => {
      // Respond with a tiny dummy WASM valid header to satisfy instantiation (if we were actually instantiating)
      // Actually, since we use WebAssembly.compile, it needs a valid WASM magic number at least: \0asm
      // Hex: 00 61 73 6D 01 00 00 00
      const dummyWasm = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
      await route.fulfill({
        status: 200,
        contentType: 'application/wasm',
        body: dummyWasm,
        headers: {
          'Content-Length': dummyWasm.length.toString()
        }
      });
    });

    const loadButton = page.locator('.demo-btn-primary');
    const progressContainer = page.locator('.demo-wasm-progress-container');
    const overlay = page.locator('.demo-wasm-overlay');

    // Click load
    await loadButton.click();

    // Verify progress container is shown
    await expect(progressContainer).toBeVisible();
    await expect(loadButton).toBeHidden();

    // The fetch should complete very fast since it's intercepted, and then it should fade out
    await expect(overlay).toBeHidden();
  });

  test('should show error state if WASM fetch fails', async ({ page }) => {
    // Intercept and fail the request
    await page.route('/onnx9000.wasm', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'text/plain',
        body: 'Internal Server Error'
      });
    });

    const loadButton = page.locator('.demo-btn-primary');
    const errorText = page.locator('.demo-wasm-error-text');
    const progressContainer = page.locator('.demo-wasm-progress-container');

    await loadButton.click();

    await expect(progressContainer).toBeHidden();
    await expect(errorText).toBeVisible();
    await expect(errorText).toHaveText('HTTP error! status: 500');

    // Load button should reappear to retry
    await expect(loadButton).toBeVisible();
  });
});
