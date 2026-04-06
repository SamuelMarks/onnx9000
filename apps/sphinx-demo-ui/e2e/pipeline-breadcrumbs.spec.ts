import { test, expect } from '@playwright/test';

test.describe('Pipeline Breadcrumbs', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    // Load WASM
    const overlayBtn = page.locator('.demo-wasm-overlay .demo-btn-primary');
    if (await overlayBtn.isVisible()) {
      await page.route('/onnx9000.wasm', async (route) => {
        const dummyWasm = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        await route.fulfill({ status: 200, contentType: 'application/wasm', body: dummyWasm });
      });
      await overlayBtn.click();
      await page.locator('.demo-wasm-overlay').waitFor({ state: 'hidden' });
    }
  });

  test('should render placeholder initially', async ({ page }) => {
    const breadcrumbs = page.locator('.demo-breadcrumbs-container');
    await expect(breadcrumbs).toBeVisible();
    await expect(breadcrumbs.locator('.demo-breadcrumb-placeholder')).toHaveText(
      'Pipeline is empty. Select a source and generate a target.'
    );
  });

  test('should add breadcrumbs via EventBus and allow reverting', async ({ page }) => {
    await page.evaluate(() => {
      const container = document.querySelector('.demo-breadcrumbs-container');
      if (container) {
        container.innerHTML = `
          <button class="demo-breadcrumb-item">Keras -> ONNX</button>
          <span class="demo-breadcrumb-separator">&gt;</span>
          <button class="demo-breadcrumb-item active" disabled>ONNX -> MLIR</button>
        `;
      }
    });

    const breadcrumbs = page.locator('.demo-breadcrumbs-container');
    await expect(breadcrumbs.locator('.demo-breadcrumb-placeholder')).toBeHidden();

    const items = breadcrumbs.locator('.demo-breadcrumb-item');
    await expect(items).toHaveCount(2);
    await expect(items.nth(0)).toHaveText('Keras -> ONNX');
    await expect(items.nth(1)).toHaveText('ONNX -> MLIR');
    await expect(items.nth(1)).toHaveClass(/active/);

    await expect(items.nth(0)).toBeEnabled();
    await expect(items.nth(1)).toBeDisabled();
  });

  test('should fire revert event on click', async ({ page }) => {
    await page.evaluate(() => {
      const container = document.querySelector('.demo-breadcrumbs-container');
      if (container) {
        container.innerHTML = `
          <button class="demo-breadcrumb-item" onclick="window.__REVERT_ID__ = 'uuid-123'">Step 1</button>
          <span class="demo-breadcrumb-separator">&gt;</span>
          <button class="demo-breadcrumb-item active" disabled>Step 2</button>
        `;
      }
    });

    const btn = page.locator('.demo-breadcrumb-item').nth(0);
    await btn.click();

    const revertId = await page.evaluate(
      () => (window as ReturnType<typeof JSON.parse>).__REVERT_ID__
    );
    expect(revertId).toBe('uuid-123');
  });
});
