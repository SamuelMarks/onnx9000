import { test, expect } from '@playwright/test';

test.describe('keras2onnx Conversion', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    // Load WASM and wait for overlay to vanish
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

  test('should verify the bottom pane exists and has a console log when running conversion', async ({
    page
  }) => {
    // Verify bottom pane exists
    const bottomPane = page.locator('.demo-pane-bottom');
    await expect(bottomPane).toBeVisible();

    // Verify Console tab exists
    const consoleTabBtn = page.locator('#tab-console');
    await expect(consoleTabBtn).toBeVisible();
    await expect(consoleTabBtn).toHaveText('Console');

    // 1. Ensure Keras is selected
    const frameworkDropdownBtn = page
      .locator('.demo-pane-lhs .demo-dropdown')
      .first()
      .locator('button');
    await frameworkDropdownBtn.click();
    const kerasItem = page
      .locator('.demo-pane-lhs .demo-dropdown')
      .first()
      .locator('.demo-dropdown-item[data-value="keras"]');
    await kerasItem.click();

    // 2. Click "Run Conversion"
    const runBtn = page.locator('.demo-btn-run-conversion');
    await runBtn.click();

    // 3. Verify console logs in the bottom pane
    const consoleOutput = page.locator('.demo-console-output');

    // It should output something about keras2onnx
    await expect(consoleOutput).toContainText('Validating Keras JSON source', { timeout: 10000 });
    await expect(consoleOutput).toContainText('Conversion successful', { timeout: 10000 });

    // Verify Visualization tab exists as well
    const vizTabBtn = page.locator('#tab-viz');
    await expect(vizTabBtn).toBeVisible();
  });
});
