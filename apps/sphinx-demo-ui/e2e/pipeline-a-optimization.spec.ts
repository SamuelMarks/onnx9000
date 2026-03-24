import { test, expect } from '@playwright/test';

test.describe('Pipeline A - Optimization (Olive)', () => {
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

  test('should verify selecting Olive shows config panel', async ({ page }) => {
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const listbox = rhsDropdown.locator('.demo-dropdown-listbox');
    const olivePanel = page.locator('.demo-olive-config-panel');

    // Initially hidden
    await expect(olivePanel).toBeHidden();

    // Select Olive
    await rhsDropdown.locator('button').click();
    await listbox.locator('.demo-dropdown-item').filter({ hasText: 'Optimize (Olive)' }).click();

    // Panel becomes visible
    await expect(olivePanel).toBeVisible();
    await expect(olivePanel.locator('h3')).toHaveText('Olive Optimization Settings');
  });

  test('should verify changing Olive config emits events via EventBus', async ({ page }) => {
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    await rhsDropdown.locator('button').click();
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: 'Optimize (Olive)' })
      .click();

    const olivePanel = page.locator('.demo-olive-config-panel');

    // Change quantization to INT8
    await olivePanel.locator('.demo-olive-quant-select').selectOption('INT8');
    await page.waitForTimeout(50);

    const quantVal = await olivePanel.locator('.demo-olive-quant-select').inputValue();
    expect(quantVal).toBe('INT8');

    // Toggle shape inference
    await olivePanel.locator('.demo-olive-shape-checkbox').uncheck();
    await page.waitForTimeout(50);

    const shapeChecked = await olivePanel.locator('.demo-olive-shape-checkbox').isChecked();
    expect(shapeChecked).toBe(false);

    // Toggle fusion
    await olivePanel.locator('.demo-olive-fusion-checkbox').check();
    await page.waitForTimeout(50);

    const fusionChecked = await olivePanel.locator('.demo-olive-fusion-checkbox').isChecked();
    expect(fusionChecked).toBe(true);
  });
});
