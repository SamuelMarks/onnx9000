import { test, expect } from '@playwright/test';

test.describe('Bottom Pane Tabs & Architecture', () => {
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

  test('should verify clicking tabs toggles visibility of underlying panels', async ({ page }) => {
    const tabList = page.locator('.demo-tab-list');
    const consoleBtn = tabList.locator('button', { hasText: 'Console' });
    const vizBtn = tabList.locator('button', { hasText: 'ONNX Visualization' });

    const consolePanel = page.locator('#panel-console');
    const vizPanel = page.locator('#panel-viz');

    // Initial state: Console should be visible
    await expect(consolePanel).toBeVisible();
    await expect(vizPanel).toBeHidden();
    await expect(consoleBtn).toHaveClass(/active/);

    // Click Viz tab
    await vizBtn.click();
    await expect(vizPanel).toBeVisible();
    await expect(consolePanel).toBeHidden();
    await expect(vizBtn).toHaveClass(/active/);

    // Click Console tab back
    await consoleBtn.click();
    await expect(consolePanel).toBeVisible();
    await expect(vizPanel).toBeHidden();
    await expect(consoleBtn).toHaveClass(/active/);
  });

  test('should verify keyboard navigation switches tabs', async ({ page }) => {
    const tabList = page.locator('.demo-tab-list');
    const consoleBtn = tabList.locator('button', { hasText: 'Console' });
    const vizBtn = tabList.locator('button', { hasText: 'ONNX Visualization' });

    // Focus the active tab first (Console)
    await consoleBtn.focus();

    // Press ArrowRight to go to Viz
    await tabList.press('ArrowRight');

    // Verify Viz is active
    await expect(vizBtn).toHaveClass(/active/);
    const vizPanel = page.locator('#panel-viz');
    await expect(vizPanel).toBeVisible();

    // Press ArrowLeft to go back to Console
    await tabList.press('ArrowLeft');

    await expect(consoleBtn).toHaveClass(/active/);
    const consolePanel = page.locator('#panel-console');
    await expect(consolePanel).toBeVisible();

    // Press End to go to Viz
    await tabList.press('End');
    await expect(vizBtn).toHaveClass(/active/);

    // Press Home to go to Console
    await tabList.press('Home');
    await expect(consoleBtn).toHaveClass(/active/);
  });
});
