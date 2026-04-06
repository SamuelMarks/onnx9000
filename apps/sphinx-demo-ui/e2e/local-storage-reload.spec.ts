import { test, expect } from '@playwright/test';

test.describe('localStorage state preservation', () => {
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

  test('verify page reload restores selected LHS source', async ({ page }) => {
    const lhsDropdown = page.locator('.demo-pane-lhs .demo-dropdown').first();

    await lhsDropdown.locator('button').click();
    await lhsDropdown.locator('.demo-dropdown-item').filter({ hasText: 'TensorFlow' }).click();

    // Verify it updated
    await expect(lhsDropdown.locator('button')).toHaveText('TensorFlow');

    // Reload the page
    await page.reload();

    // Should remember TensorFlow
    const newLhsDropdown = page.locator('.demo-pane-lhs .demo-dropdown').first();
    await expect(newLhsDropdown.locator('button')).toHaveText('TensorFlow');
  });

  test('verify page reload restores selected RHS target', async ({ page }) => {
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown').first();

    await rhsDropdown.locator('button').click();
    await rhsDropdown.locator('.demo-dropdown-item').filter({ hasText: 'PyTorch' }).click();

    // Verify it updated
    await expect(rhsDropdown.locator('button')).toHaveText('PyTorch');

    // Reload the page
    await page.reload();

    // Should remember PyTorch
    const newRhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown').first();
    await expect(newRhsDropdown.locator('button')).toHaveText('PyTorch');
  });
});
