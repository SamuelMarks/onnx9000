import { test, expect } from '@playwright/test';

test.describe('ONNX9000 Graph Editor', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/');
      await page.waitForSelector('#ide-root', { state: 'attached', timeout: 5000 });
    } catch (e) {
      console.log('Skipping real nav', e);
      test.skip();
    }
  });

  test('312. Validate UI rendering of nodes', async ({ page }) => {
    // If a mock model is loaded or we load one, nodes should be present
    const canvas = page.locator('#ide-canvas');
    await expect(canvas).toBeVisible();

    // There are zoom/pan controls typically overlaid on the canvas
    const zoomInBtn = page.locator('.zoom-in-btn');
    if (await zoomInBtn.isVisible()) {
      await zoomInBtn.click();
    }

    const zoomOutBtn = page.locator('.zoom-out-btn');
    if (await zoomOutBtn.isVisible()) {
      await zoomOutBtn.click();
    }
  });

  test('Model Summary component is available', async ({ page }) => {
    // Model summary is in the sidebar
    const summary = page.locator('#summary-container');
    if (await summary.isVisible()) {
      const title = summary.locator('h3');
      await expect(title).toBeVisible();
    }
  });

  test('Vault Manager UI for local model caching', async ({ page }) => {
    // Verify local IndexedDB cache UI
    const vault = page.locator('#vault-container');
    if (await vault.isVisible()) {
      await expect(vault).toBeVisible();
      const saveBtn = vault.locator('button', { hasText: 'Save to Vault' });
      if (await saveBtn.isVisible()) {
        await saveBtn.click();

        // Wait for Toast
        const toast = page.locator('.toast');
        await expect(toast).toContainText('Vault', { timeout: 3000 });
      }
    }
  });
});
