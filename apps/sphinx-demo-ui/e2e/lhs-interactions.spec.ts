import { test, expect } from '@playwright/test';

test.describe('LHS (Source) Implementation', () => {
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

  test('should verify changing LHS dropdown updates the file tree', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-lhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-lhs .demo-file-tree');

    // Default tree shows 'example' (which is the default dummy root we set originally)
    // Actually, in LHSContainer we default to LHS_SOURCES.keras
    await expect(tree).toContainText('keras-model');
    await expect(tree).toContainText('model.h5');

    // Click to open dropdown
    await button.click();
    await expect(listbox).toBeVisible();

    // Select Scikit-Learn
    const item = listbox.locator('.demo-dropdown-item').filter({ hasText: 'Scikit-Learn' });
    await item.click();

    // Verify tree updated to sklearn-pipeline
    await expect(tree).toContainText('sklearn-pipeline');
    await expect(tree).toContainText('pipeline.pkl');
  });

  test('should verify clicking an LHS file updates the LHS Monaco editor', async ({ page }) => {
    const tree = page.locator('.demo-pane-lhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-lhs .demo-editor-container .view-lines');

    // Make sure we have the initial Keras files
    await expect(tree).toContainText('keras-model');

    // Click model.h5
    const file1 = tree.locator('.demo-tree-file').filter({ hasText: 'model.h5' });
    await file1.click();

    // Editor updates
    await expect(editorLines).toContainText('# /keras-model/model.h5');

    // Click train.py
    const file2 = tree.locator('.demo-tree-file').filter({ hasText: 'train.py' });
    await file2.click();

    // Editor updates
    await expect(editorLines).toContainText('# /keras-model/train.py');
  });
});
