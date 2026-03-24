import { test, expect } from '@playwright/test';

test.describe('Shared UI Components (File Tree, Monaco, Dropdown)', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    // Wait for initial render
    await page.locator('.demo-ui-root').waitFor({ state: 'visible' });

    // Close overlay if it's there
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

  test('should verify dropdown opens, selects item, and closes', async ({ page }) => {
    // There are two dropdowns, let's grab the LHS one
    const dropdown = page.locator('.demo-pane-lhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');

    // Initial state
    await expect(button).toHaveText('Select Source Framework...');
    await expect(listbox).toBeHidden();

    // Click to open
    await button.click();
    await expect(listbox).toBeVisible();

    // Select second item (TensorFlow)
    const item = listbox.locator('.demo-dropdown-item').filter({ hasText: 'TensorFlow' });
    await item.click();

    // Verify it closed and button updated
    await expect(listbox).toBeHidden();
    await expect(button).toHaveText('TensorFlow');
  });

  test('should verify file tree expands/collapses', async ({ page }) => {
    const tree = page.locator('.demo-pane-lhs .demo-file-tree');
    const rootDir = tree.locator('.demo-tree-dir');
    const rootChildren = tree.locator('.demo-file-tree-children').first();

    // In our implementation, root is expanded by default
    await expect(rootChildren).toBeVisible();

    // Click label to collapse
    await rootDir.click();
    await expect(rootChildren).toBeHidden();

    // Click again to expand
    await rootDir.click();
    await expect(rootChildren).toBeVisible();
  });

  test('should verify Monaco editor renders text', async ({ page }) => {
    const editorContainer = page.locator('.demo-pane-lhs .demo-editor-container');
    const lines = editorContainer.locator('.view-lines');

    // Initial text should contain # Select a file...
    await expect(lines).toContainText('# Select a file...');

    // Now click a file in the tree to update the editor
    const fileItem = page
      .locator('.demo-pane-lhs .demo-file-tree .demo-tree-file')
      .filter({ hasText: 'model.py' });
    await fileItem.click();

    // The editor should now show "# /example/model.py"
    await expect(lines).toContainText('# /example/model.py');
  });
});
