import { test, expect } from '@playwright/test';

test.describe('RHS (Target) Implementation', () => {
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
      await overlayBtn.click({ force: true });
      await page.locator('.demo-wasm-overlay').waitFor({ state: 'hidden' });
    }
  });

  test('should verify changing RHS dropdown triggers the conversion flow / updates tree', async ({
    page
  }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Default tree shows 'output-onnx'
    await expect(tree).toContainText('output-onnx');
    await expect(tree).toContainText('model.onnx');

    // Click to open dropdown
    await button.click({ force: true });
    await expect(listbox).toBeVisible();

    // Select PyTorch
    const item = listbox.locator('.demo-dropdown-item').filter({ hasText: 'PyTorch' });
    await item.click({ force: true });

    // Verify tree updated to output-pytorch
    await expect(tree).toContainText('output-pytorch');
    await expect(tree).toContainText('module.py');
  });

  test('should verify clicking an RHS file displays read-only content in RHS editor', async ({
    page
  }) => {
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Make sure we have the initial ONNX files
    await expect(tree).toContainText('output-onnx');

    // Click model.onnx
    const file1 = tree.locator('.demo-tree-file').filter({ hasText: 'model.onnx' });
    await file1.click({ force: true });

    // Editor updates
    await expect(editorLines).toContainText('// Binary representation of /output-onnx/model.onnx');
  });

  test('should compile C++ code when C++ target is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox.locator('.demo-dropdown-item').filter({ hasText: 'C++' }).click({ force: true });

    await expect(editorLines).toContainText('namespace model_');
  });

  test('should compile CoreML when Apple CoreML target is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox
      .locator('.demo-dropdown-item')
      .filter({ hasText: 'Apple CoreML' })
      .click({ force: true });

    // the manifest structure starts with {
    await expect(editorLines).toContainText('{');
  });

  test('should compile PyTorch when PyTorch target is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox
      .locator('.demo-dropdown-item')
      .filter({ hasText: 'PyTorch' })
      .click({ force: true });

    await expect(editorLines).toContainText('import torch'); // the mocked converter returns Exported pytorch_code content for placeholder
  });

  test('should optimize Olive when Optimize (Olive) target is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox
      .locator('.demo-dropdown-item')
      .filter({ hasText: 'Optimize (Olive)' })
      .click({ force: true });

    await expect(editorLines).toContainText('producer_name');
  });

  test('should generate MLIR when MLIR target is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox.locator('.demo-dropdown-item').filter({ hasText: 'MLIR' }).click({ force: true });

    await expect(editorLines).toContainText('module');
  });

  test('should simplify ONNX when Simplify (onnx-simplifier) target is selected', async ({
    page
  }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox
      .locator('.demo-dropdown-item')
      .filter({ hasText: 'Simplify (onnx-simplifier)' })
      .click({ force: true });

    await expect(editorLines).toContainText('producer_name');
  });

  test('should convert to target when Caffe is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    // Generate an ONNX binary first
    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox
      .locator('.demo-dropdown-item')
      .filter({ hasText: 'Caffe' })
      .click({ force: true });

    await expect(editorLines).toContainText('layer {');
  });

  test('should convert to Keras when Keras is selected', async ({ page }) => {
    const dropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    const button = dropdown.locator('.demo-dropdown-button');
    const listbox = dropdown.locator('.demo-dropdown-listbox');
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');

    await page.locator('.demo-pane-lhs .demo-btn-run-conversion').click({ force: true });

    await button.click({ force: true });
    await listbox
      .locator('.demo-dropdown-item')
      .filter({ exact: true, hasText: 'Keras' })
      .click({ force: true });

    await expect(editorLines).toContainText('import keras');
  });
});
