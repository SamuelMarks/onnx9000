import { test, expect } from '@playwright/test';

test.describe('Pipeline C - Edge C/C++ Compilation (ONNX2C)', () => {
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

  test('should verify .onnx converts to C code', async ({ page }) => {
    // Select C in RHS target dropdown
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown').first();
    await rhsDropdown.locator('button').click();
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: /^C$/ })
      .click();

    // Check RHS Tree has output-c
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    await expect(tree).toContainText('output-c');
    await tree.locator('[data-path="/output-c"]').click();
    await tree.locator('[data-path="/output-c/model.c"]').click();

    // Check that we can see the C code in the editor (e.g. #include)
    const editorLine = page.locator('.demo-pane-rhs .view-lines');
    await expect(editorLine).toContainText('#include "model.h"');
  });

  test('should verify .onnx converts to C++ code', async ({ page }) => {
    // Select C++ in RHS target dropdown
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown').first();
    await rhsDropdown.locator('button').click();
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: /^C\+\+$/ })
      .click();

    // Check RHS Tree has output-cpp
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    await expect(tree).toContainText('output-cpp');
    await tree.locator('[data-path="/output-cpp"]').click();
    await tree.locator('[data-path="/output-cpp/model.h"]').click();

    // Check that we can see the C++ code in the editor
    const editorLine = page.locator('.demo-pane-rhs .view-lines');
    await expect(editorLine).toContainText('#include <vector>');
  });
});
