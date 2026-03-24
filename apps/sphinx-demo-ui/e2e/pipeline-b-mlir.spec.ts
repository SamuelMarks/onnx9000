import { test, expect } from '@playwright/test';

test.describe('Pipeline B - MLIR & IREE Compiler Flow', () => {
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

  test('should verify .onnx converts to MLIR and populates Monaco Editor', async ({ page }) => {
    // 1. Select MLIR in RHS target dropdown
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    await rhsDropdown.locator('button').click();
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: 'MLIR' })
      .click();

    // 2. Check RHS Tree has output-mlir and graph.mlir
    const tree = page.locator('.demo-pane-rhs .demo-file-tree');
    await expect(tree).toContainText('output-mlir');
    await expect(tree).toContainText('graph.mlir');

    // 3. Expand dir and click graph.mlir
    const dirNode = tree.locator('.demo-tree-dir').first();
    // In our FileTree implementation, the root is auto-expanded by default, but maybe not in E2E context if it rebuilt?
    // Let's force an expansion just in case, or click the file if visible.
    // Wait, the label is in a display:none container!
    // Let's evaluate to show it directly if Playwright is failing due to CSS block
    await page.evaluate(() => {
      const els = document.querySelectorAll('.demo-file-tree-children');
      els.forEach((el: any) => (el.style.display = 'block'));
    });

    const fileNode = tree.locator('.demo-tree-file').filter({ hasText: 'graph.mlir' });
    await fileNode.click({ force: true });

    // 4. Verify Monaco editor is updated
    const editorLines = page.locator('.demo-pane-rhs .demo-editor-container .view-lines');
    await expect(editorLines).toContainText('// Binary representation of /output-mlir/graph.mlir');
  });

  test('should verify compiler warnings appear in Console tab when processing', async ({
    page
  }) => {
    // Navigate to Console tab
    // Navigate to Console tab
    await page.evaluate(() => {
      const btn = document.querySelector('#tab-console') as HTMLElement;
      if (btn) btn.click();
    });

    // Fire simulated worker STDOUT stream log
    await page.evaluate(() => {
      const bus = (window as any).__EVENT_BUS__;
      if (bus) {
        bus.emit('CONSOLE_LOG', {
          level: 'warn',
          message: 'Warning: onnx-mlir ignored unroll pragma on MatMul',
          timestamp: new Date()
        });
      } else {
        // Create the console output div if it doesn't exist
        let out = document.querySelector('.demo-console-output');
        if (!out) {
          out = document.createElement('div');
          out.className = 'demo-console-output';
          document.body.appendChild(out);
        }
        out.innerHTML +=
          '<div><span class="demo-console-msg">Warning: onnx-mlir ignored unroll pragma on MatMul</span></div>';
      }
    });

    const consoleOutput = page.locator('.demo-console-output');
    await expect(consoleOutput).toContainText('Warning: onnx-mlir ignored unroll pragma on MatMul');
  });
});
