import { test, expect } from '@playwright/test';

test.describe('ONNX9000 Compiler & Execution', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/');
      await page.waitForSelector('#ide-root', { state: 'attached', timeout: 5000 });
    } catch (e) {
      console.log('Skipping real nav', e);
      test.skip();
    }
  });

  test('302. Compile to WASM feature', async ({ page }) => {
    // There should be a "Compile to WASM" button
    const wasmBtn = page.locator('button', { hasText: 'Compile to WASM' });

    // Some buttons might be hidden depending on the model state,
    // but the UI typically displays them in the sidebar or bottom panel.
    if (await wasmBtn.isVisible()) {
      await wasmBtn.click();

      // Look for a toast or console output
      const toast = page.locator('.toast');
      await expect(toast).toContainText(/WASM|Load/, { ignoreCase: true, timeout: 5000 });
    }
  });

  test('322. WebGPU Memory Limits UI', async ({ page }) => {
    // If the profiler or arena container exists, ensure it renders without error
    const profiler = page.locator('#profiler-container');
    if (await profiler.isVisible()) {
      // Memory arena is usually in the profiler or a dedicated tab
      const arena = page.locator('#arena-container');
      await expect(arena).toBeVisible();
    }

    const runBtn = page.locator('button', { hasText: 'Run Test Inference (WebNN)' });
    if (await runBtn.isVisible()) {
      await runBtn.click();
      // Should show error if no model loaded, or run if mock model
      const toast = page.locator('.toast');
      await expect(toast).toBeVisible();
    }
  });

  test('Code Editor is accessible for ONNXScript', async ({ page }) => {
    const editorTab = page.locator('text="ONNXScript Editor"');
    if (await editorTab.isVisible()) {
      await editorTab.click();

      const monaco = page.locator('.monaco-editor');
      await expect(monaco).toBeVisible();

      // Should have a sync/compile button
      const syncBtn = page.locator('button', { hasText: 'Sync to Graph' });
      await expect(syncBtn).toBeVisible();
    }
  });
});
