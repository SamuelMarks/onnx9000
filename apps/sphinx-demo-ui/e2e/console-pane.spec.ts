import { test, expect } from '@playwright/test';

test.describe('Console Implementation', () => {
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

  test('should verify WASM execution outputs text to the console', async ({ page }) => {
    // We can simulate WASM output by just executing console.log in the browser since our Logger intercepts it
    await page.evaluate(() => {
      console.log('WASM conversion started...');
      console.warn('WASM encountered a minor issue');
      console.error('WASM fatal failure');
    });

    const consoleOutput = page.locator('.demo-console-output');

    // We should see these messages in the UI
    await expect(consoleOutput).toContainText('WASM conversion started...');
    await expect(consoleOutput).toContainText('WASM encountered a minor issue');
    await expect(consoleOutput).toContainText('WASM fatal failure');
  });

  test('should verify Clear empties the console', async ({ page }) => {
    await page.evaluate(() => {
      console.log('Dummy message 1');
      console.log('Dummy message 2');
    });

    const consoleOutput = page.locator('.demo-console-output');
    await expect(consoleOutput).toContainText('Dummy message 1');

    const clearBtn = page.locator('.demo-console-clear-btn');
    await clearBtn.click();

    // The console should be empty
    const lines = await consoleOutput.locator('.demo-console-line').count();
    expect(lines).toBe(0);
    await expect(consoleOutput).not.toContainText('Dummy message 1');
  });
});
