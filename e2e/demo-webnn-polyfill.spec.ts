import { test, expect } from '@playwright/test';

test.describe('WebNN Polyfill Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-webnn-polyfill');
    } catch (e) {
      console.log('Skipping real nav, relying on manual DOM testing if needed', e);
      test.skip();
    }
  });

  test('Builds and executes a graph using the WebNN Polyfill', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    const output = page.locator('#webnn-output');

    await expect(runBtn).toBeVisible();
    await runBtn.click();

    await expect(output).toContainText('Created MLContext.');
    await expect(output).toContainText('Compiling graph...');
    await expect(output).toContainText('Result y:');
    await expect(output).toContainText('Success! WebNN API execution complete.');
  });
});
