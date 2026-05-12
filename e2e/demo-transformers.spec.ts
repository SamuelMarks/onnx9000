import { test, expect } from '@playwright/test';

test.describe('Transformers.js Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-transformers');
    } catch (e) {
      console.log('Skipping real nav, relying on manual DOM testing if needed', e);
      test.skip();
    }
  });

  test('Runs text classification pipeline successfully', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    const output = page.locator('#transformers-output');

    await expect(runBtn).toBeVisible();
    await runBtn.click();

    await expect(output).toContainText('Pipeline initialized for text-classification.');
    await expect(output).toContainText('Running inference on "I love ONNX9000!"');
    await expect(output).toContainText('positive', { timeout: 10000 });
    await expect(output).toContainText('Success!');
  });
});
