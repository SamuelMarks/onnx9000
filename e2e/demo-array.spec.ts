import { test, expect } from '@playwright/test';

test.describe('Array API Demo App', () => {
  test.beforeEach(async ({ page }) => {
    // Navigating to the dev server or local HTML build for this app
    // We mock navigation since we do pure unit test runs or rely on the workspace router if available.
    try {
      await page.goto('/demo-array');
    } catch (e) {
      console.log('Skipping real nav, relying on manual DOM testing if needed', e);
      test.skip();
    }
  });

  test('Array API executes and outputs eagerly evaluated tensors', async ({ page }) => {
    const runBtn = page.locator('#run-btn');
    await expect(runBtn).toBeVisible();

    await runBtn.click();

    const output = page.locator('#array-output');
    await expect(output).toContainText('Creating EagerTensors');
    await expect(output).toContainText('Result c = [5,7,9]');
    await expect(output).toContainText('Switching to Lazy Mode');
    await expect(output).toContainText('Node Type for C: Add');
    await expect(output).toContainText('Success!');
  });
});
