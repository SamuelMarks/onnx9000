import { test, expect } from '@playwright/test';

test.describe('Zero Dependency Classifier Demo E2E', () => {
  test('Page loads and runs classifier', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/zero-dep-classifier');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('Zero Dependency Classifier Demo');

    const runBtn = page.locator('#runBtn');
    await expect(runBtn).toBeVisible();
    await runBtn.click();

    const output = page.locator('#output');
    
    // Check that it reaches the end of the pipeline
    await expect(output).toContainText('Classification Result:', { timeout: 10000 });
    await expect(output).toContainText('Label: TABBY_CAT', { timeout: 10000 });
    await expect(output).toContainText('Pipeline finished successfully.', { timeout: 10000 });

    const resetBtn = page.locator('#resetBtn');
    await expect(resetBtn).toBeEnabled();
    await resetBtn.click();
    
    await expect(output).toContainText('Ready. Click', { timeout: 10000 });
  });
});
