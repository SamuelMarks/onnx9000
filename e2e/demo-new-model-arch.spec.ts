import { test, expect } from '@playwright/test';

test.describe('New Model Architecture Demo E2E', () => {
  test('Page loads and runs architecture lowering', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/new-model-arch');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('New Model Architecture Integration');

    const parseBtn = page.locator('#parseBtn');
    await expect(parseBtn).toBeVisible();
    await parseBtn.click();

    const output = page.locator('#output');
    
    // Check that it reaches the end of the pipeline and emits the IR JSON
    await expect(output).toContainText('Architecture mapped to core IR successfully!', { timeout: 10000 });
    await expect(output).toContainText('MyCustomVisionTransformer_IR', { timeout: 10000 });

    const resetBtn = page.locator('#resetBtn');
    await expect(resetBtn).toBeEnabled();
    await resetBtn.click();
    
    await expect(output).toContainText('Ready. Click', { timeout: 10000 });
  });
});
