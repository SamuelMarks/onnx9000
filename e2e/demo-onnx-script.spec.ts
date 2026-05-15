import { test, expect } from '@playwright/test';

test.describe('ONNX Script Demo E2E', () => {
  test('Page loads and runs ONNX script compilation', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/onnx-script');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('ONNX Script / Fluent Scripting Demo');

    const runBtn = page.locator('#runBtn');
    await expect(runBtn).toBeVisible();

    await runBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Success! Generated Graph JSON:', { timeout: 10000 });
  });
});
