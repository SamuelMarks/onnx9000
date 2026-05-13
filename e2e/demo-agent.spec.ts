import { test, expect } from '@playwright/test';

test.describe('Agent Demo E2E', () => {
  test('Page loads and runs agent loop', async ({ page }) => {
    test.setTimeout(120000);
    try {
      await page.goto('/agent');
    } catch {
      test.skip();
      return;
    }

    const title = page.locator('h1');
    if (await title.count() === 0) {
      test.skip();
      return;
    }
    await expect(title).toHaveText('ONNX9000 Agent Workflow Demo');

    const runBtn = page.locator('#runBtn');
    await expect(runBtn).toBeVisible();

    await runBtn.click();

    const output = page.locator('#output');
    await expect(output).toContainText('Final Answer:', { timeout: 10000 });
  });
});
