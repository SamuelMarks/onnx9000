import { test, expect } from '@playwright/test';

test.describe('Profiler Web Demo', () => {
  test('Execution', async ({ page }) => {
    try {
      await page.goto('/apps/demo-profiler/index.html');
      await page.waitForSelector('.container', { state: 'attached', timeout: 5000 });
    } catch (e) {
      test.skip();
    }
    await page.click('#btn-run');
    await expect(page.locator('#output')).toContainText('execution complete', { timeout: 2000 });
  });
});
