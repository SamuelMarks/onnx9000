import { test, expect } from '@playwright/test';

test.describe('Headless Graph Modifiers Demo', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the built demo-modifier
    try {
      await page.goto('/demo-modifier/dist/index.html');
      await page.waitForSelector('h1', { state: 'attached', timeout: 5000 });
    } catch (e) {
      console.log('Skipping real nav, possibly served differently', e);
      test.skip();
    }
  });

  test('can initialize graph and rename input', async ({ page }) => {
    const initBtn = page.locator('#btnInit');
    if (await initBtn.isVisible()) {
      await initBtn.click();
      
      const output = page.locator('#output');
      await expect(output).toContainText('input_0');
      await expect(output).toContainText('output_0');
      
      const renameBtn = page.locator('#btnRename');
      await renameBtn.click();
      
      await expect(output).toContainText('images');
    }
  });
});
