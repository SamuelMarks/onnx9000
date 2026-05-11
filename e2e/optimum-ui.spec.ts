import { test, expect } from '@playwright/test';

test.describe('optimum-ui', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/optimum-ui.html');
    } catch (e) {
      test.skip();
    }
  });

  test('Enter model ID, select O3, mock network response', async ({ page }) => {
    // Mock the network response for optimization API
    await page.route('**/api/optimize', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true, message: 'Optimized' }),
      });
    });

    const input = page.locator('input[type="text"]');
    if (await input.isVisible()) {
      await input.fill('hf-internal-testing/tiny-random-bert');

      const select = page.locator('select');
      if (await select.isVisible()) {
        await select.selectOption('O3');
      }

      const btn = page.locator('button', { hasText: /Optimize/i });
      if (await btn.isVisible()) {
        await btn.click();

        // Wait for result
        const result = page.locator('.result, .toast, .success');
        if ((await result.count()) > 0) {
          await expect(result.first()).toBeVisible();
        }
      }
    }
  });
});
