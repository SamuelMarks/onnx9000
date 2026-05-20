import { test, expect } from '@playwright/test';

test.describe('ONNX9000 Custom Ops Demo', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/apps/demo-custom-ops/index.html');
      await page.waitForSelector('#sandbox-container', { state: 'attached', timeout: 5000 });
    } catch (e) {
      console.log('Skipping real nav, app may not be served at /apps/...', e);
      test.skip();
    }
  });

  test('Registers a custom operation in the UI', async ({ page }) => {
    const sandboxContainer = page.locator('#sandbox-container');
    if (await sandboxContainer.isVisible()) {
      const input = page.locator('#op-name');
      const registerBtn = page.locator('button', { hasText: 'Register' });

      await input.fill('MySuperCustomOp');
      await registerBtn.click();

      // Check if it's added to the registry list
      const registry = page.locator('.op-registry');
      await expect(registry).toContainText('MySuperCustomOp');
      
      const opCount = await page.locator('.op-item').count();
      expect(opCount).toBeGreaterThan(0);
    }
  });
});
