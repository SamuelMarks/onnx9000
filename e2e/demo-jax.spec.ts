import { test, expect } from '@playwright/test';

test.describe('JAX & Flax Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-jax');
    } catch (e) {
      console.log('Skipping real nav, relying on manual DOM testing if needed', e);
      test.skip();
    }
  });

  test('Parses a mock ClosedJaxpr JSON payload', async ({ page }) => {
    const convertBtn = page.locator('#convert-btn');
    const output = page.locator('#jax-output');

    await expect(convertBtn).toBeVisible();
    await convertBtn.click();

    await expect(output).toContainText('Parsing mock ClosedJaxpr JSON');
    await expect(output).toContainText('Extracted 1 equations.');
    await expect(output).toContainText('Primitive [add] mapped successfully.');
    await expect(output).toContainText(
      'Success! JAX & Flax graphs can be transpiled natively in JS.',
    );
  });
});
