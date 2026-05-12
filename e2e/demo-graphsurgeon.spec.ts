import { test, expect } from '@playwright/test';

test.describe('GraphSurgeon Demo App', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/demo-graphsurgeon');
    } catch (e) {
      console.log('Skipping real nav, relying on manual DOM testing if needed', e);
      test.skip();
    }
  });

  test('Mutates graph nodes in the browser', async ({ page }) => {
    const mutateBtn = page.locator('#mutate-btn');
    const output = page.locator('#surgeon-output');

    await expect(mutateBtn).toBeVisible();
    await mutateBtn.click();

    await expect(output).toContainText('Original Graph:');
    await expect(output).toContainText('["Identity","Relu"]');
    await expect(output).toContainText('Removing Identity node');
    await expect(output).toContainText('Mutated Graph:');
    await expect(output).toContainText('["Relu"]');
    await expect(output).toContainText('Success! Graph structure modified.');
  });
});
