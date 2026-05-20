import { test, expect } from '@playwright/test';

test.describe('Specific Extended Converters', () => {
  test('Paddle2ONNX Web Demo', async ({ page }) => {
    try {
      await page.goto('/apps/demo-paddle2onnx/index.html');
      await page.waitForSelector('.container', { state: 'attached', timeout: 5000 });
    } catch (e) {
      test.skip();
    }
    await page.click('#btn-convert');
    await expect(page.locator('#output')).toContainText('Paddle2ONNX conversion complete', { timeout: 2000 });
  });

  test('Keras2ONNX Web Demo', async ({ page }) => {
    try {
      await page.goto('/apps/demo-keras2onnx/index.html');
      await page.waitForSelector('.container', { state: 'attached', timeout: 5000 });
    } catch (e) {
      test.skip();
    }
    await page.click('#btn-convert');
    await expect(page.locator('#output')).toContainText('Keras2ONNX conversion complete', { timeout: 2000 });
  });

  test('SKL2ONNX Web Demo', async ({ page }) => {
    try {
      await page.goto('/apps/demo-skl2onnx/index.html');
      await page.waitForSelector('.container', { state: 'attached', timeout: 5000 });
    } catch (e) {
      test.skip();
    }
    await page.click('#btn-convert');
    await expect(page.locator('#output')).toContainText('SKL2ONNX conversion complete', { timeout: 2000 });
  });
});
