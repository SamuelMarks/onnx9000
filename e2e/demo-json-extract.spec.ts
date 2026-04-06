import { test, expect } from '@playwright/test';

test.describe('JSON Extractor Demo E2E', () => {
  test('Page loads and elements are visible', async ({ page }) => {
    // Go to the served route
    await page.goto('/json-extract');

    await expect(page.locator('h1')).toHaveText('ONNX to JSON Extractor');
    await expect(page.locator('#drop-zone')).toBeVisible();
    await expect(page.locator('#browse-btn')).toBeVisible();
  });

  test('Validates file extension', async ({ page }) => {
    await page.goto('/json-extract');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.locator('#browse-btn').click();
    const fileChooser = await fileChooserPromise;

    // Use a fake text file
    await fileChooser.setFiles({
      name: 'invalid.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('this is not an onnx file')
    });

    await expect(page.locator('#error-box')).toBeVisible();
    await expect(page.locator('#error-box')).toHaveText('Please provide a valid .onnx file.');
  });
});