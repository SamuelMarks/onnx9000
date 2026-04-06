import { test, expect } from '@playwright/test';

test.describe('MMDNN Universal Converter Demo E2E', () => {
  test('Page loads and elements are visible', async ({ page }) => {
    await page.goto('/mmdnn');

    await expect(page.locator('h1')).toHaveText('MMDNN Universal Converter');
    await expect(page.locator('#src-framework')).toBeVisible();
    await expect(page.locator('#dst-framework')).toBeVisible();
    await expect(page.locator('#drop-zone')).toBeVisible();
    await expect(page.locator('#btn-convert')).toBeVisible();
  });

  test('Framework selection updates requirements text', async ({ page }) => {
    await page.goto('/mmdnn');

    const srcDropdown = page.locator('#src-framework');
    await srcDropdown.selectOption('mxnet');
    
    await expect(page.locator('#drop-hint')).toHaveText('Requires: -symbol.json and .params');

    await srcDropdown.selectOption('ncnn');
    await expect(page.locator('#drop-hint')).toHaveText('Requires: .param and .bin');
  });

  test('Uploading incorrect files disables convert button', async ({ page }) => {
    await page.goto('/mmdnn');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.locator('#drop-zone').click();
    const fileChooser = await fileChooserPromise;

    // Use a fake text file
    await fileChooser.setFiles([
        {
            name: 'invalid.txt',
            mimeType: 'text/plain',
            buffer: Buffer.from('this is not an onnx file')
        }
    ]);

    await expect(page.locator('.files-list .file-item')).toBeVisible();
    await expect(page.locator('#btn-convert')).toBeDisabled();
  });
});