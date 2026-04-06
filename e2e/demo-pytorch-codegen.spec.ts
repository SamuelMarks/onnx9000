import { test, expect } from '@playwright/test';

test.describe('PyTorch Codegen Demo E2E', () => {
  test('Page loads and elements are visible', async ({ page }) => {
    await page.goto('/pytorch-codegen');

    await expect(page.locator('h1')).toHaveText('ONNX9000 PyTorch Generator');
    await expect(page.locator('#drop-zone')).toBeVisible();
    await expect(page.locator('#code')).toBeVisible();
  });

  test('Validates file extension', async ({ page }) => {
    await page.goto('/pytorch-codegen');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.locator('#drop-zone').click();
    const fileChooser = await fileChooserPromise;

    // Use a fake text file
    await fileChooser.setFiles({
      name: 'invalid.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('this is not an onnx file')
    });

    const codeVal = await page.locator('#code').inputValue();
    expect(codeVal).toContain('Please provide a valid .onnx file.');
  });
});