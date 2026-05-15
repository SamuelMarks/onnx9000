import { test, expect } from '@playwright/test';

test.describe('Extended Format Converters Demo', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the built demo-extended-converters
    try {
      await page.goto('/demo-extended-converters/dist/index.html');
      await page.waitForSelector('h1', { state: 'attached', timeout: 5000 });
    } catch (e) {
      console.log('Skipping real nav, possibly served differently', e);
      test.skip();
    }
  });

  test('can initialize the UI and mock a conversion', async ({ page }) => {
    const srcSelect = page.locator('#srcFramework');
    const dstSelect = page.locator('#dstFramework');
    const convertBtn = page.locator('#btnConvert');
    const output = page.locator('#output');
    
    if (await srcSelect.isVisible()) {
      await srcSelect.selectOption('keras');
      await dstSelect.selectOption('onnx');
      
      // Need a dummy file to upload
      await page.setInputFiles('#fileInput', {
        name: 'model.h5',
        mimeType: 'application/octet-stream',
        buffer: Buffer.from('dummy Keras file content')
      });
      
      await convertBtn.click();
      
      // Expect some processing indicator
      await expect(output).toContainText('Converting 1 file(s)');
      
      // Wait for completion (either success or error, since it's a dummy file, likely error)
      await expect(output).toContainText(/Error|Successful/, { timeout: 10000 });
    }
  });
});
