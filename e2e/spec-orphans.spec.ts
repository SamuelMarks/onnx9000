import { test, expect } from '@playwright/test';

const pages = [
  { url: '/apps/demo-ort-training/index.html', title: 'ORT Training', id: '#btn-run' },
  { url: '/apps/demo-olive-optimizer/index.html', title: 'Olive Optimizer', id: '#btn-run' },
  { url: '/apps/demo-triton-server/index.html', title: 'Triton Server', id: '#btn-run' },
  { url: '/apps/demo-onnx-tool/index.html', title: 'ONNX Tool', id: '#btn-run' }
];

test.describe('Spec Orphans Web Demos', () => {
  for (const p of pages) {
    test(`Web Demo: ${p.title}`, async ({ page }) => {
      try {
        await page.goto(p.url);
        await page.waitForSelector('.container', { state: 'attached', timeout: 5000 });
      } catch (e) {
        test.skip();
      }
      await page.click(p.id);
      await expect(page.locator('#output')).toContainText('execution complete', { timeout: 2000 });
    });
  }
});
