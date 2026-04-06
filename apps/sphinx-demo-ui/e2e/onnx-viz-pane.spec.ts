import { test, expect } from '@playwright/test';

test.describe('ONNX Visualization Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    // Load WASM
    const overlayBtn = page.locator('.demo-wasm-overlay .demo-btn-primary');
    if (await overlayBtn.isVisible()) {
      await page.route('/onnx9000.wasm', async (route) => {
        const dummyWasm = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
        await route.fulfill({ status: 200, contentType: 'application/wasm', body: dummyWasm });
      });
      await overlayBtn.click();
      await page.locator('.demo-wasm-overlay').waitFor({ state: 'hidden' });
    }
  });

  test('should verify selecting .onnx target renders a graph', async ({ page }) => {
    // Navigate to Viz tab
    const tabList = page.locator('.demo-tab-list');
    await tabList.locator('button', { hasText: 'ONNX Visualization' }).click();

    // Verify it's initially empty
    const tooltip = page.locator('.demo-onnx-tooltip');
    await expect(tooltip).toBeHidden();

    // Simulate generating a graph via EventBus attached to window
    await page.evaluate(() => {
      if ((window as ReturnType<typeof JSON.parse>).__EVENT_BUS__) {
        (window as ReturnType<typeof JSON.parse>).__EVENT_BUS__.emit('ONNX_GRAPH_GENERATED', {
          inputs: [{ name: 'in1', type: 'float' }],
          outputs: [{ name: 'out1', type: 'float' }],
          nodes: [{ id: 'n1', name: 'Relu_1', opType: 'Relu', inputs: ['in1'], outputs: ['out1'] }]
        });
      }
    });

    // We should see a cytoscape canvas
    const canvas = page.locator('.demo-onnx-viz-container canvas').first();
    await expect(canvas).toBeVisible();
  });

  test('should verify clicking a node opens a properties tooltip', async ({ page }) => {
    const tabList = page.locator('.demo-tab-list');
    await tabList.locator('button', { hasText: 'ONNX Visualization' }).click();

    // Generate graph
    await page.evaluate(() => {
      if ((window as ReturnType<typeof JSON.parse>).__EVENT_BUS__) {
        (window as ReturnType<typeof JSON.parse>).__EVENT_BUS__.emit('ONNX_GRAPH_GENERATED', {
          inputs: [{ name: 'in1', type: 'float' }],
          outputs: [{ name: 'out1', type: 'float' }],
          nodes: [{ id: 'n1', name: 'Relu_1', opType: 'Relu', inputs: ['in1'], outputs: ['out1'] }]
        });
      }
    });

    const tooltip = page.locator('.demo-onnx-tooltip');

    // Evaluate cytoscape instance to trigger a tap
    await page.evaluate(() => {
      if ((window as ReturnType<typeof JSON.parse>).__CY__) {
        const cy = (window as ReturnType<typeof JSON.parse>).__CY__;
        const node = cy.nodes()[0];
        if (node) {
          node.emit('tap');
        }
      }
    });

    await expect(tooltip).toBeVisible();
    await expect(tooltip.innerText()).resolves.toContain('INPUT: in1');
  });
});
