import { test, expect } from '@playwright/test';

test.describe('End-to-end format conversions', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(msg.text()));
    // Intercept WASM download to skip wait time
    await page.route('/onnx9000.wasm', async (route) => {
      const dummyWasm = Buffer.from([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00]);
      await route.fulfill({
        status: 200,
        contentType: 'application/wasm',
        body: dummyWasm,
        headers: { 'Content-Length': dummyWasm.length.toString() }
      });
    });

    await page.goto('/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();

    // Load the WASM demo to remove the overlay
    const loadButton = page.locator('.demo-btn-primary');
    await loadButton.click();

    // Wait for the overlay to disappear
    const overlay = page.locator('.demo-wasm-overlay');
    await expect(overlay).toBeHidden({ timeout: 15000 });
  });

  const runLHSConversion = async (page: ReturnType<typeof JSON.parse>, sourceFramework: string) => {
    const sourceDropdown = page.locator('.demo-pane-lhs .demo-dropdown').first();
    await sourceDropdown.click();
    await page
      .locator('.demo-pane-lhs .demo-dropdown-listbox')
      .getByText(sourceFramework, { exact: true })
      .click();
    const lhsRunBtn = page.locator('.demo-btn-run-conversion');
    await lhsRunBtn.click();
    const rhsRunBtn = page.locator('.demo-btn-run-inference');
    await expect(rhsRunBtn).toBeEnabled({ timeout: 30000 });
  };

  const testPermutation = async (
    page: ReturnType<typeof JSON.parse>,
    target: string,
    expectedSnippet: string
  ) => {
    const targetDropdown = page.locator('.demo-pane-rhs .demo-dropdown').first();
    await targetDropdown.click();
    await page
      .locator('.demo-pane-rhs .demo-dropdown-listbox')
      .getByText(target, { exact: true })
      .click();
    const editorContent = page.locator('.demo-pane-rhs .monaco-editor');
    await expect(editorContent).toContainText(expectedSnippet, { timeout: 10000 });
  };

  test('onnxscript / spox -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'onnxscript / spox');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('onnxscript / spox -> C++', async ({ page }) => {
    await runLHSConversion(page, 'onnxscript / spox');
    await testPermutation(page, 'C++', 'namespace model_');
  });

  test('Scikit-Learn -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'Scikit-Learn');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('LightGBM -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'LightGBM');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('XGBoost -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'XGBoost');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('CatBoost -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'CatBoost');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('SparkML -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'SparkML');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('PaddlePaddle -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'PaddlePaddle');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('PaddlePaddle -> C++', async ({ page }) => {
    await runLHSConversion(page, 'PaddlePaddle');
    await testPermutation(page, 'C++', 'namespace model_');
  });

  test('TensorFlow -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'TensorFlow');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('TensorFlow -> C++', async ({ page }) => {
    await runLHSConversion(page, 'TensorFlow');
    await testPermutation(page, 'C++', 'namespace model_');
  });

  test('Keras -> ONNX', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, '.onnx', 'ir_version');
  });

  test('Keras -> C++', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'C++', 'namespace model_');
  });

  test('Keras -> C', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'C', 'void model_run');
  });

  test('Keras -> MLIR', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'MLIR', 'module');
  });

  test('Keras -> PyTorch', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'PyTorch', 'import torch');
  });

  test('Keras -> TensorFlow', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'TensorFlow', 'import tensorflow as tf');
  });

  test('Keras -> MXNet', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'MXNet', 'import mxnet as mx');
  });

  test('Keras -> Caffe', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'Caffe', 'name: ');
  });

  test('Keras -> CNTK', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'CNTK', 'import cntk as C');
  });

  test('Keras -> Optimize (Olive)', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'Optimize (Olive)', 'ir_version');
  });

  test('Keras -> Simplify (onnx-simplifier)', async ({ page }) => {
    await runLHSConversion(page, 'Keras');
    await testPermutation(page, 'Simplify (onnx-simplifier)', 'ir_version');
  });
});
