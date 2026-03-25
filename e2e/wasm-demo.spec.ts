import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import * as os from 'os';

test.describe('WASM Sphinx Demo E2E', () => {
  test('Demo workflow: load, Keras LHS, C RHS default, compile with gcc', async ({ page }) => {
    // 0. Show whole page
    await page.goto('/');

    // 0. "click to load" and a progress bar
    const startBtn = page.locator('.demo-wasm-modal button', { hasText: 'Start Demo' });
    await expect(startBtn).toBeVisible({ timeout: 5000 });
    await startBtn.click();

    // Wait for the overlay to disappear
    await expect(page.locator('.demo-wasm-overlay')).toBeHidden({ timeout: 15000 });

    // Wait for UI to settle
    await page.waitForTimeout(1000);

    // 1. First thing to show in demo is CNN MNIST in Keras.
    // Check LHS Framework Dropdown
    const lhsDropdown = page
      .locator('.demo-pane-lhs .demo-pane-header .demo-dropdown-button')
      .first();
    await expect(lhsDropdown).toHaveText('Keras');

    // Make sure the code editor contains Keras code
    // The Monaco editor content is deeply nested. We can evaluate to check it.
    await page.waitForFunction(() => !!(window as any).monaco && !!(window as any).monaco.editor);
    const lhsContent = await page.evaluate(() => {
      return (window as any).monaco.editor
        .getModels()
        .find((m: any) => m.uri.path.includes('train.py'))
        ?.getValue();
    });
    expect(lhsContent).toContain('models.Sequential');

    // 2. Default output be C. Show the .c code.
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-pane-header .demo-dropdown-button');
    await expect(rhsDropdown).toHaveText('C');

    // Click the "Run Conversion" button to generate C output
    const runBtn = page.locator('.demo-pane-lhs button', { hasText: 'Run Conversion' });
    await runBtn.click();

    // Wait for the button to reset text back to "Run Conversion" meaning it finished
    await expect(runBtn).toHaveText('Run Conversion', { timeout: 15000 });

    // Ensure RHS Editor shows C code (model.c)
    await page.waitForTimeout(2000); // Wait for conversion ONNX -> C

    const modelCContent = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelC = models.find((m: any) => m.uri.path.includes('model.c'));
      return modelC ? modelC.getValue() : '';
    });
    expect(modelCContent).toContain('void model_run');

    // The directory is expanded by default, click model.h
    await page
      .locator('.demo-pane-rhs .demo-file-tree-label', { hasText: 'model.h' })
      .click({ force: true });
    await page.waitForTimeout(500);

    const modelHContent = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelH = models.find((m: any) => m.uri.path.includes('model.h'));
      return modelH ? modelH.getValue() : '';
    });
    expect(modelHContent).toContain('void model_run');

    // 3. Confirm that the C output is actually runnable with `gcc` and provides expected output.
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnx9000-e2e-'));
    const modelCPath = path.join(tmpDir, 'model.c');
    const modelHPath = path.join(tmpDir, 'model.h');
    const mainPath = path.join(tmpDir, 'main.c');
    const outPath = path.join(tmpDir, 'a.out');

    fs.writeFileSync(modelCPath, modelCContent as string);
    fs.writeFileSync(modelHPath, modelHContent as string);

    // Create a main.c to test the generated function
    const mainC = `
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main() {
    float input[256] = {0};
    float output[256] = {0};
    input[0] = 1.0f; // some dummy input

    model_run(input, output);

    printf("SUCCESS: %f\\n", output[0]);
    return 0;
}
`;
    fs.writeFileSync(mainPath, mainC);

    try {
      execSync(`gcc -std=c99 -o ${outPath} ${mainPath} ${modelCPath} -lm`);
      const result = execSync(outPath).toString();
      expect(result).toContain('SUCCESS');
    } catch (e: any) {
      console.error(e.stdout ? e.stdout.toString() : e);
      console.error(e.stderr ? e.stderr.toString() : '');
      throw e;
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('Output changes when Keras input changes', async ({ page }) => {
    await page.goto('/');

    const startBtn = page.locator('.demo-wasm-modal button', { hasText: 'Start Demo' });
    await expect(startBtn).toBeVisible({ timeout: 5000 });
    await startBtn.click();
    await expect(page.locator('.demo-wasm-overlay')).toBeHidden({ timeout: 15000 });

    await page.waitForTimeout(2000);
    await page.waitForFunction(() => !!(window as any).monaco && !!(window as any).monaco.editor);

    // Default should be C
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-pane-header .demo-dropdown-button');
    await expect(rhsDropdown).toHaveText('C');

    const runBtn = page.locator('.demo-pane-lhs button', { hasText: 'Run Conversion' });
    await runBtn.click();
    await expect(runBtn).toHaveText('Run Conversion', { timeout: 15000 });
    await page.waitForTimeout(2000);

    // Get original C code
    const originalC = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelC = models.find((m: any) => m.uri.path.includes('model.c'));
      return modelC ? modelC.getValue() : '';
    });

    expect(originalC).toBeTruthy();

    // Change Keras code in LHS: e.g., change units from 128 to 256
    await page.evaluate(() => {
      const kerasModel = (window as any).monaco.editor
        .getModels()
        .find((m: any) => m.uri.path.includes('train.py'));
      const val = kerasModel.getValue();
      kerasModel.setValue(val.replace('128', '256')); // Assuming 128 is in the keras input
    });

    // Click the "Run Conversion" button
    await runBtn.click();

    // Wait for the button to reset text back to "Run Conversion" meaning it finished
    await expect(runBtn).toHaveText('Run Conversion', { timeout: 10000 });

    // Wait for RHS re-render
    await page.waitForTimeout(2000);

    // Get new C code
    const newC = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelC = models.find((m: any) => m.uri.path.includes('model.c'));
      return modelC ? modelC.getValue() : '';
    });

    expect(newC).not.toEqual(originalC);
  });

  test('ONNX Script generation and execution', async ({ page }) => {
    await page.goto('/');

    const startBtn = page.locator('.demo-wasm-modal button', { hasText: 'Start Demo' });
    await expect(startBtn).toBeVisible({ timeout: 5000 });
    await startBtn.click();
    await expect(page.locator('.demo-wasm-overlay')).toBeHidden({ timeout: 15000 });

    await page.waitForTimeout(2000);
    await page.waitForFunction(() => !!(window as any).monaco && !!(window as any).monaco.editor);

    // Select ONNX Script from RHS Dropdown
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    await rhsDropdown.locator('.demo-dropdown-button').click({ force: true });
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: 'ONNX Script' })
      .click({ force: true });
    await expect(rhsDropdown.locator('.demo-dropdown-button')).toHaveText('ONNX Script');

    const runBtn = page.locator('.demo-pane-lhs button', { hasText: 'Run Conversion' });
    await runBtn.click();
    await expect(runBtn).toHaveText('Run Conversion', { timeout: 15000 });
    await page.waitForTimeout(2000);

    // Get generated ONNX Script Python code
    const pyCode = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelPy = models.find((m: any) => m.uri.path.includes('model.py'));
      return modelPy ? modelPy.getValue() : '';
    });

    expect(pyCode).toContain('@onnxscript.script()');
    expect(pyCode).toContain('if __name__ == "__main__":');

    // Confirm that the ONNX script is actually runnable with `python`
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnx9000-e2e-onnxscript-'));
    const mainPath = path.join(tmpDir, 'model.py');

    fs.writeFileSync(mainPath, pyCode as string);

    try {
      // Execute the python script using uv to ensure onnxscript is available
      const result = execSync(`uv run --with onnxscript --with onnx python ${mainPath}`).toString();
      expect(result).toContain('SUCCESS: ONNXScript model generated correctly');
    } catch (e: any) {
      console.error(e.stdout ? e.stdout.toString() : e);
      console.error(e.stderr ? e.stderr.toString() : '');
      throw e;
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('PyTorch generation and execution', async ({ page }) => {
    await page.goto('/');

    const startBtn = page.locator('.demo-wasm-modal button', { hasText: 'Start Demo' });
    await expect(startBtn).toBeVisible({ timeout: 5000 });
    await startBtn.click();
    await expect(page.locator('.demo-wasm-overlay')).toBeHidden({ timeout: 15000 });

    await page.waitForTimeout(2000);
    await page.waitForFunction(() => !!(window as any).monaco && !!(window as any).monaco.editor);

    // Select PyTorch from RHS Dropdown
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    await rhsDropdown.locator('.demo-dropdown-button').click({ force: true });
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: 'PyTorch' })
      .click({ force: true });
    await expect(rhsDropdown.locator('.demo-dropdown-button')).toHaveText('PyTorch');

    const runBtn = page.locator('.demo-pane-lhs button', { hasText: 'Run Conversion' });
    await runBtn.click();
    await expect(runBtn).toHaveText('Run Conversion', { timeout: 15000 });
    await page.waitForTimeout(2000);

    // Get generated PyTorch Python code
    const pyCode = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelPy = models.find((m: any) => m.uri.path.includes('module.py'));
      return modelPy ? modelPy.getValue() : '';
    });

    expect(pyCode).toContain('import torch');
    expect(pyCode).toContain('class ONNXModel(nn.Module):');
    expect(pyCode).toContain('if __name__ == "__main__":');

    // Confirm that the PyTorch script is actually runnable with `python`
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnx9000-e2e-pytorch-'));
    const mainPath = path.join(tmpDir, 'module.py');

    fs.writeFileSync(mainPath, pyCode as string);

    try {
      // Execute the python script using uv to ensure torch is available
      const result = execSync(`uv run --with torch python ${mainPath}`).toString();
      expect(result).toContain('SUCCESS: PyTorch model generated correctly');
    } catch (e: any) {
      console.error(e.stdout ? e.stdout.toString() : e);
      console.error(e.stderr ? e.stderr.toString() : '');
      throw e;
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  test('CNTK generation and execution', async ({ page }) => {
    await page.goto('/');

    const startBtn = page.locator('.demo-wasm-modal button', { hasText: 'Start Demo' });
    await expect(startBtn).toBeVisible({ timeout: 5000 });
    await startBtn.click();
    await expect(page.locator('.demo-wasm-overlay')).toBeHidden({ timeout: 15000 });

    await page.waitForTimeout(2000);
    await page.waitForFunction(() => !!(window as any).monaco && !!(window as any).monaco.editor);

    // Select CNTK from RHS Dropdown
    const rhsDropdown = page.locator('.demo-pane-rhs .demo-dropdown');
    await rhsDropdown.locator('.demo-dropdown-button').click({ force: true });
    await rhsDropdown
      .locator('.demo-dropdown-listbox .demo-dropdown-item')
      .filter({ hasText: 'CNTK' })
      .click({ force: true });
    await expect(rhsDropdown.locator('.demo-dropdown-button')).toHaveText('CNTK');

    const runBtn = page.locator('.demo-pane-lhs button', { hasText: 'Run Conversion' });
    await runBtn.click();
    await expect(runBtn).toHaveText('Run Conversion', { timeout: 15000 });
    await page.waitForTimeout(2000);

    // Get generated CNTK Python code
    const pyCode = await page.evaluate(() => {
      const models = (window as any).monaco.editor.getModels();
      const modelPy = models.find(
        (m: any) => m.uri.path.includes('model.py') && m.uri.path.includes('output-cntk'),
      );
      return modelPy ? modelPy.getValue() : '';
    });

    expect(pyCode).toContain('import cntk as C');
    expect(pyCode).toContain('if __name__ == "__main__":');

    // Confirm that the CNTK script is actually runnable with `python`
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnx9000-e2e-cntk-'));
    const mainPath = path.join(tmpDir, 'model.py');

    // We mock CNTK because it is deprecated and won't install on python 3.14.
    const cntkMockCode = `
class MockLayer:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return "mock_out"

class layers:
    Convolution2D = MockLayer
    MaxPooling = MockLayer
    AveragePooling = MockLayer
    GlobalAveragePooling = MockLayer
    Dense = MockLayer

def relu(*args, **kwargs): return "mock_out"
def flatten(*args, **kwargs): return "mock_out"
def plus(*args, **kwargs): return "mock_out"
def softmax(*args, **kwargs): return "mock_out"
`;
    fs.writeFileSync(path.join(tmpDir, 'cntk.py'), cntkMockCode);
    fs.writeFileSync(mainPath, pyCode as string);

    try {
      const result = execSync(`python3 ${mainPath}`, { cwd: tmpDir }).toString();
      expect(result).toContain('SUCCESS: CNTK model generated correctly');
    } catch (e: any) {
      console.error(e.stdout ? e.stdout.toString() : e);
      console.error(e.stderr ? e.stderr.toString() : '');
      throw e;
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});
