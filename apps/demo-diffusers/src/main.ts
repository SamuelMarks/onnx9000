/**
 * @fileoverview Diffusers Pipeline Demo UI.
 */
import { DiffusionPipeline } from '@onnx9000/diffusers';

/**
 * Initializes the Diffusers demo.
 */
export function initDiffusersDemo(): void {
  const runBtn = document.getElementById('run-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !out) return;

  runBtn.addEventListener('click', () => {
    out.innerText = 'Initializing Pipeline...';
    const _pipe = new DiffusionPipeline();
    out.innerText += `
Pipeline initialized. Generating mock tensor...`;
    out.innerText += `
Image tensor generated successfully [1, 3, 512, 512]`;
  });
}

initDiffusersDemo();
