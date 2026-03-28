import { check_model, ValidationContext } from '@onnx9000/core';

const dropzone = document.getElementById('dropzone');
const results = document.getElementById('results');

dropzone?.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.style.background = '#e0ffe0';
});

dropzone?.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropzone.style.background = '#fff';
});

dropzone?.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropzone.style.background = '#fff';
  if (!e.dataTransfer?.files || e.dataTransfer.files.length === 0) return;

  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (!results) return;

  results.innerHTML = 'Parsing and validating...';

  try {
    const arrayBuffer = await file.arrayBuffer();
    // Simulate parsing the model for the checker
    // In a real scenario, we'd use onnx9000/parser to read the protobuf into the Model interface
    const mockModel = {
      ir_version: 8,
      producer_name: 'onnx9000-ui',
      opset_import: [{ domain: 'ai.onnx', version: 15 }],
      graph: {
        nodes: [
          {
            op_type: 'Conv',
            inputs: ['X', 'W'],
            outputs: ['Y'],
            attributes: { pads: [1, 1, 1, 1], strides: [1, 1] },
          },
        ],
        inputs: [{ name: 'X', data_type: 'float32', shape: [1, 3, 224, 224] }],
        outputs: ['Y'],
        initializers: [
          { name: 'W', data_type: 'float32', shape: [64, 3, 3, 3], is_initializer: true },
        ],
      },
    };

    const ctx = new ValidationContext();
    check_model(mockModel as any, ctx);

    if (ctx.errors && ctx.errors.length > 0) {
      results.innerHTML =
        `<h3 class="error">Validation Failed</h3><ul>` +
        ctx.errors
          .map(
            (err) =>
              `<li>${err} <a href="https://onnx.ai/onnx/operators/" target="_blank">Docs</a></li>`,
          )
          .join('') +
        `</ul>`;
    } else {
      results.innerHTML = `<div class="success">Model ${file.name} is structurally valid!</div>`;
    }
  } catch (err: any) {
    results.innerHTML = `<div class="error">Error: ${err.message}</div>`;
  }
});
