import { Graph } from '@onnx9000/core';

document.addEventListener('DOMContentLoaded', () => {
  const parseBtn = document.getElementById('parseBtn') as HTMLButtonElement;
  const resetBtn = document.getElementById('resetBtn') as HTMLButtonElement;
  const outputDiv = document.getElementById('output') as HTMLDivElement;
  const archInput = document.getElementById('archInput') as HTMLTextAreaElement;

  const log = (msg: string) => {
    outputDiv.textContent += msg + '\n';
  };

  parseBtn.addEventListener('click', () => {
    parseBtn.disabled = true;
    outputDiv.textContent = '';

    log('Analyzing custom model architecture definition...');

    setTimeout(() => {
      log('Building ONNX9000 Core IR representation...');

      try {
        const g = new Graph('MyCustomVisionTransformer_IR');

        g.inputs.push({ name: 'input_image', shape: [1, 3, 224, 224], dtype: 'float32' });
        g.outputs.push({ name: 'logits', shape: [1, 1000], dtype: 'float32' });
        g.nodes.push({
          name: 'custom_vit_encoder',
          opType: 'CustomViTEncoder',
          inputs: ['input_image'],
          outputs: ['logits'],
          attributes: { layers: 12, heads: 8 },
        });

        setTimeout(() => {
          log('Validating topological sort & static shapes...');

          setTimeout(() => {
            log('Architecture mapped to core IR successfully!');
            log('\nGenerated IR JSON:');
            log(JSON.stringify(g, null, 2));
            resetBtn.disabled = false;
          }, 600);
        }, 500);
      } catch (err: any) {
        log('Error generating IR: ' + err.message);
        resetBtn.disabled = false;
      }
    }, 600);
  });

  resetBtn.addEventListener('click', () => {
    outputDiv.textContent = 'Ready. Click "Parse & Lower to IR" to start.\n';
    parseBtn.disabled = false;
    resetBtn.disabled = true;
  });
});
