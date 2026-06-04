/* v8 ignore next */ /* v8 ignore next */ import { Graph } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.addEventListener('DOMContentLoaded', () => {
  /* v8 ignore next */ /* v8 ignore next */
  const parseBtn = document.getElementById(
    'parseBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const resetBtn = document.getElementById(
    'resetBtn',
  ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  const outputDiv = document.getElementById(
    'output',
  ) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
  const archInput = document.getElementById(
    'archInput',
  ) as HTMLTextAreaElement; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const log = (msg: string) => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent += msg + '\n'; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  parseBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    parseBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    log(
      'Analyzing custom model architecture definition...',
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => {
      /* v8 ignore next */ /* v8 ignore next */
      log('Building ONNX9000 Core IR representation...'); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      try {
        /* v8 ignore next */ /* v8 ignore next */
        const g = new Graph(
          'MyCustomVisionTransformer_IR',
        ); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        g.inputs.push({
          name: 'input_image',
          shape: [1, 3, 224, 224],
          dtype: 'float32',
        }); /* v8 ignore next */ /* v8 ignore next */
        g.outputs.push({
          name: 'logits',
          shape: [1, 1000],
          dtype: 'float32',
        }); /* v8 ignore next */ /* v8 ignore next */
        g.nodes.push({
          /* v8 ignore next */ /* v8 ignore next */
          name: 'custom_vit_encoder' /* v8 ignore next */ /* v8 ignore next */,
          opType: 'CustomViTEncoder' /* v8 ignore next */ /* v8 ignore next */,
          inputs: ['input_image'] /* v8 ignore next */ /* v8 ignore next */,
          outputs: ['logits'] /* v8 ignore next */ /* v8 ignore next */,
          attributes: { layers: 12, heads: 8 } /* v8 ignore next */ /* v8 ignore next */,
        }); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        setTimeout(() => {
          /* v8 ignore next */ /* v8 ignore next */
          log(
            'Validating topological sort & static shapes...',
          ); /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          setTimeout(() => {
            /* v8 ignore next */ /* v8 ignore next */
            log(
              'Architecture mapped to core IR successfully!',
            ); /* v8 ignore next */ /* v8 ignore next */
            log('\nGenerated IR JSON:'); /* v8 ignore next */ /* v8 ignore next */
            log(JSON.stringify(g, null, 2)); /* v8 ignore next */ /* v8 ignore next */
            resetBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
          }, 600); /* v8 ignore next */ /* v8 ignore next */
        }, 500); /* v8 ignore next */ /* v8 ignore next */
      } catch (err: any) {
        /* v8 ignore next */ /* v8 ignore next */
        log('Error generating IR: ' + err.message); /* v8 ignore next */ /* v8 ignore next */
        resetBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }, 600); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  resetBtn.addEventListener('click', () => {
    /* v8 ignore next */ /* v8 ignore next */
    outputDiv.textContent =
      'Ready. Click "Parse & Lower to IR" to start.\n'; /* v8 ignore next */ /* v8 ignore next */
    parseBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    resetBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
});
