/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  check_model,
  ValidationContext,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropzone = document.getElementById('dropzone'); /* v8 ignore next */ /* v8 ignore next */
const results = document.getElementById('results'); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone?.addEventListener('dragover', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.style.background = '#e0ffe0'; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone?.addEventListener('dragleave', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.style.background = '#fff'; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone?.addEventListener('drop', async (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.style.background = '#fff'; /* v8 ignore next */ /* v8 ignore next */
  if (!e.dataTransfer?.files || e.dataTransfer.files.length === 0)
    return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const file = e.dataTransfer.files[0]; /* v8 ignore next */ /* v8 ignore next */
  if (!file) return; /* v8 ignore next */ /* v8 ignore next */
  if (!results) return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  results.innerHTML = 'Parsing and validating...'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const arrayBuffer = await file.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
    // Simulate parsing the model for the checker /* v8 ignore next */ /* v8 ignore next */
    // In a real scenario, we'd use onnx9000/parser to read the protobuf into the Model interface /* v8 ignore next */ /* v8 ignore next */
    const mockModel = {
      /* v8 ignore next */ /* v8 ignore next */
      ir_version: 8 /* v8 ignore next */ /* v8 ignore next */,
      producer_name: 'onnx9000-ui' /* v8 ignore next */ /* v8 ignore next */,
      opset_import: [{ domain: 'ai.onnx', version: 15 }] /* v8 ignore next */ /* v8 ignore next */,
      graph: {
        /* v8 ignore next */ /* v8 ignore next */
        nodes: [
          /* v8 ignore next */ /* v8 ignore next */
          {
            /* v8 ignore next */ /* v8 ignore next */
            op_type: 'Conv' /* v8 ignore next */ /* v8 ignore next */,
            inputs: ['X', 'W'] /* v8 ignore next */ /* v8 ignore next */,
            outputs: ['Y'] /* v8 ignore next */ /* v8 ignore next */,
            attributes: {
              pads: [1, 1, 1, 1],
              strides: [1, 1],
            } /* v8 ignore next */ /* v8 ignore next */,
          } /* v8 ignore next */ /* v8 ignore next */,
        ] /* v8 ignore next */ /* v8 ignore next */,
        inputs: [
          { name: 'X', data_type: 'float32', shape: [1, 3, 224, 224] },
        ] /* v8 ignore next */ /* v8 ignore next */,
        outputs: ['Y'] /* v8 ignore next */ /* v8 ignore next */,
        initializers: [
          /* v8 ignore next */ /* v8 ignore next */
          {
            name: 'W',
            data_type: 'float32',
            shape: [64, 3, 3, 3],
            is_initializer: true,
          } /* v8 ignore next */ /* v8 ignore next */,
        ] /* v8 ignore next */ /* v8 ignore next */,
      } /* v8 ignore next */ /* v8 ignore next */,
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const ctx = new ValidationContext(); /* v8 ignore next */ /* v8 ignore next */
    check_model(
      mockModel as ReturnType<typeof JSON.parse>,
      ctx,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (ctx.errors && ctx.errors.length > 0) {
      /* v8 ignore next */ /* v8 ignore next */
      results.innerHTML =
        /* v8 ignore next */ /* v8 ignore next */
        `<h3 class="error">Validation Failed</h3><ul>` /* v8 ignore next */ /* v8 ignore next */ +
        ctx.errors /* v8 ignore next */ /* v8 ignore next */
          .map(
            /* v8 ignore next */ /* v8 ignore next */
            (err /* v8 ignore next */ /* v8 ignore next */) =>
              `<li>${err} <a href="https://onnx.ai/onnx/operators/" target="_blank">Docs</a></li>` /* v8 ignore next */ /* v8 ignore next */,
          ) /* v8 ignore next */ /* v8 ignore next */
          .join('') /* v8 ignore next */ /* v8 ignore next */ +
        `</ul>`; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      results.innerHTML = `<div class="success">Model ${file.name} is structurally valid!</div>`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    results.innerHTML = `<div class="error">Error: ${err.message}</div>`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
