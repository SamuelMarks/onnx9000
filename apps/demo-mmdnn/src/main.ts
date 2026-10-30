/**
 * @fileoverview MMDNN Multi-Framework Converter Demo UI.
 */
import { convert, type SourceFramework, type TargetFramework } from '@onnx9000/converters';
import { type Graph, Onnx9000Error, serializeModelProto } from '@onnx9000/core';

/**
 * Initializes the MMDNN demo UI.
 */
export function initMmdnnDemo(): void {
  const srcFrameworkSelect = document.getElementById('src-framework') as HTMLSelectElement | null;
  const dstFrameworkSelect = document.getElementById('dst-framework') as HTMLSelectElement | null;
  const dropZone = document.getElementById('drop-zone') as HTMLDivElement | null;
  const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
  const dropHint = document.getElementById('drop-hint') as HTMLParagraphElement | null;
  const filesList = document.getElementById('files-list') as HTMLDivElement | null;
  const btnConvert = document.getElementById('btn-convert') as HTMLButtonElement | null;
  const btnDownload = document.getElementById('btn-download') as HTMLButtonElement | null;
  const logsContainer = document.getElementById('logs') as HTMLDivElement | null;

  if (
    !srcFrameworkSelect ||
    !dstFrameworkSelect ||
    !dropZone ||
    !fileInput ||
    !dropHint ||
    !filesList ||
    !btnConvert ||
    !btnDownload ||
    !logsContainer
  ) {
    return;
  }

  const validSrcSelect = srcFrameworkSelect;
  const validDstSelect = dstFrameworkSelect;
  const validDropZone = dropZone;
  const validFileInput = fileInput;
  const validDropHint = dropHint;
  const validFilesList = filesList;
  const validBtnConvert = btnConvert;
  const validBtnDownload = btnDownload;
  const validLogsContainer = logsContainer;

  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) {
      throw new Onnx9000Error('WebGL not supported');
    }
  } catch (e) {
    alert('WebGL initialization failed. The previewer may not function correctly.');
    console.error('WebGL init error:', e);
  }

  const currentFiles: File[] = [];
  let finalBlobUrl: string | null = null;
  let finalFileName = 'model.onnx';

  const frameworkRequirements: Record<string, { desc: string; check: (files: File[]) => boolean }> =
    {
      caffe: {
        desc: 'Requires: .prototxt and .caffemodel',
        check: (files) =>
          files.some((f) => f.name.endsWith('.prototxt')) &&
          files.some((f) => f.name.endsWith('.caffemodel')),
      },
      mxnet: {
        desc: 'Requires: -symbol.json and .params',
        check: (files) =>
          files.some((f) => f.name.endsWith('-symbol.json')) &&
          files.some((f) => f.name.endsWith('.params')),
      },
      cntk: {
        desc: 'Requires: .model',
        check: (files) => files.some((f) => f.name.endsWith('.model')),
      },
      darknet: {
        desc: 'Requires: .cfg and .weights',
        check: (files) =>
          files.some((f) => f.name.endsWith('.cfg')) &&
          files.some((f) => f.name.endsWith('.weights')),
      },
      ncnn: {
        desc: 'Requires: .param and .bin',
        check: (files) =>
          files.some((f) => f.name.endsWith('.param')) &&
          files.some((f) => f.name.endsWith('.bin')),
      },
      paddle: {
        desc: 'Requires: __model__ (or .pdmodel) and weights',
        check: (files) => files.some((f) => f.name === '__model__' || f.name.endsWith('.pdmodel')),
      },
      keras: {
        desc: 'Requires: .h5 or .keras',
        check: (files) => files.some((f) => f.name.endsWith('.h5') || f.name.endsWith('.keras')),
      },
      coreml: {
        desc: 'Requires: .mlmodel',
        check: (files) => files.some((f) => f.name.endsWith('.mlmodel')),
      },
      jax: {
        desc: 'Requires: .json (jaxpr text)',
        check: (files) => files.some((f) => f.name.endsWith('.json')),
      },
      flax: {
        desc: 'Requires: .msgpack or .json (Flax nnx state)',
        check: (files) =>
          files.some((f) => f.name.endsWith('.msgpack') || f.name.endsWith('.json')),
      },
      h2o: {
        desc: 'Requires: .zip (MOJO file)',
        check: (files) => files.some((f) => f.name.endsWith('.zip')),
      },
      libsvm: {
        desc: 'Requires: .svm or .txt',
        check: (files) => files.some((f) => f.name.endsWith('.svm') || f.name.endsWith('.txt')),
      },
      sklearn: {
        desc: 'Requires: .joblib or .json',
        check: (files) => files.some((f) => f.name.endsWith('.joblib') || f.name.endsWith('.json')),
      },
      xgboost: {
        desc: 'Requires: .json dump',
        check: (files) => files.some((f) => f.name.endsWith('.json')),
      },
      catboost: {
        desc: 'Requires: .json dump',
        check: (files) => files.some((f) => f.name.endsWith('.json')),
      },
      safetensors: {
        desc: 'Requires: .safetensors',
        check: (files) => files.some((f) => f.name.endsWith('.safetensors')),
      },
      lightgbm: {
        desc: 'Requires: .txt or .json',
        check: (files) => files.some((f) => f.name.endsWith('.txt') || f.name.endsWith('.json')),
      },
      pyspark: {
        desc: 'Requires: .json dump',
        check: (files) => files.some((f) => f.name.endsWith('.json')),
      },
    };

  function updateHint(): void {
    const req = frameworkRequirements[validSrcSelect.value];
    if (req) {
      validDropHint.textContent = req.desc;
    }
    validateFiles();
  }

  function updateFileList(): void {
    validFilesList.innerHTML = '';
    currentFiles.forEach((file, index) => {
      const item = document.createElement('div');
      item.className = 'file-item';
      item.innerHTML = ` 
      <span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span> 
      <button type="button" style="padding: 0.2rem 0.5rem; flex: none;" data-index="${index}">X</button> 
    `;
      const rmBtn = item.querySelector('button');
      rmBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFiles.splice(index, 1);
        updateFileList();
      });
      validFilesList.appendChild(item);
    });
    validateFiles();
  }

  function validateFiles(): void {
    const req = frameworkRequirements[validSrcSelect.value];
    validBtnConvert.disabled = !req?.check(currentFiles);
  }

  function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
    const line = document.createElement('div');
    line.className = `log-line ${type}`;
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    validLogsContainer.appendChild(line);
    validLogsContainer.scrollTop = validLogsContainer.scrollHeight;
  }

  validSrcSelect.addEventListener('change', updateHint);

  validDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    validDropZone.classList.add('dragover');
  });

  validDropZone.addEventListener('dragleave', () => {
    validDropZone.classList.remove('dragover');
  });

  validDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    validDropZone.classList.remove('dragover');
    if (e.dataTransfer?.files) {
      Array.from(e.dataTransfer.files).forEach((f) => {
        currentFiles.push(f);
      });
      updateFileList();
    }
  });

  validDropZone.addEventListener('click', () => {
    validFileInput.click();
  });

  validFileInput.addEventListener('change', (e) => {
    const target = e.target as HTMLInputElement;
    if (target.files) {
      Array.from(target.files).forEach((f) => {
        currentFiles.push(f);
      });
      updateFileList();
    }
    target.value = '';
  });

  const extByTarget: Record<string, string> = {
    pytorch_code: '.py',
    jax_code: '.py',
    flax_nnx_code: '.py',
    tfjs: '_tfjs.json',
    coreml: '.mlmodel',
    ncnn: '.param',
  };

  validBtnConvert.addEventListener('click', async () => {
    validBtnConvert.disabled = true;
    validBtnDownload.classList.add('hidden');
    validLogsContainer.innerHTML = '';

    const src = validSrcSelect.value as SourceFramework;
    const dst = validDstSelect.value as TargetFramework;

    log(`Starting conversion from ${src.toUpperCase()} to ${dst.toUpperCase()}...`, 'info');

    try {
      const result = await convert(src, dst, currentFiles, {
        fusion: true,
        shapeInference: true,
        layoutTracking: true,
        verbose: true,
      });

      let blob: Blob;
      let ext = '.onnx';
      if (dst === 'onnx') {
        const bytes = await serializeModelProto(result as Graph);
        blob = new Blob([bytes.buffer as ArrayBuffer], {
          type: 'application/octet-stream',
        });
      } else {
        ext = extByTarget[dst] ?? '.txt';
        blob = new Blob([typeof result === 'string' ? result : JSON.stringify(result, null, 2)], {
          type: 'text/plain',
        });
      }

      if (finalBlobUrl) {
        URL.revokeObjectURL(finalBlobUrl);
      }
      finalBlobUrl = URL.createObjectURL(blob);
      finalFileName = `converted_model${ext}`;

      log('Conversion complete!', 'success');
      validBtnDownload.textContent = `Download ${finalFileName}`;
      validBtnDownload.classList.remove('hidden');
      validBtnDownload.disabled = false;

      const previewDiv = document.getElementById('graph-preview');
      if (previewDiv) {
        if (dst === 'onnx') {
          const nodeCount = (result as Graph).nodes.length;
          previewDiv.innerHTML = `<strong>ONNX Graph Generated</strong><br>Nodes: ${nodeCount}<br>Ready for download or 3D viewer.`;
          previewDiv.style.color = '#fff';
        } else {
          previewDiv.innerHTML = `<strong>Text Output Generated</strong><br>Lines: ${String(result).split(String.fromCharCode(10)).length}`;
          previewDiv.style.color = '#fff';
        }
      }
    } catch (_err) {
      const err = _err instanceof Error ? _err : new Error(String(_err));
      log(`Conversion failed: ${err.message}`, 'error');
    }

    validBtnConvert.disabled = false;
  });

  validBtnDownload.addEventListener('click', () => {
    if (finalBlobUrl) {
      const a = document.createElement('a');
      a.href = finalBlobUrl;
      a.download = finalFileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  });

  updateHint();
}

initMmdnnDemo();
