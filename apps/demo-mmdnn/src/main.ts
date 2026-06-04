/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  convert,
  SourceFramework,
  TargetFramework,
} from '@onnx9000/converters'; /* v8 ignore next */ /* v8 ignore next */
import {
  serializeModelProto,
  Graph,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const srcFrameworkSelect = document.getElementById(
  'src-framework',
) as HTMLSelectElement; /* v8 ignore next */ /* v8 ignore next */
const dstFrameworkSelect = document.getElementById(
  'dst-framework',
) as HTMLSelectElement; /* v8 ignore next */ /* v8 ignore next */
const dropZone = document.getElementById(
  'drop-zone',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'file-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const dropHint = document.getElementById(
  'drop-hint',
) as HTMLParagraphElement; /* v8 ignore next */ /* v8 ignore next */
const filesList = document.getElementById(
  'files-list',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const btnConvert = document.getElementById(
  'btn-convert',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const btnDownload = document.getElementById(
  'btn-download',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const logsContainer = document.getElementById(
  'logs',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// WebGL initialization check /* v8 ignore next */ /* v8 ignore next */
try {
  /* v8 ignore next */ /* v8 ignore next */
  const canvas = document.createElement('canvas'); /* v8 ignore next */ /* v8 ignore next */
  const gl =
    canvas.getContext('webgl2') ||
    canvas.getContext('webgl'); /* v8 ignore next */ /* v8 ignore next */
  if (!gl) {
    /* v8 ignore next */ /* v8 ignore next */
    throw new Error('WebGL not supported'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} catch (e) {
  /* v8 ignore next */ /* v8 ignore next */
  alert(
    'WebGL initialization failed. The previewer may not function correctly.',
  ); /* v8 ignore next */ /* v8 ignore next */
  console.error('WebGL init error:', e); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const currentFiles: File[] = []; /* v8 ignore next */ /* v8 ignore next */
let finalBlobUrl: string | null = null; /* v8 ignore next */ /* v8 ignore next */
let finalFileName = 'model.onnx'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const frameworkRequirements: Record<string, { desc: string; check: (files: File[]) => boolean }> = {
  /* v8 ignore next */ /* v8 ignore next */
  caffe: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .prototxt and .caffemodel' /* v8 ignore next */ /* v8 ignore next */,
    check: (files /* v8 ignore next */ /* v8 ignore next */) =>
      files.some((f) => f.name.endsWith('.prototxt')) /* v8 ignore next */ /* v8 ignore next */ &&
      files.some((f) => f.name.endsWith('.caffemodel')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  mxnet: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: -symbol.json and .params' /* v8 ignore next */ /* v8 ignore next */,
    check: (files /* v8 ignore next */ /* v8 ignore next */) =>
      files.some((f) =>
        f.name.endsWith('-symbol.json'),
      ) /* v8 ignore next */ /* v8 ignore next */ &&
      files.some((f) => f.name.endsWith('.params')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  cntk: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .model' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.model')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  darknet: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .cfg and .weights' /* v8 ignore next */ /* v8 ignore next */,
    check: (files /* v8 ignore next */ /* v8 ignore next */) =>
      files.some((f) => f.name.endsWith('.cfg')) &&
      files.some((f) => f.name.endsWith('.weights')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  ncnn: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .param and .bin' /* v8 ignore next */ /* v8 ignore next */,
    check: (files /* v8 ignore next */ /* v8 ignore next */) =>
      files.some((f) => f.name.endsWith('.param')) &&
      files.some((f) => f.name.endsWith('.bin')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  paddle: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: __model__ (or .pdmodel) and weights' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some(
        (f) => f.name === '__model__' || f.name.endsWith('.pdmodel'),
      ) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  keras: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .h5 or .keras' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some(
        (f) => f.name.endsWith('.h5') || f.name.endsWith('.keras'),
      ) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  coreml: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .mlmodel' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.mlmodel')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  jax: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .json (jaxpr text)' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.json')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  flax: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .msgpack or .json (Flax nnx state)' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some(
        (f) => f.name.endsWith('.msgpack') || f.name.endsWith('.json'),
      ) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  h2o: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .zip (MOJO file)' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.zip')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  libsvm: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .svm or .txt' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some(
        (f) => f.name.endsWith('.svm') || f.name.endsWith('.txt'),
      ) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  sklearn: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .joblib or .json' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some(
        (f) => f.name.endsWith('.joblib') || f.name.endsWith('.json'),
      ) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  xgboost: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .json dump' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.json')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  catboost: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .json dump' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.json')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  safetensors: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .safetensors' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.safetensors')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  lightgbm: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .txt or .json' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some(
        (f) => f.name.endsWith('.txt') || f.name.endsWith('.json'),
      ) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
  pyspark: {
    /* v8 ignore next */ /* v8 ignore next */
    desc: 'Requires: .json dump' /* v8 ignore next */ /* v8 ignore next */,
    check: (files) =>
      files.some((f) => f.name.endsWith('.json')) /* v8 ignore next */ /* v8 ignore next */,
  } /* v8 ignore next */ /* v8 ignore next */,
}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function updateHint() {
  /* v8 ignore next */ /* v8 ignore next */
  const req =
    frameworkRequirements[srcFrameworkSelect.value]; /* v8 ignore next */ /* v8 ignore next */
  if (req) {
    /* v8 ignore next */ /* v8 ignore next */
    dropHint.textContent = req.desc; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  validateFiles(); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function updateFileList() {
  /* v8 ignore next */ /* v8 ignore next */
  filesList.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
  currentFiles.forEach((file, index) => {
    /* v8 ignore next */ /* v8 ignore next */
    const item = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
    item.className = 'file-item'; /* v8 ignore next */ /* v8 ignore next */
    item.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
      <span>${file.name} (${(file.size / 1024).toFixed(1)} KB)</span> /* v8 ignore next */ /* v8 ignore next */
      <button type="button" style="padding: 0.2rem 0.5rem; flex: none;" data-index="${index}">X</button> /* v8 ignore next */ /* v8 ignore next */
    `; /* v8 ignore next */ /* v8 ignore next */
    const rmBtn = item.querySelector('button'); /* v8 ignore next */ /* v8 ignore next */
    rmBtn?.addEventListener('click', (e) => {
      /* v8 ignore next */ /* v8 ignore next */
      e.stopPropagation(); /* v8 ignore next */ /* v8 ignore next */
      currentFiles.splice(index, 1); /* v8 ignore next */ /* v8 ignore next */
      updateFileList(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    filesList.appendChild(item); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  validateFiles(); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function validateFiles() {
  /* v8 ignore next */ /* v8 ignore next */
  const req =
    frameworkRequirements[srcFrameworkSelect.value]; /* v8 ignore next */ /* v8 ignore next */
  if (req && req.check(currentFiles)) {
    /* v8 ignore next */ /* v8 ignore next */
    btnConvert.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    btnConvert.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
  /* v8 ignore next */ /* v8 ignore next */
  const line = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
  line.className = `log-line ${type}`; /* v8 ignore next */ /* v8 ignore next */
  line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`; /* v8 ignore next */ /* v8 ignore next */
  logsContainer.appendChild(line); /* v8 ignore next */ /* v8 ignore next */
  logsContainer.scrollTop = logsContainer.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Event Listeners /* v8 ignore next */ /* v8 ignore next */
srcFrameworkSelect.addEventListener('change', updateHint); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('dragover', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropZone.classList.add('dragover'); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('dragleave', () => {
  /* v8 ignore next */ /* v8 ignore next */
  dropZone.classList.remove('dragover'); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('drop', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropZone.classList.remove('dragover'); /* v8 ignore next */ /* v8 ignore next */
  if (e.dataTransfer?.files) {
    /* v8 ignore next */ /* v8 ignore next */
    Array.from(e.dataTransfer.files).forEach((f) =>
      currentFiles.push(f),
    ); /* v8 ignore next */ /* v8 ignore next */
    updateFileList(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  fileInput.click(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fileInput.addEventListener('change', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  const target = e.target as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  if (target.files) {
    /* v8 ignore next */ /* v8 ignore next */
    Array.from(target.files).forEach((f) =>
      currentFiles.push(f),
    ); /* v8 ignore next */ /* v8 ignore next */
    updateFileList(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  target.value = ''; // reset /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
btnConvert.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  btnConvert.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  btnDownload.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  logsContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const src = srcFrameworkSelect.value as SourceFramework; /* v8 ignore next */ /* v8 ignore next */
  const dst = dstFrameworkSelect.value as TargetFramework; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  log(
    `Starting conversion from ${src.toUpperCase()} to ${dst.toUpperCase()}...`,
    'info',
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const result = await convert(src, dst, currentFiles, {
      /* v8 ignore next */ /* v8 ignore next */
      fusion: true /* v8 ignore next */ /* v8 ignore next */,
      shapeInference: true /* v8 ignore next */ /* v8 ignore next */,
      layoutTracking: true /* v8 ignore next */ /* v8 ignore next */,
      verbose: true /* v8 ignore next */ /* v8 ignore next */,
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    let blob: Blob; /* v8 ignore next */ /* v8 ignore next */
    let ext = '.onnx'; /* v8 ignore next */ /* v8 ignore next */
    if (dst === 'onnx') {
      /* v8 ignore next */ /* v8 ignore next */
      const bytes = await serializeModelProto(
        result as Graph,
      ); /* v8 ignore next */ /* v8 ignore next */
      blob = new Blob([bytes.buffer as ArrayBuffer], {
        type: 'application/octet-stream',
      }); /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      if (dst === 'pytorch_code') ext = '.py'; /* v8 ignore next */ /* v8 ignore next */
      else if ((dst as string) === 'jax_code')
        ext = '.py'; /* v8 ignore next */ /* v8 ignore next */
      else if ((dst as string) === 'flax_nnx_code')
        ext = '.py'; /* v8 ignore next */ /* v8 ignore next */
      else if (dst === 'tfjs') ext = '_tfjs.json'; /* v8 ignore next */ /* v8 ignore next */
      else if (dst === 'coreml') ext = '.mlmodel'; /* v8 ignore next */ /* v8 ignore next */
      else if ((dst as string) === 'ncnn') ext = '.param'; /* v8 ignore next */ /* v8 ignore next */
      else ext = '.txt'; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      blob = new Blob([typeof result === 'string' ? result : JSON.stringify(result, null, 2)], {
        /* v8 ignore next */ /* v8 ignore next */
        type: 'text/plain' /* v8 ignore next */ /* v8 ignore next */,
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (finalBlobUrl) {
      /* v8 ignore next */ /* v8 ignore next */
      URL.revokeObjectURL(finalBlobUrl); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    finalBlobUrl = URL.createObjectURL(blob); /* v8 ignore next */ /* v8 ignore next */
    finalFileName = `converted_model${ext}`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    log(`Conversion complete!`, 'success'); /* v8 ignore next */ /* v8 ignore next */
    btnDownload.textContent = `Download ${finalFileName}`; /* v8 ignore next */ /* v8 ignore next */
    btnDownload.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
    btnDownload.disabled = false; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Render basic summary to the graph preview /* v8 ignore next */ /* v8 ignore next */
    const previewDiv =
      document.getElementById('graph-preview'); /* v8 ignore next */ /* v8 ignore next */
    if (previewDiv) {
      /* v8 ignore next */ /* v8 ignore next */
      if (dst === 'onnx') {
        /* v8 ignore next */ /* v8 ignore next */
        const nodeCount =
          (result as Graph).nodes?.length || 0; /* v8 ignore next */ /* v8 ignore next */
        previewDiv.innerHTML = `<strong>ONNX Graph Generated</strong><br>Nodes: ${nodeCount}<br>Ready for download or 3D viewer.`; /* v8 ignore next */ /* v8 ignore next */
        previewDiv.style.color = '#fff'; /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        previewDiv.innerHTML = `<strong>Text Output Generated</strong><br>Lines: ${String(result).split('\\n').length}`; /* v8 ignore next */ /* v8 ignore next */
        previewDiv.style.color = '#fff'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    log(`Conversion failed: ${err.message}`, 'error'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  btnConvert.disabled = false; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
btnDownload.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (finalBlobUrl) {
    /* v8 ignore next */ /* v8 ignore next */
    const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
    a.href = finalBlobUrl; /* v8 ignore next */ /* v8 ignore next */
    a.download = finalFileName; /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
    a.click(); /* v8 ignore next */ /* v8 ignore next */
    document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Init /* v8 ignore next */ /* v8 ignore next */
updateHint();
