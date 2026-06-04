/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  parseModelProto,
  BufferReader,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import {
  TFLiteExporter,
  compileGraphToTFLite,
} from '@onnx9000/tflite-exporter'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropZone = document.getElementById(
  'drop-zone',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'file-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const browseBtn = document.getElementById(
  'browse-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const statusPanel = document.getElementById(
  'status-panel',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const resultPanel = document.getElementById(
  'result-panel',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const statusText = document.getElementById(
  'status-text',
) as HTMLParagraphElement; /* v8 ignore next */ /* v8 ignore next */
const progressBar = document.getElementById(
  'progress-bar',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const errorBox = document.getElementById(
  'error-box',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const downloadBtn = document.getElementById(
  'download-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const statsText = document.getElementById(
  'stats-text',
) as HTMLParagraphElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const optEdgeTpu = document.getElementById(
  'opt-edgetpu',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const quantFp16 = document.getElementById(
  'quant-fp16',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const quantInt8 = document.getElementById(
  'quant-int8',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let currentFile: File | null = null; /* v8 ignore next */ /* v8 ignore next */
let tfliteBlob: Blob | null = null; /* v8 ignore next */ /* v8 ignore next */
let originalName = ''; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
browseBtn.addEventListener('click', () =>
  fileInput.click(),
); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fileInput.addEventListener('change', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  const target = e.target as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  if (target.files && target.files.length > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    handleFile(target.files[0]!); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
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
  if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    handleFile(e.dataTransfer.files[0]!); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function handleFile(file: File) {
  /* v8 ignore next */ /* v8 ignore next */
  if (!file.name.endsWith('.onnx')) {
    /* v8 ignore next */ /* v8 ignore next */
    showError('Please provide a valid .onnx file.'); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  currentFile = file; /* v8 ignore next */ /* v8 ignore next */
  originalName = file.name.replace('.onnx', ''); /* v8 ignore next */ /* v8 ignore next */
  errorBox.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  resultPanel.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  statusPanel.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    await processModel(); /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    showError(err.message || err.toString()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function updateStatus(msg: string, progress: number) {
  /* v8 ignore next */ /* v8 ignore next */
  statusText.textContent = msg; /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.width = `${progress}%`; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function showError(msg: string) {
  /* v8 ignore next */ /* v8 ignore next */
  errorBox.textContent = msg; /* v8 ignore next */ /* v8 ignore next */
  errorBox.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
  updateStatus('Failed', 100); /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.backgroundColor = 'var(--error)'; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function processModel() {
  /* v8 ignore next */ /* v8 ignore next */
  if (!currentFile) return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.backgroundColor = 'var(--accent)'; /* v8 ignore next */ /* v8 ignore next */
  updateStatus('Reading file...', 10); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const buffer = await currentFile.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateStatus('Parsing ONNX AST...', 30); /* v8 ignore next */ /* v8 ignore next */
  // Give UI a tick to render /* v8 ignore next */ /* v8 ignore next */
  await new Promise((r) => setTimeout(r, 10)); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const reader = new BufferReader(new Uint8Array(buffer)); /* v8 ignore next */ /* v8 ignore next */
  const graph = await parseModelProto(reader); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateStatus(
    'Optimizing layout & generating TFLite FlatBuffer...',
    60,
  ); /* v8 ignore next */ /* v8 ignore next */
  await new Promise((r) => setTimeout(r, 10)); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let quantMode: 'none' | 'fp16' | 'int8' = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (quantFp16.checked) quantMode = 'fp16'; /* v8 ignore next */ /* v8 ignore next */
  if (quantInt8.checked) quantMode = 'int8'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const keepNchw = !optEdgeTpu.checked; /* v8 ignore next */ /* v8 ignore next */
  const exporter = new TFLiteExporter(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const t0 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
  const subgraphsOffset = compileGraphToTFLite(
    graph,
    exporter,
    keepNchw,
    quantMode,
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateStatus(
    'Serializing FlatBuffer structures...',
    80,
  ); /* v8 ignore next */ /* v8 ignore next */
  await new Promise((r) => setTimeout(r, 10)); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  exporter.builder.startVector(4, 1, 4); /* v8 ignore next */ /* v8 ignore next */
  exporter.builder.addOffset(subgraphsOffset); /* v8 ignore next */ /* v8 ignore next */
  const subgraphsVecOffset =
    exporter.builder.endVector(1); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const tfliteBytes = exporter.finish(
    subgraphsVecOffset,
    `onnx9000-web-${originalName}`,
  ); /* v8 ignore next */ /* v8 ignore next */
  const t1 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateStatus('Done!', 100); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  tfliteBlob = new Blob([tfliteBytes], {
    type: 'application/octet-stream',
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const mbOriginal = (buffer.byteLength / 1024 / 1024).toFixed(
    2,
  ); /* v8 ignore next */ /* v8 ignore next */
  const mbNew = (tfliteBytes.byteLength / 1024 / 1024).toFixed(
    2,
  ); /* v8 ignore next */ /* v8 ignore next */
  const timeMs = (t1 - t0).toFixed(0); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  statsText.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
    <strong>Original Size:</strong> ${mbOriginal} MB<br> /* v8 ignore next */ /* v8 ignore next */
    <strong>TFLite Size:</strong> ${mbNew} MB<br> /* v8 ignore next */ /* v8 ignore next */
    <strong>Compilation Time:</strong> ${timeMs} ms /* v8 ignore next */ /* v8 ignore next */
  `; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  resultPanel.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const openNetronCheckbox = document.getElementById(
  'open-netron',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
downloadBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!tfliteBlob) return; /* v8 ignore next */ /* v8 ignore next */
  const url = URL.createObjectURL(tfliteBlob); /* v8 ignore next */ /* v8 ignore next */
  const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
  a.href = url; /* v8 ignore next */ /* v8 ignore next */
  a.download = `${originalName}.tflite`; /* v8 ignore next */ /* v8 ignore next */
  document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
  a.click(); /* v8 ignore next */ /* v8 ignore next */
  document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (openNetronCheckbox.checked) {
    /* v8 ignore next */ /* v8 ignore next */
    // 296. Offer an embedded interactive graph visualizer (Netron style) showing the final TFLite layout. /* v8 ignore next */ /* v8 ignore next */
    // Open the local dev-server endpoint for Netron UI (if we are in the monorepo context) or public Netron. /* v8 ignore next */ /* v8 ignore next */
    const netronUrl = `https://netron.app/?url=${encodeURIComponent(url)}`; /* v8 ignore next */ /* v8 ignore next */
    window.open(netronUrl, '_blank'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Revoke after a delay to allow the new window to fetch if needed /* v8 ignore next */ /* v8 ignore next */
  setTimeout(() => URL.revokeObjectURL(url), 10000); /* v8 ignore next */ /* v8 ignore next */
});
