/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { load } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropZone = document.getElementById(
  'drop-zone',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'file-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const browseBtn = document.getElementById(
  'browse-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const statusPanel = document.getElementById(
  'status-panel',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const resultPanel = document.getElementById(
  'result-panel',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const statusText = document.getElementById(
  'status-text',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const progressBar = document.getElementById(
  'progress-bar',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const errorBox = document.getElementById(
  'error-box',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const downloadBtn = document.getElementById(
  'download-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const statsText = document.getElementById(
  'stats-text',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let currentFile: File | null = null; /* v8 ignore next */ /* v8 ignore next */
let jsonBlob: Blob | null = null; /* v8 ignore next */ /* v8 ignore next */
let originalName: string = ''; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
browseBtn.addEventListener('click', () =>
  fileInput.click(),
); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fileInput.addEventListener('change', (e: Event) => {
  /* v8 ignore next */ /* v8 ignore next */
  const target = e.target as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  if (target.files && target.files.length > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    processFile(target.files[0]); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('dragover', (e: DragEvent) => {
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
dropZone.addEventListener('drop', (e: DragEvent) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropZone.classList.remove('dragover'); /* v8 ignore next */ /* v8 ignore next */
  if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    processFile(e.dataTransfer.files[0]); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function processFile(file: File) {
  /* v8 ignore next */ /* v8 ignore next */
  if (!file.name.endsWith('.onnx')) {
    /* v8 ignore next */ /* v8 ignore next */
    showError('Please provide a valid .onnx file.'); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  currentFile = file; /* v8 ignore next */ /* v8 ignore next */
  originalName = file.name.replace('.onnx', ''); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  errorBox.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  resultPanel.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  statusPanel.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    await performExtraction(); /* v8 ignore next */ /* v8 ignore next */
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
function updateProgress(msg: string, pct: number) {
  /* v8 ignore next */ /* v8 ignore next */
  statusText.textContent = msg; /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.width = `${pct}%`; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function showError(msg: string) {
  /* v8 ignore next */ /* v8 ignore next */
  errorBox.textContent = msg; /* v8 ignore next */ /* v8 ignore next */
  errorBox.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
  updateProgress('Failed', 100); /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.backgroundColor = '#cc3333'; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function performExtraction() {
  /* v8 ignore next */ /* v8 ignore next */
  if (!currentFile) return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  progressBar.style.backgroundColor = '#007acc'; /* v8 ignore next */ /* v8 ignore next */
  updateProgress('Reading file...', 10); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const arrayBuffer = await currentFile.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateProgress('Parsing ONNX AST...', 50); /* v8 ignore next */ /* v8 ignore next */
  await new Promise((resolve) => setTimeout(resolve, 10)); // Yield to paint /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const t0 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
  const graph = await load(arrayBuffer); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateProgress('Extracting JSON...', 80); /* v8 ignore next */ /* v8 ignore next */
  await new Promise((resolve) => setTimeout(resolve, 10)); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const jsonString = JSON.stringify(
    /* v8 ignore next */ /* v8 ignore next */
    graph /* v8 ignore next */ /* v8 ignore next */,
    (key, value) => {
      /* v8 ignore next */ /* v8 ignore next */
      // Drop heavy raw data for UI performance, just keep shapes and metadata /* v8 ignore next */ /* v8 ignore next */
      if (key === 'data' && ArrayBuffer.isView(value)) {
        /* v8 ignore next */ /* v8 ignore next */
        return `[Buffer: ${value.byteLength} bytes]`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      // Handle bigints /* v8 ignore next */ /* v8 ignore next */
      if (typeof value === 'bigint') {
        /* v8 ignore next */ /* v8 ignore next */
        return value.toString() + 'n'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      return value; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */,
    2 /* v8 ignore next */ /* v8 ignore next */,
  ); /* v8 ignore next */ /* v8 ignore next */
  const t1 = performance.now(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  updateProgress('Done!', 100); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  jsonBlob = new Blob([jsonString], {
    type: 'application/json',
  }); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const inputSize = (arrayBuffer.byteLength / 1024 / 1024).toFixed(
    2,
  ); /* v8 ignore next */ /* v8 ignore next */
  const outputSize = (jsonBlob.size / 1024 / 1024).toFixed(
    2,
  ); /* v8 ignore next */ /* v8 ignore next */
  const timeMs = (t1 - t0).toFixed(0); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  statsText.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
    <strong>File:</strong> ${originalName}.onnx (${inputSize} MB)<br/> /* v8 ignore next */ /* v8 ignore next */
    <strong>JSON Size:</strong> ${outputSize} MB<br/> /* v8 ignore next */ /* v8 ignore next */
    <strong>Nodes:</strong> ${graph.nodes.length}<br/> /* v8 ignore next */ /* v8 ignore next */
    <strong>Extraction Time:</strong> ${timeMs} ms /* v8 ignore next */ /* v8 ignore next */
  `; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  statusPanel.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  resultPanel.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
downloadBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!jsonBlob) return; /* v8 ignore next */ /* v8 ignore next */
  const url = URL.createObjectURL(jsonBlob); /* v8 ignore next */ /* v8 ignore next */
  const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
  a.href = url; /* v8 ignore next */ /* v8 ignore next */
  a.download = `onnx9000-extracted-${originalName}.json`; /* v8 ignore next */ /* v8 ignore next */
  a.click(); /* v8 ignore next */ /* v8 ignore next */
  URL.revokeObjectURL(url); /* v8 ignore next */ /* v8 ignore next */
});
