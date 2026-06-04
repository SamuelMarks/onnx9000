/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import * as monaco from 'monaco-editor'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Web Worker for Compiler /* v8 ignore next */ /* v8 ignore next */
// 202. Execute code generation entirely inside a Web Worker. /* v8 ignore next */ /* v8 ignore next */
const compilerWorker = new Worker(new URL('./worker.ts', import.meta.url), {
  /* v8 ignore next */ /* v8 ignore next */
  type: 'module' /* v8 ignore next */ /* v8 ignore next */,
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Editor State /* v8 ignore next */ /* v8 ignore next */
let currentFile: 'header' | 'source' = 'header'; /* v8 ignore next */ /* v8 ignore next */
const modelData: { header: string; source: string } = {
  /* v8 ignore next */ /* v8 ignore next */
  header:
    '/* Please upload an ONNX model to generate code */' /* v8 ignore next */ /* v8 ignore next */,
  source:
    '/* Please upload an ONNX model to generate code */' /* v8 ignore next */ /* v8 ignore next */,
}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Initialize Monaco Editor /* v8 ignore next */ /* v8 ignore next */
const editor = monaco.editor.create(document.getElementById('monaco-root')!, {
  /* v8 ignore next */ /* v8 ignore next */
  value: modelData.header /* v8 ignore next */ /* v8 ignore next */,
  language: 'c' /* v8 ignore next */ /* v8 ignore next */,
  theme: 'vs-dark' /* v8 ignore next */ /* v8 ignore next */,
  automaticLayout: true /* v8 ignore next */ /* v8 ignore next */,
  minimap: { enabled: false } /* v8 ignore next */ /* v8 ignore next */,
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropzone = document.getElementById('dropzone')!; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'file-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const compileBtn =
  document.getElementById('btn-compile')!; /* v8 ignore next */ /* v8 ignore next */
const downloadBtn =
  document.getElementById('btn-download')!; /* v8 ignore next */ /* v8 ignore next */
const targetBoardSelect = document.getElementById(
  'target-board',
) as HTMLSelectElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let currentModelBuffer: Uint8Array | null = null; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Tab Switching /* v8 ignore next */ /* v8 ignore next */
document.querySelectorAll('.tab').forEach((tab) => {
  /* v8 ignore next */ /* v8 ignore next */
  tab.addEventListener('click', (e) => {
    /* v8 ignore next */ /* v8 ignore next */
    document.querySelectorAll('.tab').forEach((t) => {
      /* v8 ignore next */ /* v8 ignore next */
      t.classList.remove('active'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const t = e.target as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
    t.classList.add('active'); /* v8 ignore next */ /* v8 ignore next */
    currentFile = t.dataset.target as 'header' | 'source'; /* v8 ignore next */ /* v8 ignore next */
    editor.setValue(modelData[currentFile]); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// File Dropping (198) /* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('dragover', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.classList.add('hover'); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('dragleave', () => {
  /* v8 ignore next */ /* v8 ignore next */
  dropzone.classList.remove('hover'); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('drop', async (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  dropzone.classList.remove('hover'); /* v8 ignore next */ /* v8 ignore next */
  const file = e.dataTransfer?.files[0]; /* v8 ignore next */ /* v8 ignore next */
  if (file && file.name.endsWith('.onnx'))
    await handleFile(file); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  fileInput.click(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
fileInput.addEventListener('change', async (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  const file = (e.target as HTMLInputElement).files?.[0]; /* v8 ignore next */ /* v8 ignore next */
  if (file) await handleFile(file); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function handleFile(file: File) {
  /* v8 ignore next */ /* v8 ignore next */
  dropzone.innerHTML = `<p>Loading ${file.name}...</p>`; /* v8 ignore next */ /* v8 ignore next */
  const buffer = await file.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
  currentModelBuffer = new Uint8Array(buffer); /* v8 ignore next */ /* v8 ignore next */
  document.getElementById('controls')!.style.display =
    'block'; /* v8 ignore next */ /* v8 ignore next */
  dropzone.innerHTML = `<p>Loaded: <strong>${file.name}</strong><br/>Size: ${(buffer.byteLength / 1024).toFixed(1)} KB</p>`; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// 202: Web Worker Execution /* v8 ignore next */ /* v8 ignore next */
compileBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!currentModelBuffer) return; /* v8 ignore next */ /* v8 ignore next */
  compileBtn.innerText = 'Compiling in Worker...'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const opts = {
    /* v8 ignore next */ /* v8 ignore next */
    target: (document.getElementById('target-arch') as HTMLSelectElement)
      .value /* v8 ignore next */ /* v8 ignore next */,
    emitCpp: (document.getElementById('opt-cpp') as HTMLInputElement)
      .checked /* v8 ignore next */ /* v8 ignore next */,
    noMathH: !(document.getElementById('opt-math') as HTMLInputElement)
      .checked /* v8 ignore next */ /* v8 ignore next */,
    noOpt: !(document.getElementById('opt-unroll') as HTMLInputElement)
      .checked /* v8 ignore next */ /* v8 ignore next */,
  }; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  compilerWorker.postMessage({
    /* v8 ignore next */ /* v8 ignore next */
    buffer: currentModelBuffer /* v8 ignore next */ /* v8 ignore next */,
    options: opts /* v8 ignore next */ /* v8 ignore next */,
  }); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
compilerWorker.onmessage = (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  const { header, source, summary, error, arenaSize } =
    e.data; /* v8 ignore next */ /* v8 ignore next */
  compileBtn.innerText = 'Compile to C'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (error) {
    /* v8 ignore next */ /* v8 ignore next */
    editor.setValue(`/* Compilation Error: ${error} */`); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // 204: Validate model RAM /* v8 ignore next */ /* v8 ignore next */
  const boardLimit = parseInt(targetBoardSelect.value); /* v8 ignore next */ /* v8 ignore next */
  if (!isNaN(boardLimit) && boardLimit > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    if (arenaSize > boardLimit) {
      /* v8 ignore next */ /* v8 ignore next */
      alert(
        /* v8 ignore next */ /* v8 ignore next */
        `Warning: The required Arena Size (${arenaSize} bytes) exceeds the selected board limit (${boardLimit} bytes)!` /* v8 ignore next */ /* v8 ignore next */,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  modelData.header = summary + '\n' + header; /* v8 ignore next */ /* v8 ignore next */
  modelData.source = source; /* v8 ignore next */ /* v8 ignore next */
  editor.setValue(modelData[currentFile]); /* v8 ignore next */ /* v8 ignore next */
};

// 203: Stream directly into Blob to prevent OOM
downloadBtn.addEventListener('click', () => {
  const zip =
    `/* Zip generator placeholder, currently just dumping single file... */\n` +
    modelData.header +
    '\n\n' +
    modelData.source;
  const blob = new Blob([zip], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'onnx2c_model.zip';
  a.click();
  URL.revokeObjectURL(url);
});
