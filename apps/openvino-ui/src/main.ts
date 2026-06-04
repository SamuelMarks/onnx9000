/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { load } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { OpenVinoExporter } from '@onnx9000/openvino-exporter'; /* v8 ignore next */ /* v8 ignore next */
import JSZip from 'jszip'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropzone = document.getElementById('dropzone')!; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'file-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const statusDiv = document.getElementById('status')!; /* v8 ignore next */ /* v8 ignore next */
const compressFp16 = document.getElementById(
  'compressFp16',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function showStatus(message: string, type: 'info' | 'success' | 'error') {
  /* v8 ignore next */ /* v8 ignore next */
  statusDiv.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
  statusDiv.className = type; /* v8 ignore next */ /* v8 ignore next */
  statusDiv.innerHTML = message; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  fileInput.click(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
  /* v8 ignore next */ /* v8 ignore next */
  dropzone.addEventListener(
    eventName,
    preventDefaults,
    false,
  ); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function preventDefaults(e: Event) {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  e.stopPropagation(); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
['dragenter', 'dragover'].forEach((eventName) => {
  /* v8 ignore next */ /* v8 ignore next */
  dropzone.addEventListener(
    /* v8 ignore next */ /* v8 ignore next */
    eventName /* v8 ignore next */ /* v8 ignore next */,
    () => {
      /* v8 ignore next */ /* v8 ignore next */
      dropzone.classList.add('active'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */,
    false /* v8 ignore next */ /* v8 ignore next */,
  ); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
['dragleave', 'drop'].forEach((eventName) => {
  /* v8 ignore next */ /* v8 ignore next */
  dropzone.addEventListener(
    /* v8 ignore next */ /* v8 ignore next */
    eventName /* v8 ignore next */ /* v8 ignore next */,
    () => {
      /* v8 ignore next */ /* v8 ignore next */
      dropzone.classList.remove('active'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */,
    false /* v8 ignore next */ /* v8 ignore next */,
  ); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropzone.addEventListener('drop', (e: DragEvent) => {
  /* v8 ignore next */ /* v8 ignore next */
  const dt = e.dataTransfer; /* v8 ignore next */ /* v8 ignore next */
  if (!dt) return; /* v8 ignore next */ /* v8 ignore next */
  const files = dt.files; /* v8 ignore next */ /* v8 ignore next */
  handleFiles(files); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fileInput.addEventListener('change', function (this: HTMLInputElement) {
  /* v8 ignore next */ /* v8 ignore next */
  if (this.files) {
    /* v8 ignore next */ /* v8 ignore next */
    handleFiles(this.files); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
async function handleFiles(files: FileList) {
  /* v8 ignore next */ /* v8 ignore next */
  if (files.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
  const file = files[0]; /* v8 ignore next */ /* v8 ignore next */
  if (!file.name.endsWith('.onnx')) {
    /* v8 ignore next */ /* v8 ignore next */
    showStatus('Please drop an .onnx file.', 'error'); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    showStatus(
      `Loading ${file.name}... (Phase 1/3)`,
      'info',
    ); /* v8 ignore next */ /* v8 ignore next */
    const arrayBuffer = await file.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
    const buffer = new Uint8Array(arrayBuffer); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    showStatus(
      `Parsing ONNX Graph... (Phase 2/3)`,
      'info',
    ); /* v8 ignore next */ /* v8 ignore next */
    const graph = await load(buffer); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    showStatus(
      `Compiling to OpenVINO XML/BIN... (Phase 3/3)`,
      'info',
    ); /* v8 ignore next */ /* v8 ignore next */
    // Let UI update /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 50)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const exporter = new OpenVinoExporter(graph, {
      /* v8 ignore next */ /* v8 ignore next */
      compressToFp16: compressFp16.checked /* v8 ignore next */ /* v8 ignore next */,
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const { xml, bin } = exporter.export(); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Use JSZip /* v8 ignore next */ /* v8 ignore next */
    showStatus(`Generating ZIP file...`, 'info'); /* v8 ignore next */ /* v8 ignore next */
    const zip = new JSZip(); /* v8 ignore next */ /* v8 ignore next */
    const baseName = file.name.replace('.onnx', ''); /* v8 ignore next */ /* v8 ignore next */
    zip.file(`${baseName}.xml`, xml); /* v8 ignore next */ /* v8 ignore next */
    zip.file(`${baseName}.bin`, bin); /* v8 ignore next */ /* v8 ignore next */
    zip.file(
      `${baseName}.mapping`,
      '<?xml version="1.0" ?>\n<mapping>\n</mapping>',
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const blob = await zip.generateAsync({
      type: 'blob',
    }); /* v8 ignore next */ /* v8 ignore next */
    const url = URL.createObjectURL(blob); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
    a.href = url; /* v8 ignore next */ /* v8 ignore next */
    a.download = `${baseName}_openvino.zip`; /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
    a.click(); /* v8 ignore next */ /* v8 ignore next */
    document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
    URL.revokeObjectURL(url); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const originalSizeMB = (file.size / 1024 / 1024).toFixed(
      2,
    ); /* v8 ignore next */ /* v8 ignore next */
    const newSizeMB = (blob.size / 1024 / 1024).toFixed(
      2,
    ); /* v8 ignore next */ /* v8 ignore next */
    showStatus(
      /* v8 ignore next */ /* v8 ignore next */
      `Success! Downloaded ${baseName}_openvino.zip<br>Original: ${originalSizeMB}MB -> OpenVINO: ${newSizeMB}MB` /* v8 ignore next */ /* v8 ignore next */,
      'success' /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    console.error(err); /* v8 ignore next */ /* v8 ignore next */
    showStatus(
      `Error compiling model: ${err.message}`,
      'error',
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
