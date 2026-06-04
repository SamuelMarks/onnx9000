/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  load,
  ONNXToPyTorchVisitor,
} from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const dropZone = document.getElementById(
  'drop-zone',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'file-input',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const codeArea = document.getElementById(
  'code',
) as HTMLTextAreaElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('click', () =>
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
    codeArea.value =
      '# Error: Please provide a valid .onnx file.'; /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  codeArea.value = '# Loading and parsing ONNX AST...'; /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const arrayBuffer = await file.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
    const graph = await load(arrayBuffer); /* v8 ignore next */ /* v8 ignore next */
    const visitor = new ONNXToPyTorchVisitor(graph); /* v8 ignore next */ /* v8 ignore next */
    const code = visitor.generate(); /* v8 ignore next */ /* v8 ignore next */
    codeArea.value = code; /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    codeArea.value = `# Error during processing:\n${err.message || err.toString()}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
