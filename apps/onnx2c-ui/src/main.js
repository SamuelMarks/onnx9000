import * as monaco from 'monaco-editor';
// Web Worker for Compiler
// 202. Execute code generation entirely inside a Web Worker.
const compilerWorker = new Worker(new URL('./worker.ts', import.meta.url), {
    type: 'module',
});
// Editor State
let currentFile = 'header';
const modelData = {
    header: '/* Please upload an ONNX model to generate code */',
    source: '/* Please upload an ONNX model to generate code */',
};
// Initialize Monaco Editor
const editor = monaco.editor.create(document.getElementById('monaco-root'), {
    value: modelData.header,
    language: 'c',
    theme: 'vs-dark',
    automaticLayout: true,
    minimap: { enabled: false },
});
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const compileBtn = document.getElementById('btn-compile');
const downloadBtn = document.getElementById('btn-download');
const targetBoardSelect = document.getElementById('target-board');
let currentModelBuffer = null;
// Tab Switching
document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', (e) => {
        document.querySelectorAll('.tab').forEach((t) => {
            t.classList.remove('active');
        });
        const t = e.target;
        t.classList.add('active');
        currentFile = t.dataset.target;
        editor.setValue(modelData[currentFile]);
    });
});
// File Dropping (198)
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('hover');
});
dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('hover');
});
dropzone.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropzone.classList.remove('hover');
    const file = e.dataTransfer?.files[0];
    if (file && file.name.endsWith('.onnx'))
        await handleFile(file);
});
dropzone.addEventListener('click', () => {
    fileInput.click();
});
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (file)
        await handleFile(file);
});
async function handleFile(file) {
    dropzone.innerHTML = `<p>Loading ${file.name}...</p>`;
    const buffer = await file.arrayBuffer();
    currentModelBuffer = new Uint8Array(buffer);
    document.getElementById('controls').style.display = 'block';
    dropzone.innerHTML = `<p>Loaded: <strong>${file.name}</strong><br/>Size: ${(buffer.byteLength / 1024).toFixed(1)} KB</p>`;
}
// 202: Web Worker Execution
compileBtn.addEventListener('click', () => {
    if (!currentModelBuffer)
        return;
    compileBtn.innerText = 'Compiling in Worker...';
    const opts = {
        target: document.getElementById('target-arch').value,
        emitCpp: document.getElementById('opt-cpp').checked,
        noMathH: !document.getElementById('opt-math').checked,
        noOpt: !document.getElementById('opt-unroll').checked,
    };
    compilerWorker.postMessage({
        buffer: currentModelBuffer,
        options: opts,
    });
});
compilerWorker.onmessage = (e) => {
    const { header, source, summary, error, arenaSize } = e.data;
    compileBtn.innerText = 'Compile to C';
    if (error) {
        editor.setValue(`/* Compilation Error: ${error} */`);
        return;
    }
    // 204: Validate model RAM
    const boardLimit = parseInt(targetBoardSelect.value);
    if (!isNaN(boardLimit) && boardLimit > 0) {
        if (arenaSize > boardLimit) {
            alert(`Warning: The required Arena Size (${arenaSize} bytes) exceeds the selected board limit (${boardLimit} bytes)!`);
        }
    }
    modelData.header = summary + '\n' + header;
    modelData.source = source;
    editor.setValue(modelData[currentFile]);
};
// 203: Stream directly into Blob to prevent OOM
downloadBtn.addEventListener('click', () => {
    const zip = `/* Zip generator placeholder, currently just dumping single file... */\n` +
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
//# sourceMappingURL=main.js.map