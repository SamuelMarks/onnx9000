import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('@onnx9000/core', () => ({
  load: vi.fn().mockResolvedValue({
    nodes: [1, 2, 3],
    bigVal: 10n,
    data: new Uint8Array([1, 2, 3]),
  }),
}));

describe('demo-json-extract app', () => {
  let dropZone: any;
  let fileInput: any;
  let browseBtn: any;
  let statusPanel: any;
  let resultPanel: any;
  let statusText: any;
  let progressBar: any;
  let errorBox: any;
  let downloadBtn: any;
  let statsText: any;

  beforeEach(() => {
    dropZone = { addEventListener: vi.fn(), classList: { add: vi.fn(), remove: vi.fn() } };
    fileInput = { addEventListener: vi.fn(), click: vi.fn(), files: [] };
    browseBtn = { addEventListener: vi.fn() };
    statusPanel = { classList: { add: vi.fn(), remove: vi.fn() } };
    resultPanel = { classList: { add: vi.fn(), remove: vi.fn() } };
    statusText = { textContent: '' };
    progressBar = { style: { width: '', backgroundColor: '' } };
    errorBox = { classList: { add: vi.fn(), remove: vi.fn() }, textContent: '' };
    downloadBtn = { addEventListener: vi.fn() };
    statsText = { innerHTML: '' };

    vi.stubGlobal('document', {
      getElementById: vi.fn((id) => {
        const map: any = {
          'drop-zone': dropZone,
          'file-input': fileInput,
          'browse-btn': browseBtn,
          'status-panel': statusPanel,
          'result-panel': resultPanel,
          'status-text': statusText,
          'progress-bar': progressBar,
          'error-box': errorBox,
          'download-btn': downloadBtn,
          'stats-text': statsText,
        };
        return map[id];
      }),
      createElement: vi.fn(() => ({ click: vi.fn(), href: '', download: '' })),
    });

    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:mock'),
      revokeObjectURL: vi.fn(),
    });

    vi.stubGlobal(
      'Blob',
      class {
        size: number;
        constructor(parts: any[]) {
          this.size = 1000;
        }
      },
    );

    vi.stubGlobal('performance', {
      now: vi.fn(() => 1000),
    });

    vi.useFakeTimers();

    // Mock global view method but keep ArrayBuffer intact
    const originalArrayBuffer = globalThis.ArrayBuffer;
    vi.stubGlobal(
      'ArrayBuffer',
      class extends originalArrayBuffer {
        static isView(v: any) {
          return v instanceof Uint8Array;
        }
      },
    );
  });

  afterEach(() => {
    vi.runOnlyPendingTimers();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.resetModules();
  });

  it('handles browse button click', async () => {
    await import('../app.js');
    const clickHandler = browseBtn.addEventListener.mock.calls[0][1];
    clickHandler();
    expect(fileInput.click).toHaveBeenCalled();
  });

  it('handles file input change', async () => {
    // Reset the load mock specifically for this to ensure we return nodes correctly
    const { load } = await import('@onnx9000/core');
    vi.mocked(load).mockResolvedValueOnce({
      nodes: [1, 2, 3],
      bigVal: 10n,
      data: new Uint8Array([1, 2, 3]),
    });

    await import('../app.js');
    const changeHandler = fileInput.addEventListener.mock.calls[0][1];

    changeHandler({ target: { files: [] } });

    changeHandler({ target: { files: [{ name: 'test.txt' }] } });
    expect(errorBox.textContent).toBe('Please provide a valid .onnx file.');

    const buf = new Uint8Array([1]).buffer;
    const validFile = { name: 'test.onnx', arrayBuffer: vi.fn().mockResolvedValue(buf) };

    changeHandler({ target: { files: [validFile] } });

    // Process async boundary 1: arrayBuffer
    await Promise.resolve();
    // Process async boundary 2: setTimeout
    await vi.advanceTimersByTimeAsync(10);
    // Process async boundary 3: load()
    await Promise.resolve();
    // Process async boundary 4: setTimeout
    await vi.advanceTimersByTimeAsync(10);
    // Process async boundary 5: finally block or completion
    await Promise.resolve();

    expect(statusPanel.classList.remove).toHaveBeenCalledWith('hidden');
    // Result panel is revealed only after processing has been performed
    expect(resultPanel.classList.remove).toHaveBeenCalledWith('hidden');
    expect(statsText.innerHTML).toContain('test.onnx');
  });

  it('handles drag and drop', async () => {
    const { load } = await import('@onnx9000/core');
    vi.mocked(load).mockResolvedValueOnce({
      nodes: [1, 2, 3],
    });

    await import('../app.js');

    const dragOverHandler = dropZone.addEventListener.mock.calls.find(
      (c: any) => c[0] === 'dragover',
    )[1];
    const dragLeaveHandler = dropZone.addEventListener.mock.calls.find(
      (c: any) => c[0] === 'dragleave',
    )[1];
    const dropHandler = dropZone.addEventListener.mock.calls.find((c: any) => c[0] === 'drop')[1];

    const e = { preventDefault: vi.fn(), dataTransfer: { files: [] } };

    dragOverHandler(e);
    expect(e.preventDefault).toHaveBeenCalled();
    expect(dropZone.classList.add).toHaveBeenCalledWith('dragover');

    dragLeaveHandler();
    expect(dropZone.classList.remove).toHaveBeenCalledWith('dragover');

    // Test empty drop
    dropHandler(e);

    // Test valid drop
    const buf = new Uint8Array([1]).buffer;
    e.dataTransfer.files = [
      { name: 'test.onnx', arrayBuffer: vi.fn().mockResolvedValue(buf) },
    ] as any;
    dropHandler(e);

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();

    expect(statusPanel.classList.remove).toHaveBeenCalledWith('hidden');
  });

  it('handles missing current file in performExtraction', async () => {
    await import('../app.js');
  });

  it('handles errors during extraction', async () => {
    const { load } = await import('@onnx9000/core');
    vi.mocked(load).mockRejectedValueOnce(new Error('Load failed'));

    await import('../app.js');
    const changeHandler = fileInput.addEventListener.mock.calls[0][1];
    const buf = new Uint8Array([1]).buffer;
    const validFile = { name: 'test.onnx', arrayBuffer: vi.fn().mockResolvedValue(buf) };
    changeHandler({ target: { files: [validFile] } });

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();

    expect(errorBox.textContent).toBe('Load failed');
    expect(progressBar.style.backgroundColor).toBe('#cc3333');
  });

  it('handles non-error objects thrown during extraction', async () => {
    const { load } = await import('@onnx9000/core');
    vi.mocked(load).mockRejectedValueOnce('String error');

    await import('../app.js');
    const changeHandler = fileInput.addEventListener.mock.calls[0][1];
    const buf = new Uint8Array([1]).buffer;
    const validFile = { name: 'test.onnx', arrayBuffer: vi.fn().mockResolvedValue(buf) };
    changeHandler({ target: { files: [validFile] } });

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();

    expect(errorBox.textContent).toBe('String error');
  });

  it('handles download', async () => {
    const { load } = await import('@onnx9000/core');
    vi.mocked(load).mockResolvedValueOnce({
      nodes: [1, 2, 3],
    });

    await import('../app.js');

    // click before blob
    const downloadHandler = downloadBtn.addEventListener.mock.calls[0][1];
    downloadHandler(); // no blob

    // Set up a valid file and extract to populate jsonBlob
    const changeHandler = fileInput.addEventListener.mock.calls[0][1];
    const buf = new Uint8Array([1]).buffer;
    const validFile = { name: 'test.onnx', arrayBuffer: vi.fn().mockResolvedValue(buf) };
    changeHandler({ target: { files: [validFile] } });

    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();
    await vi.advanceTimersByTimeAsync(10);
    await Promise.resolve();

    // click after blob
    const mockA = document.createElement('a') as any;
    vi.mocked(document.createElement).mockReturnValue(mockA);
    downloadHandler();

    expect(mockA.href).toBe('blob:mock');
    expect(mockA.download).toBe('onnx9000-extracted-test.json');
    expect(mockA.click).toHaveBeenCalled();
  });
});
