import { CanvasRenderer } from './render/canvas';
import { fetchAndParseModel } from './parser/fetcher';
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '@onnx9000/modifier/dist/GraphMutator.js';
import { ModelExporter } from '@onnx9000/modifier/dist/components/export/exporter.js';

// Initialize UI
// 131, 132. Implement Dark/Light mode via CSS custom properties
const uiHTML = `
  <div style="position: absolute; top: 10px; left: 10px; z-index: 10; color: var(--text); background: var(--panel-bg); padding: 10px; border-radius: 5px; font-family: sans-serif; max-width: 300px;">
    <h3 style="margin-top:0">ONNX9000 Netron</h3>
    <input type="file" id="file-upload" accept=".onnx,.pb,.tflite,.pt,.h5" style="display:none;" />
    <div id="drop-zone" style="border: 2px dashed var(--border); padding: 20px; text-align: center; border-radius: 5px; cursor: pointer; margin-bottom: 10px;">
      Drop .onnx file here or click to browse
    </div>
    <div id="breadcrumb" style="font-size: 0.9em; margin-bottom: 10px; color: #4A90E2; cursor: pointer; display: none;">
       &larr; Back to Main Graph
    </div>
    <input type="text" id="search-box" placeholder="Search node, op, tensor..." style="width: 100%; padding: 5px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; box-sizing: border-box; margin-bottom: 5px;" />
    <div id="search-results" style="font-size: 0.8em; color: var(--text-muted); margin-bottom: 10px;"></div>
    
    <div style="font-size: 0.9em; margin-bottom: 10px;">
      <b>View Options:</b><br/>
      <label><input type="checkbox" id="filter-control-edges" /> Hide control flow edges</label><br/>
      <input type="text" id="color-regex" placeholder="Regex (e.g. ^LayerNorm) for coloring..." style="width: 100%; padding: 3px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); border-radius: 2px; margin-top: 3px; font-size: 0.8em;" />
    </div>

    <div id="status" aria-live="polite">Waiting for file...</div>
    <div id="aria-announcer" aria-live="assertive" class="sr-only" style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0;"></div>
    <div style="margin-top: 15px;">
       <button id="btn-help" style="width: 100%; padding: 8px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); cursor: pointer; border-radius: 4px;">Keyboard Shortcuts</button>
    </div>
  </div>
  
  <div id="help-modal" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--panel-bg); color: var(--text); padding: 20px; border: 1px solid var(--border); border-radius: 8px; z-index: 1000; min-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
    <h2 style="margin-top: 0;">Keyboard Shortcuts</h2>
    <ul style="padding-left: 20px; line-height: 1.6;">
      <li><b>Click</b>: Select Node</li>
      <li><b>Shift+Click</b>: Multi-Select Nodes</li>
      <li><b>Right Click</b>: Context Menu</li>
      <li><b>Scroll / Trackpad</b>: Pan</li>
      <li><b>Pinch</b>: Zoom</li>
      <li><b>Ctrl/Cmd + Scroll</b>: Zoom</li>
      <li><b>Enter (in search)</b>: Step through results</li>
    </ul>
    <button id="close-help-btn" style="margin-top: 10px; width: 100%; padding: 8px; background: #4A90E2; color: white; border: none; border-radius: 4px; cursor: pointer;">Close</button>
  </div>
  <div id="sidebar" style="position: absolute; top: 0; right: 0; width: 350px; height: 100vh; background: var(--sidebar-bg); color: var(--text); border-left: 1px solid var(--border); overflow-y: auto; display: none; padding: 15px; font-family: sans-serif; box-sizing: border-box;">
  </div>
  <div id="context-menu" style="position: absolute; display: none; background: var(--panel-bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 5px; font-family: sans-serif; font-size: 14px; z-index: 100; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);">
    <div class="menu-item" id="menu-extract-subgraph" style="padding: 5px 10px; cursor: pointer;">Extract Subgraph</div>
    <div class="menu-item" id="menu-extract-python" style="padding: 5px 10px; cursor: pointer;">Extract to Python script</div>
    <div class="menu-item" id="menu-replace-constant" style="padding: 5px 10px; cursor: pointer;">Replace with Constant</div>
    <div class="menu-item" id="menu-export-png" style="padding: 5px 10px; cursor: pointer;">Export as PNG</div>
    <div class="menu-item" id="menu-export-json" style="padding: 5px 10px; cursor: pointer;">Export Node to JSON</div>
    <div class="menu-item" id="menu-copy-attributes" style="padding: 5px 10px; cursor: pointer;">Copy Attributes</div>
    <div class="menu-item" id="menu-paste-attributes" style="padding: 5px 10px; cursor: pointer; display: none;">Paste Attributes</div>
    <div class="menu-item" id="menu-duplicate-node" style="padding: 5px 10px; cursor: pointer;">Duplicate Node</div>
  </div>
  <canvas id="view" style="display:block; background: var(--bg); width: 100vw; height: 100vh;"></canvas>
`;
document.body.innerHTML = uiHTML;

const style = document.createElement('style');
style.innerHTML = `
  :root {
    --bg: #f4f4f4;
    --panel-bg: #ffffff;
    --sidebar-bg: #f9f9f9;
    --text: #333333;
    --text-muted: #666666;
    --border: #dddddd;
    --input-bg: #ffffff;
    --hover-bg: #e0e0e0;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #111111;
      --panel-bg: #222222;
      --sidebar-bg: #1a1a1a;
      --text: #dddddd;
      --text-muted: #aaaaaa;
      --border: #333333;
      --input-bg: #333333;
      --hover-bg: #555555;
    }
  }
  .menu-item:hover { background: var(--hover-bg); }
`;
document.head.appendChild(style);

document.body.style.margin = '0';
document.body.style.overflow = 'hidden';

const canvas = document.getElementById('view') as HTMLCanvasElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;
const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const sidebar = document.getElementById('sidebar') as HTMLDivElement;

// 244. Handle the "Drop ONNX file here" UX explicitly
// 245. Validate file drop on all operating systems

dropZone.addEventListener('click', () => {
  fileUpload.click();
});

fileUpload.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) handleFile(file);
});

// Window level drag/drop
window.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.style.backgroundColor = 'var(--hover-bg)';
});

window.addEventListener('dragleave', (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.style.backgroundColor = '';
});

window.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.style.backgroundColor = '';
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});
const searchBox = document.getElementById('search-box') as HTMLInputElement;
const filterControlEdges = document.getElementById('filter-control-edges') as HTMLInputElement;
const colorRegex = document.getElementById('color-regex') as HTMLInputElement;
const contextMenu = document.getElementById('context-menu') as HTMLDivElement;

// Apply View Options
filterControlEdges.addEventListener('change', () => {
  renderer.setFilterControlEdges(filterControlEdges.checked);
});

colorRegex.addEventListener('input', () => {
  renderer.setCustomColorRegex(colorRegex.value);
});
const menuExtractSubgraph = document.getElementById('menu-extract-subgraph') as HTMLDivElement;
const menuExtractPython = document.getElementById('menu-extract-python') as HTMLDivElement;
const menuReplaceConstant = document.getElementById('menu-replace-constant') as HTMLDivElement;
const menuExportPng = document.getElementById('menu-export-png') as HTMLDivElement;
const menuExportJson = document.getElementById('menu-export-json') as HTMLDivElement;
const menuCopyAttributes = document.getElementById('menu-copy-attributes') as HTMLDivElement;
const menuPasteAttributes = document.getElementById('menu-paste-attributes') as HTMLDivElement;
const menuDuplicateNode = document.getElementById('menu-duplicate-node') as HTMLDivElement;

let copiedAttributes: ReturnType<typeof JSON.parse> = null;
let copiedOpType: string | null = null;

// 134, 135. Support complete keyboard navigation and screen-reader announcements
document.addEventListener('keydown', (e) => {
  if (e.target !== document.body && e.target !== canvas) return; // Don't intercept inputs

  if (currentGraph && (renderer as ReturnType<typeof JSON.parse>).layout) {
    if (e.key === 'Tab') {
      e.preventDefault();
      const nodes = (renderer as ReturnType<typeof JSON.parse>).layout.nodes;
      if (nodes.length === 0) return;

      const currentSelection = renderer.selectedNodes.length > 0 ? renderer.selectedNodes[0] : null;
      let nextIdx = 0;
      if (currentSelection) {
        const currIdx = nodes.findIndex(
          (n: ReturnType<typeof JSON.parse>) => n.id === currentSelection,
        );
        if (currIdx !== -1) {
          nextIdx = (currIdx + (e.shiftKey ? -1 : 1) + nodes.length) % nodes.length;
        }
      }
      const nextNode = nodes[nextIdx];
      renderer.selectedNodes = [nextNode?.id];
      renderer.focusNode(nextNode?.id);
      renderSidebar(nextNode?.id);
      renderer.render();
      const announcer = document.getElementById('aria-announcer');
      if (announcer) announcer.textContent = 'Selected node ' + nextNode?.name;
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      if (renderer.selectedNodes.length > 0) {
        const mutator = new GraphMutator(currentGraph);
        renderer.selectedNodes.forEach((id) => {
          mutator.removeNode(id);
        });
        renderer.selectedNodes = [];
        renderSidebar(null);
        import('./layout/dag').then(({ computeLayout }) => {
          renderer.setLayout(computeLayout(currentGraph!, 'TB'));
        });
        const announcer = document.getElementById('aria-announcer');
        if (announcer) announcer.textContent = 'Deleted selected nodes.';
      }
    }
  }
});

canvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (currentGraph) {
    if (renderer.selectedNodes.length === 1) {
      const selectedId = renderer.selectedNodes[0];
      const node = currentGraph.nodes.find((n) => n.id === selectedId);

      menuCopyAttributes.style.display = node ? 'block' : 'none';
      if (node && copiedAttributes && copiedOpType === node.opType) {
        menuPasteAttributes.style.display = 'block';
      } else {
        menuPasteAttributes.style.display = 'none';
      }
      menuExtractSubgraph.style.display = 'block';
      menuExtractPython.style.display = 'block';
      menuReplaceConstant.style.display = 'block';
      menuExportJson.style.display = node ? 'block' : 'none';
      menuDuplicateNode.style.display = node ? 'block' : 'none';
    } else if (renderer.selectedNodes.length > 1) {
      menuCopyAttributes.style.display = 'none';
      menuPasteAttributes.style.display = 'none';
      menuExtractSubgraph.style.display = 'block';
      menuExtractPython.style.display = 'block';
      menuReplaceConstant.style.display = 'none';
      menuExportJson.style.display = 'none';
      menuDuplicateNode.style.display = 'none';
    } else {
      menuCopyAttributes.style.display = 'none';
      menuPasteAttributes.style.display = 'none';
      menuExtractSubgraph.style.display = 'none';
      menuExtractPython.style.display = 'none';
      menuReplaceConstant.style.display = 'none';
      menuExportJson.style.display = 'none';
      menuDuplicateNode.style.display = 'none';
    }

    contextMenu.style.display = 'block';
    contextMenu.style.left = `${e.clientX}px`;
    contextMenu.style.top = `${e.clientY}px`;
  }
});

document.addEventListener('click', (e) => {
  if (e.target !== contextMenu && !contextMenu.contains(e.target as HTMLElement)) {
    contextMenu.style.display = 'none';
  }
});

menuDuplicateNode.addEventListener('click', () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length !== 1) return;
  const selectedId = renderer.selectedNodes[0];
  const node = currentGraph.nodes.find((n) => n.id === selectedId);
  if (node) {
    const mutator = new GraphMutator(currentGraph);
    const newName = (node.name || node.id) + '_copy';
    const newOutputs = node.outputs.map((o) => (o ? o + '_copy' : ''));
    mutator.addNode(
      node.opType,
      [...node.inputs],
      newOutputs,
      JSON.parse(JSON.stringify(node.attributes)),
      newName,
    );
    // 178. Re-render
    statusDiv.textContent = 'Recalculating layout for duplicated node...';
    import('./layout/dag').then(({ computeLayout }) => {
      renderer.setLayout(computeLayout(currentGraph!, 'TB'));
      renderer.focusNode(newName);
      renderer.selectedNodes = [newName];
      renderSidebar(newName);
      statusDiv.textContent = 'Rendered Model: ' + currentGraph!.name;
    });
  }
});

menuExportPng.addEventListener('click', () => {
  contextMenu.style.display = 'none';
  if (!currentGraph) return;
  const dataUrl = canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `${currentGraph.name || 'model'}_visual.png`;
  a.click();
});

menuExportJson.addEventListener('click', () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length !== 1) return;
  const selectedId = renderer.selectedNodes[0];
  const node = currentGraph.nodes.find((n) => n.id === selectedId);
  if (node) {
    const dataStr =
      'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(node, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `${node.name || node.opType}.json`;
    a.click();
  }
});

menuCopyAttributes.addEventListener('click', () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length !== 1) return;
  const selectedId = renderer.selectedNodes[0];
  const node = currentGraph.nodes.find((n) => n.id === selectedId);
  if (node) {
    copiedAttributes = JSON.parse(JSON.stringify(node.attributes));
    copiedOpType = node.opType;
  }
});

menuPasteAttributes.addEventListener('click', () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length !== 1 || !copiedAttributes || !copiedOpType)
    return;
  const selectedId = renderer.selectedNodes[0];
  const node = currentGraph.nodes.find((n) => n.id === selectedId);
  if (node && node.opType === copiedOpType) {
    const mutator = new GraphMutator(currentGraph);
    for (const [k, v] of Object.entries(copiedAttributes)) {
      mutator.setNodeAttribute(
        node.name || node.id,
        k,
        (v as ReturnType<typeof JSON.parse>).value,
        (v as ReturnType<typeof JSON.parse>).type,
      );
    }
    renderSidebar(selectedId || null);
  }
});

menuExtractSubgraph.addEventListener('click', async () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length === 0) return;

  try {
    const mutator = new GraphMutator(currentGraph);
    const subGraph = mutator.extractSubgraph(renderer.selectedNodes);

    // We export using a temporary mutator
    const tempMutator = new GraphMutator(subGraph);
    const exporter = new ModelExporter(tempMutator);
    const data = await exporter.exportModel();

    exporter.downloadBlob('subgraph.onnx', data);
    alert(`Successfully extracted ${renderer.selectedNodes.length} nodes to subgraph.onnx`);
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    alert(`Extraction failed: ${err.message}`);
  }
});

menuExtractPython.addEventListener('click', async () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length === 0) return;

  try {
    const mutator = new GraphMutator(currentGraph);
    const subGraph = mutator.extractSubgraph(renderer.selectedNodes);
    const tempMutator = new GraphMutator(subGraph);
    const exporter = new ModelExporter(tempMutator);
    const script = exporter.generatePythonHelperScript();

    await navigator.clipboard.writeText(script);
    alert(`Python script for ${renderer.selectedNodes.length} nodes copied to clipboard!`);
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    alert(`Extraction failed: ${err.message}`);
  }
});

menuReplaceConstant.addEventListener('click', async () => {
  contextMenu.style.display = 'none';
  if (!currentGraph || renderer.selectedNodes.length === 0) return;

  alert(
    'Replace with Constant: This feature requires executing the node to obtain the constant value. Execution engine integration is currently mocked.',
  );
});

const btnHelp = document.getElementById('btn-help') as HTMLButtonElement;
const helpModal = document.getElementById('help-modal') as HTMLDivElement;
const closeHelpBtn = document.getElementById('close-help-btn') as HTMLButtonElement;

btnHelp.addEventListener('click', () => {
  helpModal.style.display = 'block';
});

closeHelpBtn.addEventListener('click', () => {
  helpModal.style.display = 'none';
});

const renderer = new CanvasRenderer(canvas);
const rootGraph: Graph | null = null;
let currentGraph: Graph | null = null;
const graphStack: Graph[] = [];
let currentSearchResults: string[] = [];
let currentSearchIndex: number = 0;

const breadcrumb = document.getElementById('breadcrumb') as HTMLDivElement;

breadcrumb.addEventListener('click', () => {
  if (graphStack.length > 0) {
    const parentGraph = graphStack.pop()!;
    currentGraph = parentGraph;
    if (graphStack.length === 0) {
      breadcrumb.style.display = 'none';
      statusDiv.textContent = 'Rendered Model: ' + currentGraph.name;
    } else {
      statusDiv.textContent = 'Rendered Subgraph: ' + currentGraph.name;
    }

    // We must re-run layout worker
    statusDiv.textContent = 'Calculating layout...';
    // Instead of passing a File, we pass the raw graph object to a special handler or just compute it here.
    // For now we'll send it back to the worker, but we need to modify the worker to accept a raw graph.
    // To keep it simple, we compute layout synchronously here:
    import('./layout/dag').then(({ computeLayout }) => {
      const layout = computeLayout(currentGraph!, 'TB');
      renderer.setLayout(layout);
      renderSidebar(null);
      statusDiv.textContent =
        'Rendered ' + (graphStack.length > 0 ? 'Subgraph' : 'Model') + ': ' + currentGraph!.name;
    });
  }
});

window.addEventListener('open-subgraph', (e: ReturnType<typeof JSON.parse>) => {
  if (currentGraph) graphStack.push(currentGraph);
  currentGraph = e.detail.graph as Graph;
  currentGraph.name = e.detail.name;

  breadcrumb.style.display = 'block';
  breadcrumb.innerHTML = `&larr; Back (Depth ${graphStack.length})`;

  statusDiv.textContent = 'Calculating layout for Subgraph...';
  import('./layout/dag').then(({ computeLayout }) => {
    const layout = computeLayout(currentGraph!, 'TB');
    renderer.setLayout(layout);
    renderSidebar(null);
    statusDiv.textContent = 'Rendered Subgraph: ' + currentGraph!.name;
  });
});

// Worker for layout computation

// Window level drag/drop
function handleFile(file: File) {
  if (file) {
    statusDiv.textContent = 'Parsing and calculating layout in Worker...';
    dropZone.textContent = `Loaded: ${file.name}`;
    worker.postMessage({ type: 'PARSE_FILE', file, direction: 'TB' });
  }
}

const searchResults = document.getElementById('search-results') as HTMLDivElement;
searchBox.addEventListener('input', (e) => {
  const query = searchBox.value.toLowerCase().trim();
  if (!query || !currentGraph) {
    currentSearchResults = [];
    renderer.setSearchResults([]);
    searchResults.textContent = '';
    return;
  }

  currentSearchResults = [];

  // Search logic (Node Name, Op Type, Tensor Name, Attribute Name/Value)
  for (const node of currentGraph.nodes) {
    if (node.name.toLowerCase().includes(query) || node.opType.toLowerCase().includes(query)) {
      currentSearchResults.push(node.id);
      continue;
    }
    // Check tensors (inputs/outputs)
    let found = false;
    for (const i of node.inputs) {
      if (i.toLowerCase().includes(query)) found = true;
    }
    for (const o of node.outputs) {
      if (o.toLowerCase().includes(query)) found = true;
    }
    if (found) {
      currentSearchResults.push(node.id);
      continue;
    }

    // Check attributes
    for (const [k, v] of Object.entries(node.attributes)) {
      if (k.toLowerCase().includes(query) || String(v.value).toLowerCase().includes(query)) {
        currentSearchResults.push(node.id);
        break;
      }
    }
  }

  // Also add inputs/outputs/constants to results if they match
  for (const i of currentGraph.inputs) {
    if (i.name.toLowerCase().includes(query)) currentSearchResults.push('input_' + i.name);
  }
  for (const o of currentGraph.outputs) {
    if (o.name.toLowerCase().includes(query)) currentSearchResults.push('output_' + o.name);
  }
  for (const init of currentGraph.initializers) {
    if (init.toLowerCase().includes(query)) currentSearchResults.push('const_' + init);
  }

  renderer.setSearchResults(currentSearchResults);
  searchResults.textContent =
    currentSearchResults.length > 0
      ? `Found ${currentSearchResults.length} items. Press Enter to step.`
      : 'No results found.';
  currentSearchIndex = -1;
});

searchBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && currentSearchResults.length > 0) {
    currentSearchIndex = (currentSearchIndex + 1) % currentSearchResults.length;
    const targetId = currentSearchResults[currentSearchIndex]!;
    renderer.focusNode(targetId);
    renderer.selectedNodes = [targetId];
    renderSidebar(targetId);
    renderer.render();
  }
});

// Worker for layout computation
const worker = new Worker(new URL('./parser/worker.ts', import.meta.url), { type: 'module' });

worker.onmessage = (e) => {
  if (e.data.type === 'PARSE_SUCCESS') {
    currentGraph = e.data.graph;
    statusDiv.textContent = 'Rendered Model: ' + currentGraph!.name;
    renderer.setLayout(e.data.layout);
    renderSidebar(null);
  } else if (e.data.type === 'PARSE_ERROR') {
    statusDiv.textContent = 'Error: ' + e.data.error;
    console.error(e.data.error);
  }
};

function renderSidebar(nodeId: string | null) {
  if (!currentGraph) {
    sidebar.style.display = 'none';
    return;
  }

  sidebar.style.display = 'block';

  if (!nodeId) {
    // Show Graph Properties
    let content = `<h2>Graph Properties</h2>`;
    content += `<b>Name:</b> ${currentGraph.name}<br/>`;
    content += `<b>Producer:</b> ${currentGraph.producerName} v${currentGraph.producerVersion}<br/>`;
    content += `<b>Domain:</b> ${currentGraph.domain || 'N/A'}<br/>`;
    content += `<b>Model Version:</b> ${currentGraph.modelVersion}<br/>`;

    content += `<hr style="border-color:var(--border)"/>`;
    content += `<h3>Opset Imports</h3>`;
    content += `<div id="opset-imports-container">`;
    for (const [domain, version] of Object.entries(currentGraph.opsetImports || {})) {
      content += `<div style="margin-bottom: 5px;">
        <input type="text" value="${domain}" disabled style="width: 120px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" /> : 
        <input type="number" class="opset-version-input" data-domain="${domain}" value="${version}" style="width: 60px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" />
      </div>`;
    }
    content += `</div>`;

    content += `<div style="margin-top: 10px;">
      <input type="text" id="new-opset-domain" placeholder="Domain (e.g., ai.onnx)" style="width: 120px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" />
      <input type="number" id="new-opset-version" placeholder="Version" style="width: 60px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" />
      <button id="add-opset-btn" style="background: #4A90E2; color: white; border: none; padding: 4px 8px; cursor: pointer;">Add</button>
    </div>
    
    <div style="margin-top: 20px;">
       <button id="btn-auto-format" style="width: 100%; padding: 8px; background: #555; color: white; border: 1px solid #777; cursor: pointer; border-radius: 4px;">Auto-Format Node Names</button>
    </div>`;

    sidebar.innerHTML = content;

    // Attach events
    const mutator = new GraphMutator(currentGraph);

    const versionInputs = sidebar.querySelectorAll('.opset-version-input');
    versionInputs.forEach((input) => {
      input.addEventListener('change', (e) => {
        const el = e.target as HTMLInputElement;
        const domain = el.getAttribute('data-domain')!;
        const newVersion = parseInt(el.value, 10);
        if (!isNaN(newVersion)) {
          currentGraph!.opsetImports[domain] = newVersion;
        }
      });
    });

    const addBtn = sidebar.querySelector('#add-opset-btn') as HTMLButtonElement;
    addBtn.addEventListener('click', () => {
      const domainInput = sidebar.querySelector('#new-opset-domain') as HTMLInputElement;
      const versionInput = sidebar.querySelector('#new-opset-version') as HTMLInputElement;
      const domain = domainInput.value.trim();
      const version = parseInt(versionInput.value, 10);
      if (domain && !isNaN(version)) {
        currentGraph!.opsetImports[domain] = version;
        renderSidebar(null); // Re-render
      }
    });

    const btnAutoFormat = sidebar.querySelector('#btn-auto-format') as HTMLButtonElement;
    if (btnAutoFormat) {
      btnAutoFormat.addEventListener('click', () => {
        import('@onnx9000/modifier/dist/components/utilities.js').then(({ ModifierUtilities }) => {
          const utils = new ModifierUtilities(mutator);
          utils.autoFormatNodeNames();

          // Re-layout and render
          statusDiv.textContent = 'Recalculating layout...';
          import('./layout/dag').then(({ computeLayout }) => {
            const layout = computeLayout(currentGraph!, 'TB');
            renderer.setLayout(layout);
            renderSidebar(null);
            statusDiv.textContent = 'Rendered Model: ' + currentGraph!.name;
          });
        });
      });
    }

    return;
  }

  // Find node
  let title = '';
  let content = '';

  if (nodeId.startsWith('input_')) {
    const name = nodeId.substring(6);
    const info = currentGraph.inputs.find((i) => i.name === name);
    title = 'Graph Input';
    content += `<b>Name:</b> ${name}<br/>`;
    if (info) {
      content += `<b>Type:</b> ${info.dtype}<br/>`;
      content += `<b>Shape:</b> [${info.shape.join(', ')}]<br/>`;
    }
  } else if (nodeId.startsWith('output_')) {
    const name = nodeId.substring(7);
    const info = currentGraph.outputs.find((o) => o.name === name);
    title = 'Graph Output';
    content += `<b>Name:</b> ${name}<br/>`;
    if (info) {
      content += `<b>Type:</b> ${info.dtype}<br/>`;
      content += `<b>Shape:</b> [${info.shape.join(', ')}]<br/>`;
    }
  } else if (nodeId.startsWith('const_')) {
    const name = nodeId.substring(6);
    const t = currentGraph.tensors[name];
    title = 'Initializer / Constant';
    content += `<b>Name:</b> ${name}<br/>`;
    if (t) {
      content += `<b>Type:</b> ${t.dtype}<br/>`;
      content += `<b>Shape:</b> [${t.shape.join(', ')}]<br/>`;

      if (t.data && t.data instanceof Uint8Array && t.dtype === 'float32') {
        const f32 = new Float32Array(t.data.buffer, t.data.byteOffset, Math.min(t.size, 100));
        let min = Infinity,
          max = -Infinity,
          sum = 0;
        const fullF32 = new Float32Array(t.data.buffer, t.data.byteOffset, t.size);
        for (let i = 0; i < t.size; i++) {
          const v = fullF32[i]!;
          if (v < min) min = v;
          if (v > max) max = v;
          sum += v;
        }
        const mean = sum / t.size;
        let vSum = 0;
        for (let i = 0; i < t.size; i++) {
          vSum += Math.pow(fullF32[i]! - mean, 2);
        }
        const variance = vSum / t.size;

        content += `<br/><b>Statistics:</b><br/>`;
        content += `Min: ${min.toFixed(4)}<br/>Max: ${max.toFixed(4)}<br/>Mean: ${mean.toFixed(4)}<br/>Variance: ${variance.toFixed(4)}<br/>`;

        // Generate heatmap for 2D/3D/4D weights if small enough
        if (t.shape.length >= 2) {
          const dims = t.shape as number[];
          const lastH = dims[dims.length - 2]!;
          const lastW = dims[dims.length - 1]!;
          const strideH = lastW;

          if (lastH > 0 && lastH <= 64 && lastW > 0 && lastW <= 64) {
            const hCanvas = document.createElement('canvas');
            const hScale = Math.max(1, Math.floor(200 / Math.max(lastH, lastW)));
            hCanvas.width = lastW * hScale;
            hCanvas.height = lastH * hScale;
            const hCtx = hCanvas.getContext('2d');
            if (hCtx) {
              const range = Math.max(1e-5, max - min);
              // Draw just the first slice (e.g., first out_channel, first in_channel)
              for (let y = 0; y < lastH; y++) {
                for (let x = 0; x < lastW; x++) {
                  const val = fullF32[y * strideH + x]!;
                  // Normalize to 0-1
                  const norm = (val - min) / range;

                  // Simple diverging colormap: min=blue, mid=black, max=red
                  let r = 0,
                    g = 0,
                    b = 0;
                  if (norm > 0.5) {
                    r = Math.floor((norm - 0.5) * 2 * 255);
                  } else {
                    b = Math.floor((0.5 - norm) * 2 * 255);
                  }

                  hCtx.fillStyle = `rgb(${r},${g},${b})`;
                  hCtx.fillRect(x * hScale, y * hScale, hScale, hScale);
                }
              }
              const dataUrl = hCanvas.toDataURL();
              content += `<br/><b>Heatmap (First Slice):</b><br/>`;
              content += `<img src="${dataUrl}" style="border:1px solid #555; image-rendering: pixelated;" /><br/>`;
            }
          }
        }

        content += `<br/><b>Data (Flattened):</b><br/>`;
        content += `<div style="max-height: 200px; overflow-y: auto; background: var(--input-bg); padding: 5px; font-size: 0.9em; word-break: break-all; border: 1px solid var(--border);">`;

        let matrixText = '';
        if (t.shape.length >= 2) {
          const rows = Math.min(t.shape[0] as number, 10);
          const cols = Math.min(t.shape[1] as number, 10);
          matrixText += `[\n`;
          for (let r = 0; r < rows; r++) {
            let rowStr = '  [';
            for (let c = 0; c < cols; c++) {
              rowStr +=
                fullF32[r * (t.shape[1] as number) + c]!.toFixed(4) + (c < cols - 1 ? ', ' : '');
            }
            rowStr += (cols < (t.shape[1] as number) ? ' ...' : '') + ']';
            matrixText += rowStr + (r < rows - 1 ? ',\n' : '\n');
          }
          matrixText += (rows < (t.shape[0] as number) ? '  ...\n' : '') + ']';
        } else {
          matrixText = `[${Array.from(f32)
            .map((x) => x.toFixed(4))
            .join(', ')}${t.size > 100 ? ' ...' : ''}]`;
        }

        content += `<pre style="margin:0; font-family: monospace;">${matrixText}</pre></div>`;
      }
    }
  } else {
    // Normal node
    const node = currentGraph.nodes.find((n) => n.id === nodeId);
    if (node) {
      title = node.opType;
      content += `<b>Name:</b> ${node.name || '(unnamed)'}<br/>`;
      content += `<b>Domain:</b> ${node.domain || 'ai.onnx'}<br/>`;
      // 253. Render Graph Doc Strings using Markdown in the UI (simplified regex replacement)
      if (node.docString) {
        let md = node.docString;
        md = md.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>'); // bold
        md = md.replace(/\*(.+?)\*/g, '<i>$1</i>'); // italic
        md = md.replace(/\n/g, '<br/>'); // newlines
        content += `<p style="font-size: 0.9em; background: var(--input-bg); border: 1px solid var(--border); padding: 5px; border-radius: 4px;">${md}</p>`;
      }

      content += `<hr style="border-color:var(--border)"/>`;

      content += `<b>Inputs:</b><ul style="padding-left: 20px; margin-top: 5px">`;
      node.inputs.forEach((i, idx) => {
        if (!i) {
          content += `<li><i>(optional/missing)</i></li>`;
        } else {
          // Find producer
          let producer = 'Graph Input';
          const pNode = currentGraph!.nodes.find((n) => n.outputs.includes(i));
          if (pNode) producer = pNode.name || pNode.opType;
          else if (currentGraph!.initializers.includes(i)) producer = 'Constant';
          content += `<li><b>${i}</b> <span style="color:var(--text-muted)">&larr; ${producer}</span></li>`;
        }
      });
      content += `</ul>`;

      content += `<b>Outputs:</b><ul style="padding-left: 20px; margin-top: 5px">`;
      node.outputs.forEach((o) => {
        if (!o) {
          content += `<li><i>(optional/missing)</i></li>`;
        } else {
          // Find consumers
          const consumers = currentGraph!.nodes
            .filter((n) => n.inputs.includes(o))
            .map((n) => n.name || n.opType);
          const cStr = consumers.length > 0 ? consumers.join(', ') : 'Graph Output';
          content += `<li><b>${o}</b> <span style="color:var(--text-muted)">&rarr; ${cStr}</span></li>`;
        }
      });
      content += `</ul>`;

      const attrKeys = Object.keys(node.attributes);
      if (attrKeys.length > 0) {
        content += `<hr style="border-color:var(--border)"/><b>Attributes:</b><br/>`;
        for (const k of attrKeys) {
          const a = node.attributes[k]!;
          let valStr = String(a.value);
          if (a.type === 'TENSOR') valStr = '[Tensor]';
          if (a.type === 'GRAPH') valStr = '[Graph]';
          content += `<i>${k}</i> (${a.type}): ${valStr}<br/>`;
        }
      }

      // 213. UI warnings for nodes using `double` (float64) precision
      let hasDouble = false;
      if (node.opType === 'Cast' && node.attributes['to'] && node.attributes['to'].value === 11)
        hasDouble = true;
      for (const i of node.inputs) {
        if (!i) continue;
        const info = currentGraph.inputs.find((vi) => vi.name === i) || currentGraph.tensors[i];
        if (info && info.dtype === 'float64') hasDouble = true;
      }
      if (hasDouble) {
        content += `<div style="margin-top: 10px; padding: 5px; background: #5a2e1e; border: 1px solid #ff9800; border-radius: 4px; font-size: 0.9em; color: #ffeb3b;">
          <b>⚠️ Performance Warning:</b> This node operates on \`float64\` (double) precision, which is extremely slow on WebGPU and unsupported on many mobile NPUs. Consider casting to \`float32\` or \`float16\`.
        </div>`;
      }
    }
  }

  sidebar.innerHTML = `<h2 style="margin-top:0">${title}</h2>${content}`;
}

renderer.onSelect = (nodeId) => {
  renderSidebar(nodeId);
  const announcer = document.getElementById('aria-announcer');
  if (announcer && nodeId) announcer.textContent = 'Selected node ' + nodeId;
};

// 229. Expose a global window.onnxModifier object for developer console hacking
(window as ReturnType<typeof JSON.parse>).onnxModifier = {
  getGraph: () => currentGraph,
  getRenderer: () => renderer,
  getMutator: () => new GraphMutator(currentGraph!),
  download: async () => {
    if (currentGraph) {
      const exporter = new ModelExporter(new GraphMutator(currentGraph));
      const data = await exporter.exportModel();
      exporter.downloadBlob('console_hacked.onnx', data);
    }
  },
};
