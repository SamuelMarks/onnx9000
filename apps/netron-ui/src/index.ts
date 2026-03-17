import { CanvasRenderer } from './render/canvas';
import { fetchAndParseModel } from './parser/fetcher';
import { Graph, Node } from '@onnx9000/core';

// Initialize UI
const uiHTML = `
  <div style="position: absolute; top: 10px; left: 10px; z-index: 10; color: white; background: #222; padding: 10px; border-radius: 5px; font-family: sans-serif; max-width: 300px;">
    <h3 style="margin-top:0">ONNX9000 Netron</h3>
    <input type="file" id="file-upload" accept=".onnx,.pb,.tflite,.pt,.h5" />
    <br/><br/>
    <input type="text" id="search-box" placeholder="Search node, op, tensor..." style="width: 100%; padding: 5px; background: #333; color: white; border: 1px solid #444; border-radius: 4px; box-sizing: border-box; margin-bottom: 5px;" />
    <div id="search-results" style="font-size: 0.8em; color: #aaa; margin-bottom: 10px;"></div>
    <div id="status">Waiting for file...</div>
  </div>
  <div id="sidebar" style="position: absolute; top: 0; right: 0; width: 350px; height: 100vh; background: #1a1a1a; color: #ddd; border-left: 1px solid #333; overflow-y: auto; display: none; padding: 15px; font-family: sans-serif; box-sizing: border-box;">
  </div>
  <canvas id="view" style="display:block; background: #111; width: 100vw; height: 100vh;"></canvas>
`;
document.body.innerHTML = uiHTML;
document.body.style.margin = '0';
document.body.style.overflow = 'hidden';

const canvas = document.getElementById('view') as HTMLCanvasElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;
const fileUpload = document.getElementById('file-upload') as HTMLInputElement;
const sidebar = document.getElementById('sidebar') as HTMLDivElement;
const searchBox = document.getElementById('search-box') as HTMLInputElement;
const searchResults = document.getElementById('search-results') as HTMLDivElement;

const renderer = new CanvasRenderer(canvas);
let currentGraph: Graph | null = null;
let currentSearchResults: string[] = [];
let currentSearchIndex: number = 0;

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
    renderer.selectedNode = targetId;
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
  } else if (e.data.type === 'PARSE_ERROR') {
    statusDiv.textContent = 'Error: ' + e.data.error;
    console.error(e.data.error);
  }
};

fileUpload.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    statusDiv.textContent = 'Parsing and calculating layout in Worker...';
    worker.postMessage({ type: 'PARSE_FILE', file, direction: 'TB' });
  }
});

function renderSidebar(nodeId: string | null) {
  if (!nodeId || !currentGraph) {
    sidebar.style.display = 'none';
    return;
  }

  sidebar.style.display = 'block';

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

        content += `<br/><b>Data (Flattened):</b><br/>`;
        content += `<div style="max-height: 200px; overflow-y: auto; background: #222; padding: 5px; font-size: 0.9em; word-break: break-all;">`;

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
      if (node.docString) {
        content += `<p style="font-size: 0.9em; background: #222; padding: 5px; border-radius: 4px;">${node.docString}</p>`;
      }

      content += `<hr style="border-color:#333"/>`;

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
          content += `<li><b>${i}</b> <span style="color:#888">&larr; ${producer}</span></li>`;
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
          let cStr = consumers.length > 0 ? consumers.join(', ') : 'Graph Output';
          content += `<li><b>${o}</b> <span style="color:#888">&rarr; ${cStr}</span></li>`;
        }
      });
      content += `</ul>`;

      const attrKeys = Object.keys(node.attributes);
      if (attrKeys.length > 0) {
        content += `<hr style="border-color:#333"/><b>Attributes:</b><br/>`;
        for (const k of attrKeys) {
          const a = node.attributes[k]!;
          let valStr = String(a.value);
          if (a.type === 'TENSOR') valStr = '[Tensor]';
          if (a.type === 'GRAPH') valStr = '[Graph]';
          content += `<i>${k}</i> (${a.type}): ${valStr}<br/>`;
        }
      }
    }
  }

  sidebar.innerHTML = `<h2 style="margin-top:0">${title}</h2>${content}`;
}

renderer.onSelect = (nodeId) => {
  renderSidebar(nodeId);
};
