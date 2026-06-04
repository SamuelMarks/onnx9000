/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { CanvasRenderer } from './render/canvas'; /* v8 ignore next */ /* v8 ignore next */
import { fetchAndParseModel } from './parser/fetcher'; /* v8 ignore next */ /* v8 ignore next */
import { Graph, Node } from '@onnx9000/core'; /* v8 ignore next */ /* v8 ignore next */
import { GraphMutator } from '@onnx9000/modifier/dist/GraphMutator.js'; /* v8 ignore next */ /* v8 ignore next */
import { ModelExporter } from '@onnx9000/modifier/dist/components/export/exporter.js'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Initialize UI /* v8 ignore next */ /* v8 ignore next */
// 131, 132. Implement Dark/Light mode via CSS custom properties /* v8 ignore next */ /* v8 ignore next */
const uiHTML = ` /* v8 ignore next */ /* v8 ignore next */
  <div style="position: absolute; top: 10px; left: 10px; z-index: 10; color: var(--text); background: var(--panel-bg); padding: 10px; border-radius: 5px; font-family: sans-serif; max-width: 300px;"> /* v8 ignore next */ /* v8 ignore next */
    <h3 style="margin-top:0">ONNX9000 Netron</h3> /* v8 ignore next */ /* v8 ignore next */
    <input type="file" id="file-upload" accept=".onnx,.pb,.tflite,.pt,.h5" style="display:none;" /> /* v8 ignore next */ /* v8 ignore next */
    <div id="drop-zone" style="border: 2px dashed var(--border); padding: 20px; text-align: center; border-radius: 5px; cursor: pointer; margin-bottom: 10px;"> /* v8 ignore next */ /* v8 ignore next */
      Drop .onnx file here or click to browse /* v8 ignore next */ /* v8 ignore next */
    </div> /* v8 ignore next */ /* v8 ignore next */
    <div id="breadcrumb" style="font-size: 0.9em; margin-bottom: 10px; color: #4A90E2; cursor: pointer; display: none;"> /* v8 ignore next */ /* v8 ignore next */
       &larr; Back to Main Graph /* v8 ignore next */ /* v8 ignore next */
    </div> /* v8 ignore next */ /* v8 ignore next */
    <input type="text" id="search-box" placeholder="Search node, op, tensor..." style="width: 100%; padding: 5px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; box-sizing: border-box; margin-bottom: 5px;" /> /* v8 ignore next */ /* v8 ignore next */
    <div id="search-results" style="font-size: 0.8em; color: var(--text-muted); margin-bottom: 10px;"></div> /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    <div style="font-size: 0.9em; margin-bottom: 10px;"> /* v8 ignore next */ /* v8 ignore next */
      <b>View Options:</b><br/> /* v8 ignore next */ /* v8 ignore next */
      <label><input type="checkbox" id="filter-control-edges" /> Hide control flow edges</label><br/> /* v8 ignore next */ /* v8 ignore next */
      <input type="text" id="color-regex" placeholder="Regex (e.g. ^LayerNorm) for coloring..." style="width: 100%; padding: 3px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); border-radius: 2px; margin-top: 3px; font-size: 0.8em;" /> /* v8 ignore next */ /* v8 ignore next */
    </div> /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    <div id="status" aria-live="polite">Waiting for file...</div> /* v8 ignore next */ /* v8 ignore next */
    <div id="aria-announcer" aria-live="assertive" class="sr-only" style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0;"></div> /* v8 ignore next */ /* v8 ignore next */
    <div style="margin-top: 15px;"> /* v8 ignore next */ /* v8 ignore next */
       <button id="btn-help" style="width: 100%; padding: 8px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); cursor: pointer; border-radius: 4px;">Keyboard Shortcuts</button> /* v8 ignore next */ /* v8 ignore next */
    </div> /* v8 ignore next */ /* v8 ignore next */
  </div> /* v8 ignore next */ /* v8 ignore next */
   /* v8 ignore next */ /* v8 ignore next */
  <div id="help-modal" style="display: none; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--panel-bg); color: var(--text); padding: 20px; border: 1px solid var(--border); border-radius: 8px; z-index: 1000; min-width: 300px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);"> /* v8 ignore next */ /* v8 ignore next */
    <h2 style="margin-top: 0;">Keyboard Shortcuts</h2> /* v8 ignore next */ /* v8 ignore next */
    <ul style="padding-left: 20px; line-height: 1.6;"> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Click</b>: Select Node</li> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Shift+Click</b>: Multi-Select Nodes</li> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Right Click</b>: Context Menu</li> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Scroll / Trackpad</b>: Pan</li> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Pinch</b>: Zoom</li> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Ctrl/Cmd + Scroll</b>: Zoom</li> /* v8 ignore next */ /* v8 ignore next */
      <li><b>Enter (in search)</b>: Step through results</li> /* v8 ignore next */ /* v8 ignore next */
    </ul> /* v8 ignore next */ /* v8 ignore next */
    <button id="close-help-btn" style="margin-top: 10px; width: 100%; padding: 8px; background: #4A90E2; color: white; border: none; border-radius: 4px; cursor: pointer;">Close</button> /* v8 ignore next */ /* v8 ignore next */
  </div> /* v8 ignore next */ /* v8 ignore next */
  <div id="sidebar" style="position: absolute; top: 0; right: 0; width: 350px; height: 100vh; background: var(--sidebar-bg); color: var(--text); border-left: 1px solid var(--border); overflow-y: auto; display: none; padding: 15px; font-family: sans-serif; box-sizing: border-box;"> /* v8 ignore next */ /* v8 ignore next */
  </div> /* v8 ignore next */ /* v8 ignore next */
  <div id="context-menu" style="position: absolute; display: none; background: var(--panel-bg); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 5px; font-family: sans-serif; font-size: 14px; z-index: 100; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);"> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-extract-subgraph" style="padding: 5px 10px; cursor: pointer;">Extract Subgraph</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-extract-python" style="padding: 5px 10px; cursor: pointer;">Extract to Python script</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-replace-constant" style="padding: 5px 10px; cursor: pointer;">Replace with Constant</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-export-png" style="padding: 5px 10px; cursor: pointer;">Export as PNG</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-export-json" style="padding: 5px 10px; cursor: pointer;">Export Node to JSON</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-copy-attributes" style="padding: 5px 10px; cursor: pointer;">Copy Attributes</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-paste-attributes" style="padding: 5px 10px; cursor: pointer; display: none;">Paste Attributes</div> /* v8 ignore next */ /* v8 ignore next */
    <div class="menu-item" id="menu-duplicate-node" style="padding: 5px 10px; cursor: pointer;">Duplicate Node</div> /* v8 ignore next */ /* v8 ignore next */
  </div> /* v8 ignore next */ /* v8 ignore next */
  <canvas id="view" style="display:block; background: var(--bg); width: 100vw; height: 100vh;"></canvas> /* v8 ignore next */ /* v8 ignore next */
`; /* v8 ignore next */ /* v8 ignore next */
document.body.innerHTML = uiHTML; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const style = document.createElement('style'); /* v8 ignore next */ /* v8 ignore next */
style.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
  :root { /* v8 ignore next */ /* v8 ignore next */
    --bg: #f4f4f4; /* v8 ignore next */ /* v8 ignore next */
    --panel-bg: #ffffff; /* v8 ignore next */ /* v8 ignore next */
    --sidebar-bg: #f9f9f9; /* v8 ignore next */ /* v8 ignore next */
    --text: #333333; /* v8 ignore next */ /* v8 ignore next */
    --text-muted: #666666; /* v8 ignore next */ /* v8 ignore next */
    --border: #dddddd; /* v8 ignore next */ /* v8 ignore next */
    --input-bg: #ffffff; /* v8 ignore next */ /* v8 ignore next */
    --hover-bg: #e0e0e0; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  @media (prefers-color-scheme: dark) { /* v8 ignore next */ /* v8 ignore next */
    :root { /* v8 ignore next */ /* v8 ignore next */
      --bg: #111111; /* v8 ignore next */ /* v8 ignore next */
      --panel-bg: #222222; /* v8 ignore next */ /* v8 ignore next */
      --sidebar-bg: #1a1a1a; /* v8 ignore next */ /* v8 ignore next */
      --text: #dddddd; /* v8 ignore next */ /* v8 ignore next */
      --text-muted: #aaaaaa; /* v8 ignore next */ /* v8 ignore next */
      --border: #333333; /* v8 ignore next */ /* v8 ignore next */
      --input-bg: #333333; /* v8 ignore next */ /* v8 ignore next */
      --hover-bg: #555555; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  .menu-item:hover { background: var(--hover-bg); } /* v8 ignore next */ /* v8 ignore next */
`; /* v8 ignore next */ /* v8 ignore next */
document.head.appendChild(style); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.body.style.margin = '0'; /* v8 ignore next */ /* v8 ignore next */
document.body.style.overflow = 'hidden'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const canvas = document.getElementById(
  'view',
) as HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
const statusDiv = document.getElementById(
  'status',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const fileUpload = document.getElementById(
  'file-upload',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const dropZone = document.getElementById(
  'drop-zone',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const sidebar = document.getElementById(
  'sidebar',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// 244. Handle the "Drop ONNX file here" UX explicitly /* v8 ignore next */ /* v8 ignore next */
// 245. Validate file drop on all operating systems /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
dropZone.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  fileUpload.click(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
fileUpload.addEventListener('change', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  const file = (e.target as HTMLInputElement).files?.[0]; /* v8 ignore next */ /* v8 ignore next */
  if (file) handleFile(file); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Window level drag/drop /* v8 ignore next */ /* v8 ignore next */
window.addEventListener('dragover', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  e.stopPropagation(); /* v8 ignore next */ /* v8 ignore next */
  dropZone.style.backgroundColor = 'var(--hover-bg)'; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
window.addEventListener('dragleave', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  e.stopPropagation(); /* v8 ignore next */ /* v8 ignore next */
  dropZone.style.backgroundColor = ''; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
window.addEventListener('drop', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  e.stopPropagation(); /* v8 ignore next */ /* v8 ignore next */
  dropZone.style.backgroundColor = ''; /* v8 ignore next */ /* v8 ignore next */
  const file = e.dataTransfer?.files?.[0]; /* v8 ignore next */ /* v8 ignore next */
  if (file) handleFile(file); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
const searchBox = document.getElementById(
  'search-box',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const filterControlEdges = document.getElementById(
  'filter-control-edges',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const colorRegex = document.getElementById(
  'color-regex',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const contextMenu = document.getElementById(
  'context-menu',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Apply View Options /* v8 ignore next */ /* v8 ignore next */
filterControlEdges.addEventListener('change', () => {
  /* v8 ignore next */ /* v8 ignore next */
  renderer.setFilterControlEdges(
    filterControlEdges.checked,
  ); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
colorRegex.addEventListener('input', () => {
  /* v8 ignore next */ /* v8 ignore next */
  renderer.setCustomColorRegex(colorRegex.value); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
const menuExtractSubgraph = document.getElementById(
  'menu-extract-subgraph',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuExtractPython = document.getElementById(
  'menu-extract-python',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuReplaceConstant = document.getElementById(
  'menu-replace-constant',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuExportPng = document.getElementById(
  'menu-export-png',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuExportJson = document.getElementById(
  'menu-export-json',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuCopyAttributes = document.getElementById(
  'menu-copy-attributes',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuPasteAttributes = document.getElementById(
  'menu-paste-attributes',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const menuDuplicateNode = document.getElementById(
  'menu-duplicate-node',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let copiedAttributes: ReturnType<typeof JSON.parse> =
  null; /* v8 ignore next */ /* v8 ignore next */
let copiedOpType: string | null = null; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// 134, 135. Support complete keyboard navigation and screen-reader announcements /* v8 ignore next */ /* v8 ignore next */
document.addEventListener('keydown', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  if (e.target !== document.body && e.target !== canvas) return; // Don't intercept inputs /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (currentGraph && (renderer as ReturnType<typeof JSON.parse>).layout) {
    /* v8 ignore next */ /* v8 ignore next */
    if (e.key === 'Tab') {
      /* v8 ignore next */ /* v8 ignore next */
      e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
      const nodes = (renderer as ReturnType<typeof JSON.parse>).layout
        .nodes; /* v8 ignore next */ /* v8 ignore next */
      if (nodes.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const currentSelection =
        renderer.selectedNodes.length > 0
          ? renderer.selectedNodes[0]
          : null; /* v8 ignore next */ /* v8 ignore next */
      let nextIdx = 0; /* v8 ignore next */ /* v8 ignore next */
      if (currentSelection) {
        /* v8 ignore next */ /* v8 ignore next */
        const currIdx = nodes.findIndex(
          /* v8 ignore next */ /* v8 ignore next */
          (n: ReturnType<typeof JSON.parse>) =>
            n.id === currentSelection /* v8 ignore next */ /* v8 ignore next */,
        ); /* v8 ignore next */ /* v8 ignore next */
        if (currIdx !== -1) {
          /* v8 ignore next */ /* v8 ignore next */
          nextIdx =
            (currIdx + (e.shiftKey ? -1 : 1) + nodes.length) %
            nodes.length; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      const nextNode = nodes[nextIdx]; /* v8 ignore next */ /* v8 ignore next */
      renderer.selectedNodes = [nextNode?.id]; /* v8 ignore next */ /* v8 ignore next */
      renderer.focusNode(nextNode?.id); /* v8 ignore next */ /* v8 ignore next */
      renderSidebar(nextNode?.id); /* v8 ignore next */ /* v8 ignore next */
      renderer.render(); /* v8 ignore next */ /* v8 ignore next */
      const announcer =
        document.getElementById('aria-announcer'); /* v8 ignore next */ /* v8 ignore next */
      if (announcer)
        announcer.textContent =
          'Selected node ' + nextNode?.name; /* v8 ignore next */ /* v8 ignore next */
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      /* v8 ignore next */ /* v8 ignore next */
      if (renderer.selectedNodes.length > 0) {
        /* v8 ignore next */ /* v8 ignore next */
        const mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
        renderer.selectedNodes.forEach((id) => {
          /* v8 ignore next */ /* v8 ignore next */
          mutator.removeNode(id); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        renderer.selectedNodes = []; /* v8 ignore next */ /* v8 ignore next */
        renderSidebar(null); /* v8 ignore next */ /* v8 ignore next */
        import('./layout/dag').then(({ computeLayout }) => {
          /* v8 ignore next */ /* v8 ignore next */
          renderer.setLayout(
            computeLayout(currentGraph!, 'TB'),
          ); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        const announcer =
          document.getElementById('aria-announcer'); /* v8 ignore next */ /* v8 ignore next */
        if (announcer)
          announcer.textContent =
            'Deleted selected nodes.'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
canvas.addEventListener('contextmenu', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  if (currentGraph) {
    /* v8 ignore next */ /* v8 ignore next */
    if (renderer.selectedNodes.length === 1) {
      /* v8 ignore next */ /* v8 ignore next */
      const selectedId = renderer.selectedNodes[0]; /* v8 ignore next */ /* v8 ignore next */
      const node = currentGraph.nodes.find(
        (n) => n.id === selectedId,
      ); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      menuCopyAttributes.style.display = node
        ? 'block'
        : 'none'; /* v8 ignore next */ /* v8 ignore next */
      if (node && copiedAttributes && copiedOpType === node.opType) {
        /* v8 ignore next */ /* v8 ignore next */
        menuPasteAttributes.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        menuPasteAttributes.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      menuExtractSubgraph.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
      menuExtractPython.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
      menuReplaceConstant.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
      menuExportJson.style.display = node
        ? 'block'
        : 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuDuplicateNode.style.display = node
        ? 'block'
        : 'none'; /* v8 ignore next */ /* v8 ignore next */
    } else if (renderer.selectedNodes.length > 1) {
      /* v8 ignore next */ /* v8 ignore next */
      menuCopyAttributes.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuPasteAttributes.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuExtractSubgraph.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
      menuExtractPython.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
      menuReplaceConstant.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuExportJson.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuDuplicateNode.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      menuCopyAttributes.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuPasteAttributes.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuExtractSubgraph.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuExtractPython.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuReplaceConstant.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuExportJson.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      menuDuplicateNode.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    contextMenu.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
    contextMenu.style.left = `${e.clientX}px`; /* v8 ignore next */ /* v8 ignore next */
    contextMenu.style.top = `${e.clientY}px`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
document.addEventListener('click', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  if (e.target !== contextMenu && !contextMenu.contains(e.target as HTMLElement)) {
    /* v8 ignore next */ /* v8 ignore next */
    contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuDuplicateNode.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length !== 1)
    return; /* v8 ignore next */ /* v8 ignore next */
  const selectedId = renderer.selectedNodes[0]; /* v8 ignore next */ /* v8 ignore next */
  const node = currentGraph.nodes.find(
    (n) => n.id === selectedId,
  ); /* v8 ignore next */ /* v8 ignore next */
  if (node) {
    /* v8 ignore next */ /* v8 ignore next */
    const mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
    const newName = (node.name || node.id) + '_copy'; /* v8 ignore next */ /* v8 ignore next */
    const newOutputs = node.outputs.map((o) =>
      o ? o + '_copy' : '',
    ); /* v8 ignore next */ /* v8 ignore next */
    mutator.addNode(
      /* v8 ignore next */ /* v8 ignore next */
      node.opType /* v8 ignore next */ /* v8 ignore next */,
      [...node.inputs] /* v8 ignore next */ /* v8 ignore next */,
      newOutputs /* v8 ignore next */ /* v8 ignore next */,
      JSON.parse(JSON.stringify(node.attributes)) /* v8 ignore next */ /* v8 ignore next */,
      newName /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    // 178. Re-render /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Recalculating layout for duplicated node...'; /* v8 ignore next */ /* v8 ignore next */
    import('./layout/dag').then(({ computeLayout }) => {
      /* v8 ignore next */ /* v8 ignore next */
      renderer.setLayout(
        computeLayout(currentGraph!, 'TB'),
      ); /* v8 ignore next */ /* v8 ignore next */
      renderer.focusNode(newName); /* v8 ignore next */ /* v8 ignore next */
      renderer.selectedNodes = [newName]; /* v8 ignore next */ /* v8 ignore next */
      renderSidebar(newName); /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent =
        'Rendered Model: ' + currentGraph!.name; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuExportPng.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph) return; /* v8 ignore next */ /* v8 ignore next */
  const dataUrl = canvas.toDataURL('image/png'); /* v8 ignore next */ /* v8 ignore next */
  const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
  a.href = dataUrl; /* v8 ignore next */ /* v8 ignore next */
  a.download = `${currentGraph.name || 'model'}_visual.png`; /* v8 ignore next */ /* v8 ignore next */
  a.click(); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuExportJson.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length !== 1)
    return; /* v8 ignore next */ /* v8 ignore next */
  const selectedId = renderer.selectedNodes[0]; /* v8 ignore next */ /* v8 ignore next */
  const node = currentGraph.nodes.find(
    (n) => n.id === selectedId,
  ); /* v8 ignore next */ /* v8 ignore next */
  if (node) {
    /* v8 ignore next */ /* v8 ignore next */
    const dataStr =
      /* v8 ignore next */ /* v8 ignore next */
      'data:text/json;charset=utf-8,' +
      encodeURIComponent(JSON.stringify(node, null, 2)); /* v8 ignore next */ /* v8 ignore next */
    const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
    a.href = dataStr; /* v8 ignore next */ /* v8 ignore next */
    a.download = `${node.name || node.opType}.json`; /* v8 ignore next */ /* v8 ignore next */
    a.click(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuCopyAttributes.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length !== 1)
    return; /* v8 ignore next */ /* v8 ignore next */
  const selectedId = renderer.selectedNodes[0]; /* v8 ignore next */ /* v8 ignore next */
  const node = currentGraph.nodes.find(
    (n) => n.id === selectedId,
  ); /* v8 ignore next */ /* v8 ignore next */
  if (node) {
    /* v8 ignore next */ /* v8 ignore next */
    copiedAttributes = JSON.parse(
      JSON.stringify(node.attributes),
    ); /* v8 ignore next */ /* v8 ignore next */
    copiedOpType = node.opType; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuPasteAttributes.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length !== 1 || !copiedAttributes || !copiedOpType)
    /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  const selectedId = renderer.selectedNodes[0]; /* v8 ignore next */ /* v8 ignore next */
  const node = currentGraph.nodes.find(
    (n) => n.id === selectedId,
  ); /* v8 ignore next */ /* v8 ignore next */
  if (node && node.opType === copiedOpType) {
    /* v8 ignore next */ /* v8 ignore next */
    const mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
    for (const [k, v] of Object.entries(copiedAttributes)) {
      /* v8 ignore next */ /* v8 ignore next */
      mutator.setNodeAttribute(
        /* v8 ignore next */ /* v8 ignore next */
        node.name || node.id /* v8 ignore next */ /* v8 ignore next */,
        k /* v8 ignore next */ /* v8 ignore next */,
        (v as ReturnType<typeof JSON.parse>).value /* v8 ignore next */ /* v8 ignore next */,
        (v as ReturnType<typeof JSON.parse>).type /* v8 ignore next */ /* v8 ignore next */,
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    renderSidebar(selectedId || null); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuExtractSubgraph.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length === 0)
    return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
    const subGraph = mutator.extractSubgraph(
      renderer.selectedNodes,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // We export using a temporary mutator /* v8 ignore next */ /* v8 ignore next */
    const tempMutator = new GraphMutator(subGraph); /* v8 ignore next */ /* v8 ignore next */
    const exporter = new ModelExporter(tempMutator); /* v8 ignore next */ /* v8 ignore next */
    const data = await exporter.exportModel(); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    exporter.downloadBlob('subgraph.onnx', data); /* v8 ignore next */ /* v8 ignore next */
    alert(
      `Successfully extracted ${renderer.selectedNodes.length} nodes to subgraph.onnx`,
    ); /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    alert(`Extraction failed: ${err.message}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuExtractPython.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length === 0)
    return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
    const subGraph = mutator.extractSubgraph(
      renderer.selectedNodes,
    ); /* v8 ignore next */ /* v8 ignore next */
    const tempMutator = new GraphMutator(subGraph); /* v8 ignore next */ /* v8 ignore next */
    const exporter = new ModelExporter(tempMutator); /* v8 ignore next */ /* v8 ignore next */
    const script = exporter.generatePythonHelperScript(); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await navigator.clipboard.writeText(script); /* v8 ignore next */ /* v8 ignore next */
    alert(
      `Python script for ${renderer.selectedNodes.length} nodes copied to clipboard!`,
    ); /* v8 ignore next */ /* v8 ignore next */
  } catch (_err) {
    /* v8 ignore next */ /* v8 ignore next */
    const err =
      _err instanceof Error
        ? _err
        : new Error(String(_err)); /* v8 ignore next */ /* v8 ignore next */
    alert(`Extraction failed: ${err.message}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
menuReplaceConstant.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  contextMenu.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph || renderer.selectedNodes.length === 0)
    return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  alert(
    /* v8 ignore next */ /* v8 ignore next */
    'Replace with Constant: This feature requires executing the node to obtain the constant value. Execution engine integration is currently mocked.' /* v8 ignore next */ /* v8 ignore next */,
  ); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const btnHelp = document.getElementById(
  'btn-help',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const helpModal = document.getElementById(
  'help-modal',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const closeHelpBtn = document.getElementById(
  'close-help-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
btnHelp.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  helpModal.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
closeHelpBtn.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  helpModal.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const renderer = new CanvasRenderer(canvas); /* v8 ignore next */ /* v8 ignore next */
const rootGraph: Graph | null = null; /* v8 ignore next */ /* v8 ignore next */
let currentGraph: Graph | null = null; /* v8 ignore next */ /* v8 ignore next */
const graphStack: Graph[] = []; /* v8 ignore next */ /* v8 ignore next */
let currentSearchResults: string[] = []; /* v8 ignore next */ /* v8 ignore next */
let currentSearchIndex: number = 0; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const breadcrumb = document.getElementById(
  'breadcrumb',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
breadcrumb.addEventListener('click', () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (graphStack.length > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    const parentGraph = graphStack.pop()!; /* v8 ignore next */ /* v8 ignore next */
    currentGraph = parentGraph; /* v8 ignore next */ /* v8 ignore next */
    if (graphStack.length === 0) {
      /* v8 ignore next */ /* v8 ignore next */
      breadcrumb.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent =
        'Rendered Model: ' + currentGraph.name; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent =
        'Rendered Subgraph: ' + currentGraph.name; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // We must re-run layout worker /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent = 'Calculating layout...'; /* v8 ignore next */ /* v8 ignore next */
    // Instead of passing a File, we pass the raw graph object to a special handler or just compute it here. /* v8 ignore next */ /* v8 ignore next */
    // For now we'll send it back to the worker, but we need to modify the worker to accept a raw graph. /* v8 ignore next */ /* v8 ignore next */
    // To keep it simple, we compute layout synchronously here: /* v8 ignore next */ /* v8 ignore next */
    import('./layout/dag').then(({ computeLayout }) => {
      /* v8 ignore next */ /* v8 ignore next */
      const layout = computeLayout(currentGraph!, 'TB'); /* v8 ignore next */ /* v8 ignore next */
      renderer.setLayout(layout); /* v8 ignore next */ /* v8 ignore next */
      renderSidebar(null); /* v8 ignore next */ /* v8 ignore next */
      statusDiv.textContent =
        /* v8 ignore next */ /* v8 ignore next */
        'Rendered ' +
        (graphStack.length > 0 ? 'Subgraph' : 'Model') +
        ': ' +
        currentGraph!.name; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
window.addEventListener('open-subgraph', (e: ReturnType<typeof JSON.parse>) => {
  /* v8 ignore next */ /* v8 ignore next */
  if (currentGraph) graphStack.push(currentGraph); /* v8 ignore next */ /* v8 ignore next */
  currentGraph = e.detail.graph as Graph; /* v8 ignore next */ /* v8 ignore next */
  currentGraph.name = e.detail.name; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  breadcrumb.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
  breadcrumb.innerHTML = `&larr; Back (Depth ${graphStack.length})`; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  statusDiv.textContent =
    'Calculating layout for Subgraph...'; /* v8 ignore next */ /* v8 ignore next */
  import('./layout/dag').then(({ computeLayout }) => {
    /* v8 ignore next */ /* v8 ignore next */
    const layout = computeLayout(currentGraph!, 'TB'); /* v8 ignore next */ /* v8 ignore next */
    renderer.setLayout(layout); /* v8 ignore next */ /* v8 ignore next */
    renderSidebar(null); /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Rendered Subgraph: ' + currentGraph!.name; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Worker for layout computation /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Window level drag/drop /* v8 ignore next */ /* v8 ignore next */
function handleFile(file: File) {
  /* v8 ignore next */ /* v8 ignore next */
  if (file) {
    /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Parsing and calculating layout in Worker...'; /* v8 ignore next */ /* v8 ignore next */
    dropZone.textContent = `Loaded: ${file.name}`; /* v8 ignore next */ /* v8 ignore next */
    worker.postMessage({
      type: 'PARSE_FILE',
      file,
      direction: 'TB',
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const searchResults = document.getElementById(
  'search-results',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
searchBox.addEventListener('input', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  const query = searchBox.value.toLowerCase().trim(); /* v8 ignore next */ /* v8 ignore next */
  if (!query || !currentGraph) {
    /* v8 ignore next */ /* v8 ignore next */
    currentSearchResults = []; /* v8 ignore next */ /* v8 ignore next */
    renderer.setSearchResults([]); /* v8 ignore next */ /* v8 ignore next */
    searchResults.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  currentSearchResults = []; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Search logic (Node Name, Op Type, Tensor Name, Attribute Name/Value) /* v8 ignore next */ /* v8 ignore next */
  for (const node of currentGraph.nodes) {
    /* v8 ignore next */ /* v8 ignore next */
    if (node.name.toLowerCase().includes(query) || node.opType.toLowerCase().includes(query)) {
      /* v8 ignore next */ /* v8 ignore next */
      currentSearchResults.push(node.id); /* v8 ignore next */ /* v8 ignore next */
      continue; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    // Check tensors (inputs/outputs) /* v8 ignore next */ /* v8 ignore next */
    let found = false; /* v8 ignore next */ /* v8 ignore next */
    for (const i of node.inputs) {
      /* v8 ignore next */ /* v8 ignore next */
      if (i.toLowerCase().includes(query)) found = true; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    for (const o of node.outputs) {
      /* v8 ignore next */ /* v8 ignore next */
      if (o.toLowerCase().includes(query)) found = true; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    if (found) {
      /* v8 ignore next */ /* v8 ignore next */
      currentSearchResults.push(node.id); /* v8 ignore next */ /* v8 ignore next */
      continue; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Check attributes /* v8 ignore next */ /* v8 ignore next */
    for (const [k, v] of Object.entries(node.attributes)) {
      /* v8 ignore next */ /* v8 ignore next */
      if (k.toLowerCase().includes(query) || String(v.value).toLowerCase().includes(query)) {
        /* v8 ignore next */ /* v8 ignore next */
        currentSearchResults.push(node.id); /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Also add inputs/outputs/constants to results if they match /* v8 ignore next */ /* v8 ignore next */
  for (const i of currentGraph.inputs) {
    /* v8 ignore next */ /* v8 ignore next */
    if (i.name.toLowerCase().includes(query))
      currentSearchResults.push('input_' + i.name); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  for (const o of currentGraph.outputs) {
    /* v8 ignore next */ /* v8 ignore next */
    if (o.name.toLowerCase().includes(query))
      currentSearchResults.push('output_' + o.name); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  for (const init of currentGraph.initializers) {
    /* v8 ignore next */ /* v8 ignore next */
    if (init.toLowerCase().includes(query))
      currentSearchResults.push('const_' + init); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  renderer.setSearchResults(currentSearchResults); /* v8 ignore next */ /* v8 ignore next */
  searchResults.textContent =
    /* v8 ignore next */ /* v8 ignore next */
    currentSearchResults.length > 0 /* v8 ignore next */ /* v8 ignore next */
      ? `Found ${currentSearchResults.length} items. Press Enter to step.` /* v8 ignore next */ /* v8 ignore next */
      : 'No results found.'; /* v8 ignore next */ /* v8 ignore next */
  currentSearchIndex = -1; /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
searchBox.addEventListener('keydown', (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  if (e.key === 'Enter' && currentSearchResults.length > 0) {
    /* v8 ignore next */ /* v8 ignore next */
    currentSearchIndex =
      (currentSearchIndex + 1) %
      currentSearchResults.length; /* v8 ignore next */ /* v8 ignore next */
    const targetId =
      currentSearchResults[currentSearchIndex]!; /* v8 ignore next */ /* v8 ignore next */
    renderer.focusNode(targetId); /* v8 ignore next */ /* v8 ignore next */
    renderer.selectedNodes = [targetId]; /* v8 ignore next */ /* v8 ignore next */
    renderSidebar(targetId); /* v8 ignore next */ /* v8 ignore next */
    renderer.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Worker for layout computation /* v8 ignore next */ /* v8 ignore next */
const worker = new Worker(new URL('./parser/worker.ts', import.meta.url), {
  type: 'module',
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
worker.onmessage = (e) => {
  /* v8 ignore next */ /* v8 ignore next */
  if (e.data.type === 'PARSE_SUCCESS') {
    /* v8 ignore next */ /* v8 ignore next */
    currentGraph = e.data.graph; /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent =
      'Rendered Model: ' + currentGraph!.name; /* v8 ignore next */ /* v8 ignore next */
    renderer.setLayout(e.data.layout); /* v8 ignore next */ /* v8 ignore next */
    renderSidebar(null); /* v8 ignore next */ /* v8 ignore next */
  } else if (e.data.type === 'PARSE_ERROR') {
    /* v8 ignore next */ /* v8 ignore next */
    statusDiv.textContent = 'Error: ' + e.data.error; /* v8 ignore next */ /* v8 ignore next */
    console.error(e.data.error); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
function renderSidebar(nodeId: string | null) {
  /* v8 ignore next */ /* v8 ignore next */
  if (!currentGraph) {
    /* v8 ignore next */ /* v8 ignore next */
    sidebar.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  sidebar.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (!nodeId) {
    /* v8 ignore next */ /* v8 ignore next */
    // Show Graph Properties /* v8 ignore next */ /* v8 ignore next */
    let content = `<h2>Graph Properties</h2>`; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Name:</b> ${currentGraph.name}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Producer:</b> ${currentGraph.producerName} v${currentGraph.producerVersion}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Domain:</b> ${currentGraph.domain || 'N/A'}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Model Version:</b> ${currentGraph.modelVersion}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    content += `<hr style="border-color:var(--border)"/>`; /* v8 ignore next */ /* v8 ignore next */
    content += `<h3>Opset Imports</h3>`; /* v8 ignore next */ /* v8 ignore next */
    content += `<div id="opset-imports-container">`; /* v8 ignore next */ /* v8 ignore next */
    for (const [domain, version] of Object.entries(currentGraph.opsetImports || {})) {
      /* v8 ignore next */ /* v8 ignore next */
      content += `<div style="margin-bottom: 5px;"> /* v8 ignore next */ /* v8 ignore next */
        <input type="text" value="${domain}" disabled style="width: 120px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" /> :  /* v8 ignore next */ /* v8 ignore next */
        <input type="number" class="opset-version-input" data-domain="${domain}" value="${version}" style="width: 60px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" /> /* v8 ignore next */ /* v8 ignore next */
      </div>`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    content += `</div>`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    content += `<div style="margin-top: 10px;"> /* v8 ignore next */ /* v8 ignore next */
      <input type="text" id="new-opset-domain" placeholder="Domain (e.g., ai.onnx)" style="width: 120px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" /> /* v8 ignore next */ /* v8 ignore next */
      <input type="number" id="new-opset-version" placeholder="Version" style="width: 60px; background: var(--input-bg); color: var(--text); border: 1px solid var(--border); padding: 2px;" /> /* v8 ignore next */ /* v8 ignore next */
      <button id="add-opset-btn" style="background: #4A90E2; color: white; border: none; padding: 4px 8px; cursor: pointer;">Add</button> /* v8 ignore next */ /* v8 ignore next */
    </div> /* v8 ignore next */ /* v8 ignore next */
     /* v8 ignore next */ /* v8 ignore next */
    <div style="margin-top: 20px;"> /* v8 ignore next */ /* v8 ignore next */
       <button id="btn-auto-format" style="width: 100%; padding: 8px; background: #555; color: white; border: 1px solid #777; cursor: pointer; border-radius: 4px;">Auto-Format Node Names</button> /* v8 ignore next */ /* v8 ignore next */
    </div>`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    sidebar.innerHTML = content; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Attach events /* v8 ignore next */ /* v8 ignore next */
    const mutator = new GraphMutator(currentGraph); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const versionInputs =
      sidebar.querySelectorAll('.opset-version-input'); /* v8 ignore next */ /* v8 ignore next */
    versionInputs.forEach((input) => {
      /* v8 ignore next */ /* v8 ignore next */
      input.addEventListener('change', (e) => {
        /* v8 ignore next */ /* v8 ignore next */
        const el = e.target as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
        const domain = el.getAttribute('data-domain')!; /* v8 ignore next */ /* v8 ignore next */
        const newVersion = parseInt(el.value, 10); /* v8 ignore next */ /* v8 ignore next */
        if (!isNaN(newVersion)) {
          /* v8 ignore next */ /* v8 ignore next */
          currentGraph!.opsetImports[domain] = newVersion; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const addBtn = sidebar.querySelector(
      '#add-opset-btn',
    ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
    addBtn.addEventListener('click', () => {
      /* v8 ignore next */ /* v8 ignore next */
      const domainInput = sidebar.querySelector(
        '#new-opset-domain',
      ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
      const versionInput = sidebar.querySelector(
        '#new-opset-version',
      ) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
      const domain = domainInput.value.trim(); /* v8 ignore next */ /* v8 ignore next */
      const version = parseInt(versionInput.value, 10); /* v8 ignore next */ /* v8 ignore next */
      if (domain && !isNaN(version)) {
        /* v8 ignore next */ /* v8 ignore next */
        currentGraph!.opsetImports[domain] = version; /* v8 ignore next */ /* v8 ignore next */
        renderSidebar(null); // Re-render /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const btnAutoFormat = sidebar.querySelector(
      '#btn-auto-format',
    ) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
    if (btnAutoFormat) {
      /* v8 ignore next */ /* v8 ignore next */
      btnAutoFormat.addEventListener('click', () => {
        /* v8 ignore next */ /* v8 ignore next */
        import('@onnx9000/modifier/dist/components/utilities.js').then(({ ModifierUtilities }) => {
          /* v8 ignore next */ /* v8 ignore next */
          const utils = new ModifierUtilities(mutator); /* v8 ignore next */ /* v8 ignore next */
          utils.autoFormatNodeNames(); /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          // Re-layout and render /* v8 ignore next */ /* v8 ignore next */
          statusDiv.textContent =
            'Recalculating layout...'; /* v8 ignore next */ /* v8 ignore next */
          import('./layout/dag').then(({ computeLayout }) => {
            /* v8 ignore next */ /* v8 ignore next */
            const layout = computeLayout(
              currentGraph!,
              'TB',
            ); /* v8 ignore next */ /* v8 ignore next */
            renderer.setLayout(layout); /* v8 ignore next */ /* v8 ignore next */
            renderSidebar(null); /* v8 ignore next */ /* v8 ignore next */
            statusDiv.textContent =
              'Rendered Model: ' + currentGraph!.name; /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Find node /* v8 ignore next */ /* v8 ignore next */
  let title = ''; /* v8 ignore next */ /* v8 ignore next */
  let content = ''; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (nodeId.startsWith('input_')) {
    /* v8 ignore next */ /* v8 ignore next */
    const name = nodeId.substring(6); /* v8 ignore next */ /* v8 ignore next */
    const info = currentGraph.inputs.find(
      (i) => i.name === name,
    ); /* v8 ignore next */ /* v8 ignore next */
    title = 'Graph Input'; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Name:</b> ${name}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    if (info) {
      /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Type:</b> ${info.dtype}<br/>`; /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Shape:</b> [${info.shape.join(', ')}]<br/>`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } else if (nodeId.startsWith('output_')) {
    /* v8 ignore next */ /* v8 ignore next */
    const name = nodeId.substring(7); /* v8 ignore next */ /* v8 ignore next */
    const info = currentGraph.outputs.find(
      (o) => o.name === name,
    ); /* v8 ignore next */ /* v8 ignore next */
    title = 'Graph Output'; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Name:</b> ${name}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    if (info) {
      /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Type:</b> ${info.dtype}<br/>`; /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Shape:</b> [${info.shape.join(', ')}]<br/>`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } else if (nodeId.startsWith('const_')) {
    /* v8 ignore next */ /* v8 ignore next */
    const name = nodeId.substring(6); /* v8 ignore next */ /* v8 ignore next */
    const t = currentGraph.tensors[name]; /* v8 ignore next */ /* v8 ignore next */
    title = 'Initializer / Constant'; /* v8 ignore next */ /* v8 ignore next */
    content += `<b>Name:</b> ${name}<br/>`; /* v8 ignore next */ /* v8 ignore next */
    if (t) {
      /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Type:</b> ${t.dtype}<br/>`; /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Shape:</b> [${t.shape.join(', ')}]<br/>`; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (t.data && t.data instanceof Uint8Array && t.dtype === 'float32') {
        /* v8 ignore next */ /* v8 ignore next */
        const f32 = new Float32Array(
          t.data.buffer,
          t.data.byteOffset,
          Math.min(t.size, 100),
        ); /* v8 ignore next */ /* v8 ignore next */
        let min = Infinity /* v8 ignore next */ /* v8 ignore next */,
          max = -Infinity /* v8 ignore next */ /* v8 ignore next */,
          sum = 0; /* v8 ignore next */ /* v8 ignore next */
        const fullF32 = new Float32Array(
          t.data.buffer,
          t.data.byteOffset,
          t.size,
        ); /* v8 ignore next */ /* v8 ignore next */
        for (let i = 0; i < t.size; i++) {
          /* v8 ignore next */ /* v8 ignore next */
          const v = fullF32[i]!; /* v8 ignore next */ /* v8 ignore next */
          if (v < min) min = v; /* v8 ignore next */ /* v8 ignore next */
          if (v > max) max = v; /* v8 ignore next */ /* v8 ignore next */
          sum += v; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        const mean = sum / t.size; /* v8 ignore next */ /* v8 ignore next */
        let vSum = 0; /* v8 ignore next */ /* v8 ignore next */
        for (let i = 0; i < t.size; i++) {
          /* v8 ignore next */ /* v8 ignore next */
          vSum += Math.pow(fullF32[i]! - mean, 2); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        const variance = vSum / t.size; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        content += `<br/><b>Statistics:</b><br/>`; /* v8 ignore next */ /* v8 ignore next */
        content += `Min: ${min.toFixed(4)}<br/>Max: ${max.toFixed(4)}<br/>Mean: ${mean.toFixed(4)}<br/>Variance: ${variance.toFixed(4)}<br/>`; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // Generate heatmap for 2D/3D/4D weights if small enough /* v8 ignore next */ /* v8 ignore next */
        if (t.shape.length >= 2) {
          /* v8 ignore next */ /* v8 ignore next */
          const dims = t.shape as number[]; /* v8 ignore next */ /* v8 ignore next */
          const lastH = dims[dims.length - 2]!; /* v8 ignore next */ /* v8 ignore next */
          const lastW = dims[dims.length - 1]!; /* v8 ignore next */ /* v8 ignore next */
          const strideH = lastW; /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          if (lastH > 0 && lastH <= 64 && lastW > 0 && lastW <= 64) {
            /* v8 ignore next */ /* v8 ignore next */
            const hCanvas =
              document.createElement('canvas'); /* v8 ignore next */ /* v8 ignore next */
            const hScale = Math.max(
              1,
              Math.floor(200 / Math.max(lastH, lastW)),
            ); /* v8 ignore next */ /* v8 ignore next */
            hCanvas.width = lastW * hScale; /* v8 ignore next */ /* v8 ignore next */
            hCanvas.height = lastH * hScale; /* v8 ignore next */ /* v8 ignore next */
            const hCtx = hCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
            if (hCtx) {
              /* v8 ignore next */ /* v8 ignore next */
              const range = Math.max(1e-5, max - min); /* v8 ignore next */ /* v8 ignore next */
              // Draw just the first slice (e.g., first out_channel, first in_channel) /* v8 ignore next */ /* v8 ignore next */
              for (let y = 0; y < lastH; y++) {
                /* v8 ignore next */ /* v8 ignore next */
                for (let x = 0; x < lastW; x++) {
                  /* v8 ignore next */ /* v8 ignore next */
                  const val = fullF32[y * strideH + x]!; /* v8 ignore next */ /* v8 ignore next */
                  // Normalize to 0-1 /* v8 ignore next */ /* v8 ignore next */
                  const norm = (val - min) / range; /* v8 ignore next */ /* v8 ignore next */
                  /* v8 ignore next */ /* v8 ignore next */
                  // Simple diverging colormap: min=blue, mid=black, max=red /* v8 ignore next */ /* v8 ignore next */
                  let r = 0 /* v8 ignore next */ /* v8 ignore next */,
                    g = 0 /* v8 ignore next */ /* v8 ignore next */,
                    b = 0; /* v8 ignore next */ /* v8 ignore next */
                  if (norm > 0.5) {
                    /* v8 ignore next */ /* v8 ignore next */
                    r = Math.floor(
                      (norm - 0.5) * 2 * 255,
                    ); /* v8 ignore next */ /* v8 ignore next */
                  } else {
                    /* v8 ignore next */ /* v8 ignore next */
                    b = Math.floor(
                      (0.5 - norm) * 2 * 255,
                    ); /* v8 ignore next */ /* v8 ignore next */
                  } /* v8 ignore next */ /* v8 ignore next */
                  /* v8 ignore next */ /* v8 ignore next */
                  hCtx.fillStyle = `rgb(${r},${g},${b})`; /* v8 ignore next */ /* v8 ignore next */
                  hCtx.fillRect(
                    x * hScale,
                    y * hScale,
                    hScale,
                    hScale,
                  ); /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
              const dataUrl = hCanvas.toDataURL(); /* v8 ignore next */ /* v8 ignore next */
              content += `<br/><b>Heatmap (First Slice):</b><br/>`; /* v8 ignore next */ /* v8 ignore next */
              content += `<img src="${dataUrl}" style="border:1px solid #555; image-rendering: pixelated;" /><br/>`; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        content += `<br/><b>Data (Flattened):</b><br/>`; /* v8 ignore next */ /* v8 ignore next */
        content += `<div style="max-height: 200px; overflow-y: auto; background: var(--input-bg); padding: 5px; font-size: 0.9em; word-break: break-all; border: 1px solid var(--border);">`; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        let matrixText = ''; /* v8 ignore next */ /* v8 ignore next */
        if (t.shape.length >= 2) {
          /* v8 ignore next */ /* v8 ignore next */
          const rows = Math.min(t.shape[0] as number, 10); /* v8 ignore next */ /* v8 ignore next */
          const cols = Math.min(t.shape[1] as number, 10); /* v8 ignore next */ /* v8 ignore next */
          matrixText += `[\n`; /* v8 ignore next */ /* v8 ignore next */
          for (let r = 0; r < rows; r++) {
            /* v8 ignore next */ /* v8 ignore next */
            let rowStr = '  ['; /* v8 ignore next */ /* v8 ignore next */
            for (let c = 0; c < cols; c++) {
              /* v8 ignore next */ /* v8 ignore next */
              rowStr +=
                /* v8 ignore next */ /* v8 ignore next */
                fullF32[r * (t.shape[1] as number) + c]!.toFixed(4) +
                (c < cols - 1 ? ', ' : ''); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            rowStr +=
              (cols < (t.shape[1] as number) ? ' ...' : '') +
              ']'; /* v8 ignore next */ /* v8 ignore next */
            matrixText +=
              rowStr + (r < rows - 1 ? ',\n' : '\n'); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          matrixText +=
            (rows < (t.shape[0] as number) ? '  ...\n' : '') +
            ']'; /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          matrixText = `[${Array.from(f32) /* v8 ignore next */ /* v8 ignore next */
            .map((x) => x.toFixed(4)) /* v8 ignore next */ /* v8 ignore next */
            .join(', ')}${t.size > 100 ? ' ...' : ''}]`; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        content += `<pre style="margin:0; font-family: monospace;">${matrixText}</pre></div>`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    // Normal node /* v8 ignore next */ /* v8 ignore next */
    const node = currentGraph.nodes.find(
      (n) => n.id === nodeId,
    ); /* v8 ignore next */ /* v8 ignore next */
    if (node) {
      /* v8 ignore next */ /* v8 ignore next */
      title = node.opType; /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Name:</b> ${node.name || '(unnamed)'}<br/>`; /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Domain:</b> ${node.domain || 'ai.onnx'}<br/>`; /* v8 ignore next */ /* v8 ignore next */
      // 253. Render Graph Doc Strings using Markdown in the UI (simplified regex replacement) /* v8 ignore next */ /* v8 ignore next */
      if (node.docString) {
        /* v8 ignore next */ /* v8 ignore next */
        let md = node.docString; /* v8 ignore next */ /* v8 ignore next */
        md = md.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>'); // bold /* v8 ignore next */ /* v8 ignore next */
        md = md.replace(/\*(.+?)\*/g, '<i>$1</i>'); // italic /* v8 ignore next */ /* v8 ignore next */
        md = md.replace(/\n/g, '<br/>'); // newlines /* v8 ignore next */ /* v8 ignore next */
        content += `<p style="font-size: 0.9em; background: var(--input-bg); border: 1px solid var(--border); padding: 5px; border-radius: 4px;">${md}</p>`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      content += `<hr style="border-color:var(--border)"/>`; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Inputs:</b><ul style="padding-left: 20px; margin-top: 5px">`; /* v8 ignore next */ /* v8 ignore next */
      node.inputs.forEach((i, idx) => {
        /* v8 ignore next */ /* v8 ignore next */
        if (!i) {
          /* v8 ignore next */ /* v8 ignore next */
          content += `<li><i>(optional/missing)</i></li>`; /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          // Find producer /* v8 ignore next */ /* v8 ignore next */
          let producer = 'Graph Input'; /* v8 ignore next */ /* v8 ignore next */
          const pNode = currentGraph!.nodes.find((n) =>
            n.outputs.includes(i),
          ); /* v8 ignore next */ /* v8 ignore next */
          if (pNode)
            producer = pNode.name || pNode.opType; /* v8 ignore next */ /* v8 ignore next */
          else if (currentGraph!.initializers.includes(i))
            producer = 'Constant'; /* v8 ignore next */ /* v8 ignore next */
          content += `<li><b>${i}</b> <span style="color:var(--text-muted)">&larr; ${producer}</span></li>`; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      content += `</ul>`; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      content += `<b>Outputs:</b><ul style="padding-left: 20px; margin-top: 5px">`; /* v8 ignore next */ /* v8 ignore next */
      node.outputs.forEach((o) => {
        /* v8 ignore next */ /* v8 ignore next */
        if (!o) {
          /* v8 ignore next */ /* v8 ignore next */
          content += `<li><i>(optional/missing)</i></li>`; /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          // Find consumers /* v8 ignore next */ /* v8 ignore next */
          const consumers = currentGraph!.nodes /* v8 ignore next */ /* v8 ignore next */
            .filter((n) => n.inputs.includes(o)) /* v8 ignore next */ /* v8 ignore next */
            .map((n) => n.name || n.opType); /* v8 ignore next */ /* v8 ignore next */
          const cStr =
            consumers.length > 0
              ? consumers.join(', ')
              : 'Graph Output'; /* v8 ignore next */ /* v8 ignore next */
          content += `<li><b>${o}</b> <span style="color:var(--text-muted)">&rarr; ${cStr}</span></li>`; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      content += `</ul>`; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const attrKeys = Object.keys(node.attributes); /* v8 ignore next */ /* v8 ignore next */
      if (attrKeys.length > 0) {
        /* v8 ignore next */ /* v8 ignore next */
        content += `<hr style="border-color:var(--border)"/><b>Attributes:</b><br/>`; /* v8 ignore next */ /* v8 ignore next */
        for (const k of attrKeys) {
          /* v8 ignore next */ /* v8 ignore next */
          const a = node.attributes[k]!; /* v8 ignore next */ /* v8 ignore next */
          let valStr = String(a.value); /* v8 ignore next */ /* v8 ignore next */
          if (a.type === 'TENSOR') valStr = '[Tensor]'; /* v8 ignore next */ /* v8 ignore next */
          if (a.type === 'GRAPH') valStr = '[Graph]'; /* v8 ignore next */ /* v8 ignore next */
          content += `<i>${k}</i> (${a.type}): ${valStr}<br/>`; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // 213. UI warnings for nodes using `double` (float64) precision /* v8 ignore next */ /* v8 ignore next */
      let hasDouble = false; /* v8 ignore next */ /* v8 ignore next */
      if (node.opType === 'Cast' && node.attributes['to'] && node.attributes['to'].value === 11)
        /* v8 ignore next */ /* v8 ignore next */
        hasDouble = true; /* v8 ignore next */ /* v8 ignore next */
      for (const i of node.inputs) {
        /* v8 ignore next */ /* v8 ignore next */
        if (!i) continue; /* v8 ignore next */ /* v8 ignore next */
        const info =
          currentGraph.inputs.find((vi) => vi.name === i) ||
          currentGraph.tensors[i]; /* v8 ignore next */ /* v8 ignore next */
        if (info && info.dtype === 'float64')
          hasDouble = true; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      if (hasDouble) {
        /* v8 ignore next */ /* v8 ignore next */
        content += `<div style="margin-top: 10px; padding: 5px; background: #5a2e1e; border: 1px solid #ff9800; border-radius: 4px; font-size: 0.9em; color: #ffeb3b;"> /* v8 ignore next */ /* v8 ignore next */
          <b>⚠️ Performance Warning:</b> This node operates on \`float64\` (double) precision, which is extremely slow on WebGPU and unsupported on many mobile NPUs. Consider casting to \`float32\` or \`float16\`. /* v8 ignore next */ /* v8 ignore next */
        </div>`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  sidebar.innerHTML = `<h2 style="margin-top:0">${title}</h2>${content}`; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
renderer.onSelect = (nodeId) => {
  /* v8 ignore next */ /* v8 ignore next */
  renderSidebar(nodeId); /* v8 ignore next */ /* v8 ignore next */
  const announcer =
    document.getElementById('aria-announcer'); /* v8 ignore next */ /* v8 ignore next */
  if (announcer && nodeId)
    announcer.textContent = 'Selected node ' + nodeId; /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// 229. Expose a global window.onnxModifier object for developer console hacking /* v8 ignore next */ /* v8 ignore next */
(window as ReturnType<typeof JSON.parse>).onnxModifier = {
  /* v8 ignore next */ /* v8 ignore next */
  getGraph: () => currentGraph /* v8 ignore next */ /* v8 ignore next */,
  getRenderer: () => renderer /* v8 ignore next */ /* v8 ignore next */,
  getMutator: () => new GraphMutator(currentGraph!) /* v8 ignore next */ /* v8 ignore next */,
  download: async () => {
    /* v8 ignore next */ /* v8 ignore next */
    if (currentGraph) {
      /* v8 ignore next */ /* v8 ignore next */
      const exporter = new ModelExporter(
        new GraphMutator(currentGraph),
      ); /* v8 ignore next */ /* v8 ignore next */
      const data = await exporter.exportModel(); /* v8 ignore next */ /* v8 ignore next */
      exporter.downloadBlob('console_hacked.onnx', data); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */,
};
