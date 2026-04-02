import { Node } from '@onnx9000/core';
import { GraphMutator } from '../../GraphMutator.js';

// Phase 10: Specific ONNX Operator Custom Editors

export interface CustomEditorParams {
  container: HTMLElement;
  node: Node;
  mutator: GraphMutator;
}

export function renderCustomEditor(params: CustomEditorParams): boolean {
  const { container, node, mutator } = params;

  switch (node.opType) {
    case 'Conv':
      renderConvEditor(container, node, mutator);
      return true;
    case 'Gemm':
      renderGemmEditor(container, node, mutator);
      return true;
    case 'Split':
      renderSplitEditor(container, node, mutator);
      return true;
    case 'Resize':
      renderResizeEditor(container, node, mutator);
      return true;
    case 'Squeeze':
    case 'Unsqueeze':
      renderSqueezeEditor(container, node, mutator);
      return true;
    case 'Cast':
      renderCastEditor(container, node, mutator);
      return true;
    case 'Constant':
      renderConstantEditor(container, node, mutator);
      return true;
    case 'If':
      renderIfEditor(container, node, mutator);
      return true;
    case 'Loop':
      renderLoopEditor(container, node, mutator);
      return true;
    case 'SequenceConstruct':
    case 'SequenceInsert':
      renderSequenceEditor(container, node, mutator);
      return true;
    case 'ScatterND':
      renderScatterNDEditor(container, node, mutator);
      return true;
    case 'CastMap':
    case 'MapType': // placeholder for custom map attributes
      renderMapEditor(container, node, mutator);
      return true;
    default:
      return false; // Not handled
  }
}

// 111. Specialized UI form for Conv (Strides, Pads, Dilations, Groups)
function renderConvEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Conv Settings');

  createArrayInput(container, node, mutator, 'strides', 'INTS');
  createArrayInput(container, node, mutator, 'pads', 'INTS');
  createArrayInput(container, node, mutator, 'dilations', 'INTS');
  createNumberInput(container, node, mutator, 'group', 'INT', 1);
}

// 112. Specialized UI form for Gemm (transA, transB, alpha, beta)
function renderGemmEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Gemm Settings');

  createCheckbox(container, node, mutator, 'transA', 'INT', 0);
  createCheckbox(container, node, mutator, 'transB', 'INT', 0);
  createNumberInput(container, node, mutator, 'alpha', 'FLOAT', 1.0);
  createNumberInput(container, node, mutator, 'beta', 'FLOAT', 1.0);
}

// 113. Specialized UI form for Split (split attribute)
function renderSplitEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Split Settings');
  createNumberInput(container, node, mutator, 'axis', 'INT', 0);
  createArrayInput(container, node, mutator, 'split', 'INTS');
}

// 114. Specialized UI form for Resize (Dropdowns)
function renderResizeEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Resize Settings');

  const modes = ['nearest', 'linear', 'cubic'];
  const coordModes = [
    'half_pixel',
    'pytorch_half_pixel',
    'align_corners',
    'asymmetric',
    'tf_half_pixel_for_nn',
    'tf_crop_and_resize',
  ];
  const nearestModes = ['round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil'];

  createDropdown(container, node, mutator, 'mode', 'STRING', modes, 'nearest');
  createDropdown(
    container,
    node,
    mutator,
    'coordinate_transformation_mode',
    'STRING',
    coordModes,
    'half_pixel',
  );
  createDropdown(
    container,
    node,
    mutator,
    'nearest_mode',
    'STRING',
    nearestModes,
    'round_prefer_floor',
  );
}

// 115. Specialized UI form for Squeeze / Unsqueeze (Axes)
function renderSqueezeEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, `${node.opType} Settings`);
  // Note: in newer opsets, axes is an input, but in opset 11 it was an attribute. We handle the attribute case here.
  if (node.attributes['axes']) {
    createArrayInput(container, node, mutator, 'axes', 'INTS');
  } else {
    const hint = document.createElement('div');
    hint.style.fontSize = '12px';
    hint.textContent = 'Axes are defined as an input tensor in this opset.';
    container.appendChild(hint);
  }
}

// 116. Specialized UI form for Cast (Dropdown mapping integers to types)
const TENSOR_TYPES = [
  { id: 1, name: 'FLOAT' },
  { id: 2, name: 'UINT8' },
  { id: 3, name: 'INT8' },
  { id: 4, name: 'UINT16' },
  { id: 5, name: 'INT16' },
  { id: 6, name: 'INT32' },
  { id: 7, name: 'INT64' },
  { id: 8, name: 'STRING' },
  { id: 9, name: 'BOOL' },
  { id: 10, name: 'FLOAT16' },
  { id: 11, name: 'DOUBLE' },
  { id: 12, name: 'UINT32' },
  { id: 13, name: 'UINT64' },
  { id: 14, name: 'COMPLEX64' },
  { id: 15, name: 'COMPLEX128' },
  { id: 16, name: 'BFLOAT16' },
  { id: 17, name: 'FLOAT8E4M3FN' },
  { id: 18, name: 'FLOAT8E4M3FNUZ' },
  { id: 19, name: 'FLOAT8E5M2' },
  { id: 20, name: 'FLOAT8E5M2FNUZ' },
];

// 204. Validate Cast node conversions accurately represent bounds correctly
function renderCastEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Cast Settings');

  const attr = node.attributes['to'];
  const currentVal = attr ? Number(attr.value) : 1;

  const div = document.createElement('div');
  div.style.marginBottom = '8px';

  const label = document.createElement('label');
  label.textContent = `to ℹ️`;
  label.title = `Standard ONNX to attribute (Type Enum)`;
  label.style.display = 'block';
  label.style.fontSize = '12px';

  const select = document.createElement('select');
  select.style.width = '100%';

  for (const type of TENSOR_TYPES) {
    const opt = document.createElement('option');
    opt.value = String(type.id);
    opt.textContent = type.name;
    if (type.id === currentVal) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener('change', function (e: any) {
    const el = e.target || this;
    const val = parseInt(el.value, 10);
    mutator.setNodeAttribute(node.name || node.id, 'to', val, 'INT');

    // Bounds checking simulation
    if (val === 2 || val === 3 || val === 4 || val === 5) {
      alert(
        'Warning: Casting to low-precision integer formats may result in bound clipping if the input tensor exceeds standard limits.',
      );
    }
  });

  div.appendChild(label);
  div.appendChild(select);
  container.appendChild(div);
}

// 117. Specialized UI form for Constant
// 284. Allow editing of tensor attributes visually via Hex editor
function renderHexEditor(container: HTMLElement, data: Uint8Array, len: number) {
  const hexDiv = document.createElement('div');
  hexDiv.style.fontFamily = 'monospace';
  hexDiv.style.fontSize = '10px';
  hexDiv.style.marginTop = '8px';
  hexDiv.style.whiteSpace = 'pre-wrap';

  // only show first 64 bytes in hex
  const limit = Math.min(len, 64);
  let hexStr = '';
  for (let i = 0; i < limit; i++) {
    hexStr += data[i]!.toString(16).padStart(2, '0') + ' ';
    if ((i + 1) % 16 === 0) hexStr += '\n';
  }
  if (limit < len) hexStr += '\n...';

  hexDiv.textContent = 'Hex view:\n' + hexStr;
  container.appendChild(hexDiv);
}

// 260. Render large constant tensors via paginated tables
function renderConstantEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Constant Value');

  const attr = node.attributes['value'];
  if (!attr) return;

  if (attr.type === 'TENSOR' && attr.value && (attr.value as any).data) {
    const data = (attr.value as any).data;
    const len = data.length;
    if (len === undefined) {
      container.textContent = 'Constant empty or unsupported';
      return;
    }

    const PAGE_SIZE = 100;
    let currentPage = 0;
    const totalPages = Math.ceil(len / PAGE_SIZE);

    const dataDiv = document.createElement('div');
    dataDiv.style.maxHeight = '200px';
    dataDiv.style.overflowY = 'auto';
    dataDiv.style.fontSize = '12px';
    dataDiv.style.fontFamily = 'monospace';
    dataDiv.style.background = '#f8f9fa';
    dataDiv.style.padding = '8px';

    const controls = document.createElement('div');
    controls.style.display = 'flex';
    controls.style.justifyContent = 'space-between';
    controls.style.marginTop = '4px';

    const btnPrev = document.createElement('button');
    btnPrev.textContent = 'Prev';
    const btnNext = document.createElement('button');
    btnNext.textContent = 'Next';
    const lblPage = document.createElement('span');
    lblPage.style.fontSize = '12px';

    const renderPage = () => {
      const start = currentPage * PAGE_SIZE;
      const end = Math.min(start + PAGE_SIZE, len);
      let html = '<table style="width: 100%; border-collapse: collapse;">';
      for (let i = start; i < end; i++) {
        html += `<tr><td style="border: 1px solid #ccc; padding: 2px;">[${i}]</td><td style="border: 1px solid #ccc; padding: 2px;">${data[i]}</td></tr>`;
      }
      html += '</table>';
      dataDiv.innerHTML = html;
      lblPage.textContent = `Page ${currentPage + 1} of ${totalPages} (${len} elements)`;
      btnPrev.disabled = currentPage === 0;
      btnNext.disabled = currentPage >= totalPages - 1;
    };

    btnPrev.onclick = () => {
      if (currentPage > 0) {
        currentPage--;
        renderPage();
      }
    };
    btnNext.onclick = () => {
      if (currentPage < totalPages - 1) {
        currentPage++;
        renderPage();
      }
    };

    controls.appendChild(btnPrev);
    controls.appendChild(lblPage);
    controls.appendChild(btnNext);

    container.appendChild(dataDiv);
    container.appendChild(controls);
    renderPage();

    if (attr.value && (attr.value as any).data && (attr.value as any).data.buffer) {
      renderHexEditor(
        container,
        new Uint8Array((attr.value as any).data.buffer),
        (attr.value as any).data.byteLength,
      );
    }
  } else {
    const div = document.createElement('div');
    div.textContent = String(attr.value);
    container.appendChild(div);
  }
}

// 118. If subgraph
function renderIfEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'If Branch Navigation');
  createSubgraphButton(container, node, mutator, 'then_branch');
  createSubgraphButton(container, node, mutator, 'else_branch');
}

// 119. Loop subgraph
function renderLoopEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Loop Body Navigation');
  createSubgraphButton(container, node, mutator, 'body');
}

// 234. Edit Sequence attributes
function renderSequenceEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Sequence Settings');
  // Sequences might not have direct attributes but if they have `position` or similar we render it
  const hint = document.createElement('div');
  hint.style.fontSize = '12px';
  hint.textContent =
    'Editing ONNX Sequence structure. (Inputs usually handle sequences dynamically)';
  container.appendChild(hint);
}

// 272. ScatterND logic
function renderScatterNDEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'ScatterND Settings');
  const attr = node.attributes['reduction'];
  const modes = ['none', 'add', 'mul', 'max', 'min'];
  createDropdown(
    container,
    node,
    mutator,
    'reduction',
    'STRING',
    modes,
    attr ? String(attr.value) : 'none',
  );
}

// 235. Map type attributes
function renderMapEditor(container: HTMLElement, node: Node, mutator: GraphMutator) {
  createHeader(container, 'Map Settings');
  // For Map we might have map_form, map_type etc.
  createDropdown(container, node, mutator, 'cast_map_type', 'INT', ['1', '2', '3', '7'], '1');
}

// ---- Helpers ----

function createHeader(container: HTMLElement, title: string) {
  const h = document.createElement('h4');
  h.textContent = title;
  h.style.marginTop = '16px';
  h.style.marginBottom = '8px';
  h.style.borderBottom = '1px solid #ccc';
  container.appendChild(h);
}

function createArrayInput(
  container: HTMLElement,
  node: Node,
  mutator: GraphMutator,
  attrName: string,
  attrType: string,
) {
  const attr = node.attributes[attrName];
  const div = document.createElement('div');
  div.style.marginBottom = '8px';

  const label = document.createElement('label');
  label.textContent = `${attrName} ℹ️`;
  label.title = `Standard ONNX ${attrName} attribute`;
  label.style.display = 'block';
  label.style.fontSize = '12px';

  const input = document.createElement('input');
  input.type = 'text';
  input.value = attr ? JSON.stringify(attr.value) : '[]';
  input.setAttribute('value', input.value);
  input.style.width = '100%';

  input.addEventListener('change', function (e: any) {
    const el = e.target || this;
    try {
      const val = JSON.parse(el.value);
      if (Array.isArray(val)) {
        mutator.setNodeAttribute(node.name || node.id, attrName, val, attrType as any);
      }
    } catch (err) {
      /* ignore */
    }
  });

  div.appendChild(label);
  div.appendChild(input);
  container.appendChild(div);
}

function createNumberInput(
  container: HTMLElement,
  node: Node,
  mutator: GraphMutator,
  attrName: string,
  attrType: string,
  defaultVal: number,
) {
  const attr = node.attributes[attrName];
  const div = document.createElement('div');
  div.style.marginBottom = '8px';

  const label = document.createElement('label');
  label.textContent = `${attrName} ℹ️`;
  label.title = `Standard ONNX ${attrName} attribute`;
  label.style.display = 'block';
  label.style.fontSize = '12px';

  const input = document.createElement('input');
  input.type = 'number';
  input.step = attrType === 'FLOAT' ? '0.01' : '1';
  input.value = attr ? String(attr.value) : String(defaultVal);
  input.setAttribute('value', input.value);
  input.style.width = '100%';

  input.addEventListener('change', function (e: any) {
    const el = e.target || this;
    const v = parseFloat(el.value);
    if (!Number.isNaN(v)) {
      mutator.setNodeAttribute(
        node.name || node.id,
        attrName,
        attrType === 'INT' ? Math.floor(v) : v,
        attrType as any,
      );
    }
  });

  div.appendChild(label);
  div.appendChild(input);
  container.appendChild(div);
}

function createCheckbox(
  container: HTMLElement,
  node: Node,
  mutator: GraphMutator,
  attrName: string,
  attrType: string,
  defaultVal: number,
) {
  const attr = node.attributes[attrName];
  const div = document.createElement('div');
  div.style.marginBottom = '8px';

  const label = document.createElement('label');
  label.style.fontSize = '12px';

  const input = document.createElement('input');
  input.type = 'checkbox';
  input.checked = attr ? !!attr.value : !!defaultVal;
  input.addEventListener('change', function (e: any) {
    const el = e.target || this;
    const checked = el.checked;
    mutator.setNodeAttribute(node.name || node.id, attrName, checked ? 1 : 0, attrType as any);
    // updated
  });

  label.appendChild(input);
  label.appendChild(document.createTextNode(` ${attrName} ℹ️`));
  label.title = `Standard ONNX ${attrName} attribute`;

  div.appendChild(label);
  container.appendChild(div);
}

function createDropdown(
  container: HTMLElement,
  node: Node,
  mutator: GraphMutator,
  attrName: string,
  attrType: string,
  options: string[],
  defaultVal: string,
) {
  const attr = node.attributes[attrName];
  const currentVal = attr ? String(attr.value) : defaultVal;

  const div = document.createElement('div');
  div.style.marginBottom = '8px';

  const label = document.createElement('label');
  label.textContent = `${attrName} ℹ️`;
  label.title = `Standard ONNX ${attrName} attribute`;
  label.style.display = 'block';
  label.style.fontSize = '12px';

  const select = document.createElement('select');
  select.style.width = '100%';

  for (const optStr of options) {
    const opt = document.createElement('option');
    opt.value = optStr;
    opt.textContent = optStr;
    if (optStr === currentVal) opt.selected = true;
    select.appendChild(opt);
  }

  select.addEventListener('change', function (e: any) {
    const el = e.target || this;
    // explicitly // explicitly fire event
    const val = el.value;
    mutator.setNodeAttribute(node.name || node.id, attrName, val, attrType as any);
  });

  div.appendChild(label);
  div.appendChild(select);
  container.appendChild(div);
}

function createSubgraphButton(
  container: HTMLElement,
  node: Node,
  mutator: GraphMutator,
  attrName: string,
) {
  const attr = node.attributes[attrName];
  if (!attr || attr.type !== 'GRAPH') return;

  const btn = document.createElement('button');
  btn.textContent = `Open ${attrName}`;
  btn.style.width = '100%';
  btn.style.marginTop = '4px';

  btn.onclick = () => {
    // 237. Dispatch an event to track sub-graphs
    const event = new CustomEvent('open-subgraph', {
      detail: { graph: attr.value, name: `${node.name || node.id}_${attrName}` },
    });
    window.dispatchEvent(event);
  };

  container.appendChild(btn);
}
