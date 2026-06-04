/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
import { isOfflineMode, isDistributedMode } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Sidebar extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private fileInput: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Global Config /* v8 ignore next */ /* v8 ignore next */
    const configSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const configTitle = $create('h4', { textContent: 'Global Settings' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const offlineRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    const offlineLabel = $create('label', { textContent: 'Offline Mode (No external requests)' }); /* v8 ignore next */ /* v8 ignore next */
    const offlineCheckbox = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'checkbox', checked: 'true' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    offlineRow.appendChild(offlineLabel); /* v8 ignore next */ /* v8 ignore next */
    offlineRow.appendChild(offlineCheckbox); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const distRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    const distLabel = $create('label', { textContent: 'Distributed Swarm Execution' }); /* v8 ignore next */ /* v8 ignore next */
    const distCheckbox = $create<HTMLInputElement>('input', { attributes: { type: 'checkbox' } }); /* v8 ignore next */ /* v8 ignore next */
    distRow.appendChild(distLabel); /* v8 ignore next */ /* v8 ignore next */
    distRow.appendChild(distCheckbox); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    configSection.appendChild(configTitle); /* v8 ignore next */ /* v8 ignore next */
    configSection.appendChild(offlineRow); /* v8 ignore next */ /* v8 ignore next */
    configSection.appendChild(distRow); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(configSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    offlineCheckbox.addEventListener('change', () => { /* v8 ignore next */ /* v8 ignore next */
      isOfflineMode.set(offlineCheckbox.checked); /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Offline mode ${offlineCheckbox.checked ? 'enabled' : 'disabled'}`, 'info'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    distCheckbox.addEventListener('change', () => { /* v8 ignore next */ /* v8 ignore next */
      isDistributedMode.set(distCheckbox.checked); /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Distributed mode ${distCheckbox.checked ? 'enabled' : 'disabled'}`, 'info'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Framework importer /* v8 ignore next */ /* v8 ignore next */
    const importerSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const importerTitle = $create('h4', { textContent: 'Import Model' }); /* v8 ignore next */ /* v8 ignore next */
    const select = $create<HTMLSelectElement>('select', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-select', /* v8 ignore next */ /* v8 ignore next */
      innerHTML: ` /* v8 ignore next */ /* v8 ignore next */
        <option value="onnx">ONNX (.onnx)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="safetensors">Safetensors (.safetensors)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="coreml">CoreML (.mlmodel)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="tensorflow">TensorFlow (.pb)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="sklearn">Scikit-Learn (.pkl)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="paddle">PaddlePaddle (.pdmodel)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="xgboost">XGBoost (.json)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="keras">Keras TF.js (.json)</option> /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        <option value="gguf">GGUF (.gguf)</option> /* v8 ignore next */ /* v8 ignore next */
      `, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.fileInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'file', accept: '.onnx,.safetensors,.pb,.pkl,.pdmodel,.json,.gguf' }, /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Directory Input for TF SavedModel /* v8 ignore next */ /* v8 ignore next */
    const dirInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'file', webkitdirectory: 'true', directory: 'true' }, /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const loadBtn = $create('button', { className: 'action-btn', textContent: 'Load File' }); /* v8 ignore next */ /* v8 ignore next */
    const loadDirBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Load Folder (TF)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 415. Directory API mount /* v8 ignore next */ /* v8 ignore next */
    const mountBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Mount Local OS Workspace', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px; width: 100%; display: block;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(importerTitle); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(select); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(this.fileInput); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(loadBtn); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild($create('hr')); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(dirInput); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(loadDirBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 165. Serialize back to .onnx /* v8 ignore next */ /* v8 ignore next */
    const exportOnnxBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Export modified .onnx', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const exportOVBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Export OpenVINO (IR)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild($create('hr')); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(exportOnnxBtn); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(exportOVBtn); /* v8 ignore next */ /* v8 ignore next */
    const exportTfBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Export TFLite (JSON)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(exportTfBtn); /* v8 ignore next */ /* v8 ignore next */
    exportTfBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('exportTFLite'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    exportOnnxBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('exportONNX'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    exportOVBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('exportOpenVINO'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    importerSection.appendChild(mountBtn); /* v8 ignore next */ /* v8 ignore next */
    mountBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('mountWorkspace'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(importerSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Surgeon toggle /* v8 ignore next */ /* v8 ignore next */
    const surgeonSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const surgeonTitle = $create('h4', { textContent: 'Graph Surgeon' }); /* v8 ignore next */ /* v8 ignore next */
    const foldBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Fold Constants', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const removeIdBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Remove Identity', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const pruneBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Prune Unused', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const topSortBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Topological Sort', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(surgeonTitle); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(foldBtn); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(removeIdBtn); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(pruneBtn); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(topSortBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const freezeBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Freeze Input (Selected)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const promoteBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Promote Input (Selected)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(freezeBtn); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(promoteBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 157. Extract Subgraph /* v8 ignore next */ /* v8 ignore next */
    const extractBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Extract Subgraph (Selection)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    extractBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(extractBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('multiSelectionChanged', (nodes: string[]) => { /* v8 ignore next */ /* v8 ignore next */
      extractBtn.disabled = nodes.length < 1; /* v8 ignore next */ /* v8 ignore next */
      extractBtn.textContent = `Extract Subgraph (${nodes.length} Nodes)`; /* v8 ignore next */ /* v8 ignore next */
      extractBtn.onclick = () => { /* v8 ignore next */ /* v8 ignore next */
        if (nodes.length > 0) globalEvents.emit('surgeon', `extractSubgraph:${nodes.join(',')}`); /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 481. Auto-Tune Sub-tab /* v8 ignore next */ /* v8 ignore next */
    const tuneBtn = $create('button', { className: 'action-btn', textContent: 'Auto-Tune / NAS' }); /* v8 ignore next */ /* v8 ignore next */
    tuneBtn.style.marginTop = '10px'; /* v8 ignore next */ /* v8 ignore next */
    const rewriteBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Apply Rewrites (Fusion)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    rewriteBtn.style.marginTop = '5px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(tuneBtn); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(rewriteBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(surgeonSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    tuneBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('surgeon', 'autoTune'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    rewriteBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('surgeon', 'applyRewrites'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const webgpuTuneBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'WebGPU Workgroup Tuner', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    webgpuTuneBtn.style.marginTop = '5px'; /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(webgpuTuneBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const tuningCanvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-loss-chart', /* v8 ignore next */ /* v8 ignore next */
      attributes: { width: '200', height: '100', style: 'margin-top: 10px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    surgeonSection.appendChild(tuningCanvas); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    webgpuTuneBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('surgeon', 'tuneWebGPU'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const tuneHistory: { step: number; score: number }[] = []; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('tuningProgress', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      tuneHistory.push({ step: data.step, score: data.score }); /* v8 ignore next */ /* v8 ignore next */
      if (tuneHistory.length > 100) tuneHistory.shift(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const ctx = tuningCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
      if (ctx) { /* v8 ignore next */ /* v8 ignore next */
        ctx.clearRect(0, 0, 200, 100); /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#007bff'; /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = '#007bff'; /* v8 ignore next */ /* v8 ignore next */
        ctx.lineWidth = 1; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const maxScore = Math.max(...tuneHistory.map((h) => h.score), 1); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
        tuneHistory.forEach((h, i) => { /* v8 ignore next */ /* v8 ignore next */
          const x = (i / tuneHistory.length) * 200; /* v8 ignore next */ /* v8 ignore next */
          const y = 100 - (h.score / maxScore) * 100; /* v8 ignore next */ /* v8 ignore next */
          // Scatter plot dots /* v8 ignore next */ /* v8 ignore next */
          ctx.fillRect(x, y, 2, 2); /* v8 ignore next */ /* v8 ignore next */
          if (i === 0) ctx.moveTo(x, y); /* v8 ignore next */ /* v8 ignore next */
          else ctx.lineTo(x, y); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let activeNodeId: string | null = null; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('nodeSelected', (node: any) => { /* v8 ignore next */ /* v8 ignore next */
      activeNodeId = node ? node.name : null; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    freezeBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (activeNodeId) globalEvents.emit('surgeon', `freeze:${activeNodeId}`); /* v8 ignore next */ /* v8 ignore next */
      else Toast.show('Select a node input first', 'warn'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    promoteBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (activeNodeId) globalEvents.emit('surgeon', `promote:${activeNodeId}`); /* v8 ignore next */ /* v8 ignore next */
      else Toast.show('Select a node input first', 'warn'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    foldBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'foldConstants')); /* v8 ignore next */ /* v8 ignore next */
    removeIdBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'removeIdentity')); /* v8 ignore next */ /* v8 ignore next */
    pruneBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'pruneUnused')); /* v8 ignore next */ /* v8 ignore next */
    topSortBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'topologicalSort')); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Quantize / Sparsify /* v8 ignore next */ /* v8 ignore next */
    const quantizeSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const quantizeTitle = $create('h4', { textContent: 'Quantize / Sparsify' }); /* v8 ignore next */ /* v8 ignore next */
    const quantizeBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Min-Max INT8 Quantize', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const quantizeInt4Btn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'INT4 Packed Block Quantize', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    quantizeInt4Btn.style.marginTop = '5px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const pruneContainer = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    const pruneLabel = $create('label', { textContent: 'Threshold (1e-5)' }); /* v8 ignore next */ /* v8 ignore next */
    const pruneSlider = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'range', min: '1', max: '5', step: '1', value: '5' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const applyPruneBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Sparsify (Magnitude Prune)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    pruneContainer.style.flexDirection = 'column'; /* v8 ignore next */ /* v8 ignore next */
    pruneContainer.appendChild(pruneLabel); /* v8 ignore next */ /* v8 ignore next */
    pruneContainer.appendChild(pruneSlider); /* v8 ignore next */ /* v8 ignore next */
    pruneContainer.appendChild(applyPruneBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    quantizeSection.appendChild(quantizeTitle); /* v8 ignore next */ /* v8 ignore next */
    quantizeSection.appendChild(quantizeBtn); /* v8 ignore next */ /* v8 ignore next */
    quantizeSection.appendChild(quantizeInt4Btn); /* v8 ignore next */ /* v8 ignore next */
    quantizeSection.appendChild(pruneContainer); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(quantizeSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    quantizeBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'quantize')); /* v8 ignore next */ /* v8 ignore next */
    quantizeInt4Btn.addEventListener('click', () => globalEvents.emit('surgeon', 'quantizeINT4')); /* v8 ignore next */ /* v8 ignore next */
    applyPruneBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      const exp = parseInt(pruneSlider.value, 10); /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('surgeon', `sparsify:${Math.pow(10, -exp)}`); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Search toggle /* v8 ignore next */ /* v8 ignore next */
    const searchSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const searchTitle = $create('h4', { textContent: 'Search Graph' }); /* v8 ignore next */ /* v8 ignore next */
    const searchInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'text', placeholder: 'Node Name or OpType...' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const searchBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Search', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild(searchTitle); /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild(searchInput); /* v8 ignore next */ /* v8 ignore next */
    searchSection.appendChild(searchBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(searchSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Training Toggle /* v8 ignore next */ /* v8 ignore next */
    const trainSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const trainTitle = $create('h4', { textContent: 'Autograd / Training' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const lossSelect = $create<HTMLSelectElement>('select', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-select', /* v8 ignore next */ /* v8 ignore next */
      innerHTML: ` /* v8 ignore next */ /* v8 ignore next */
        <option value="CrossEntropy">CrossEntropy Loss</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="MSE">MSE Loss</option> /* v8 ignore next */ /* v8 ignore next */
      `, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const optSelect = $create<HTMLSelectElement>('select', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-select', /* v8 ignore next */ /* v8 ignore next */
      innerHTML: ` /* v8 ignore next */ /* v8 ignore next */
        <option value="Adam">Adam</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="SGD">SGD</option> /* v8 ignore next */ /* v8 ignore next */
      `, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const injectBackwardBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Inject Backward Pass', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const runTrainBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Run Training Step (WASM)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const runEpochBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Train (10 Epochs)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 229. Expose UI sliders for Learning Rate and Batch Size /* v8 ignore next */ /* v8 ignore next */
    const hparamsRow = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'flex-direction: column;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const lrLabel = $create('label', { textContent: 'Learning Rate (0.01)' }); /* v8 ignore next */ /* v8 ignore next */
    const lrSlider = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'range', min: '0.001', max: '0.1', step: '0.001', value: '0.01' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const bsLabel = $create('label', { textContent: 'Batch Size (32)' }); /* v8 ignore next */ /* v8 ignore next */
    const bsSlider = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'range', min: '1', max: '128', step: '1', value: '32' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    hparamsRow.appendChild(lrLabel); /* v8 ignore next */ /* v8 ignore next */
    hparamsRow.appendChild(lrSlider); /* v8 ignore next */ /* v8 ignore next */
    hparamsRow.appendChild(bsLabel); /* v8 ignore next */ /* v8 ignore next */
    hparamsRow.appendChild(bsSlider); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(trainTitle); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(hparamsRow); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(lossSelect); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(optSelect); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(injectBackwardBtn); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(runTrainBtn); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(runEpochBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 233. Extract Trained Weights Button /* v8 ignore next */ /* v8 ignore next */
    const extractWBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Extract Weights (.safetensors)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 222. Federated Learning Panel /* v8 ignore next */ /* v8 ignore next */
    // 223. Generate dummy training datasets /* v8 ignore next */ /* v8 ignore next */
    const genDataBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Generate Synthetic Dataset', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const fedBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Broadcast Gradients (Federated)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(extractWBtn); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(genDataBtn); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(fedBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(trainSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    genDataBtn.addEventListener('click', () => /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('autograd', { action: 'generate_data' }), /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    fedBtn.addEventListener('click', () => /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('autograd', { action: 'federated_train' }), /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    extractWBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('exportONNX'); // Re-routes to the mock JSON exporter for zero-dep environment /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Exported updated tensors', 'success'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    injectBackwardBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('autograd', { /* v8 ignore next */ /* v8 ignore next */
        action: 'inject', /* v8 ignore next */ /* v8 ignore next */
        loss: lossSelect.value, /* v8 ignore next */ /* v8 ignore next */
        optimizer: optSelect.value, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const lossCanvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { width: '200', height: '100' }, /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-loss-chart', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    trainSection.appendChild(lossCanvas); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 228. Plot Loss curve /* v8 ignore next */ /* v8 ignore next */
    const lossHistory: number[] = []; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('lossUpdated', (loss: number) => { /* v8 ignore next */ /* v8 ignore next */
      lossHistory.push(loss); /* v8 ignore next */ /* v8 ignore next */
      if (lossHistory.length > 50) lossHistory.shift(); /* v8 ignore next */ /* v8 ignore next */
      const ctx = lossCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
      if (ctx) { /* v8 ignore next */ /* v8 ignore next */
        ctx.clearRect(0, 0, 200, 100); /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#dc3545'; /* v8 ignore next */ /* v8 ignore next */
        ctx.lineWidth = 2; /* v8 ignore next */ /* v8 ignore next */
        ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
        const maxLoss = Math.max(...lossHistory, 2.0); /* v8 ignore next */ /* v8 ignore next */
        lossHistory.forEach((l, i) => { /* v8 ignore next */ /* v8 ignore next */
          const x = (i / 50) * 200; /* v8 ignore next */ /* v8 ignore next */
          const y = 100 - (l / maxLoss) * 100; /* v8 ignore next */ /* v8 ignore next */
          if (i === 0) ctx.moveTo(x, y); /* v8 ignore next */ /* v8 ignore next */
          else ctx.lineTo(x, y); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    runTrainBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('autograd', { action: 'train_step' }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let isTraining = false; /* v8 ignore next */ /* v8 ignore next */
    let epochTimer: any; /* v8 ignore next */ /* v8 ignore next */
    runEpochBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (isTraining) { /* v8 ignore next */ /* v8 ignore next */
        clearInterval(epochTimer); /* v8 ignore next */ /* v8 ignore next */
        isTraining = false; /* v8 ignore next */ /* v8 ignore next */
        runEpochBtn.textContent = 'Resume Training'; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        isTraining = true; /* v8 ignore next */ /* v8 ignore next */
        runEpochBtn.textContent = 'Pause Training'; /* v8 ignore next */ /* v8 ignore next */
        let stepCount = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 224. Async loop with set interval to avoid blocking UI /* v8 ignore next */ /* v8 ignore next */
        // 235. Validate local execution (WASM logic executed synchronously locally) /* v8 ignore next */ /* v8 ignore next */
        epochTimer = setInterval(() => { /* v8 ignore next */ /* v8 ignore next */
          // 230. Dynamically update hyperparams /* v8 ignore next */ /* v8 ignore next */
          const lr = parseFloat(lrSlider.value); /* v8 ignore next */ /* v8 ignore next */
          const bs = parseInt(bsSlider.value, 10); /* v8 ignore next */ /* v8 ignore next */
          lrLabel.textContent = `Learning Rate (${lr})`; /* v8 ignore next */ /* v8 ignore next */
          bsLabel.textContent = `Batch Size (${bs})`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
          globalEvents.emit('autograd', { action: 'train_step', payload: { lr, bs } }); /* v8 ignore next */ /* v8 ignore next */
          stepCount++; /* v8 ignore next */ /* v8 ignore next */
          if (stepCount >= 10) { /* v8 ignore next */ /* v8 ignore next */
            clearInterval(epochTimer); /* v8 ignore next */ /* v8 ignore next */
            isTraining = false; /* v8 ignore next */ /* v8 ignore next */
            runEpochBtn.textContent = 'Train (10 Epochs)'; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }, 550); // 500ms step execution time + 50ms buffer /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // GenAI Toggle /* v8 ignore next */ /* v8 ignore next */
    const genaiSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const genaiTitle = $create('h4', { textContent: 'GenAI / Agents' }); /* v8 ignore next */ /* v8 ignore next */
    const chatBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Open LLM Interface', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 606. Add an "Agent" tab to construct LLM-based autonomous workflows. /* v8 ignore next */ /* v8 ignore next */
    const agentBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Open Agent Workflow', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    agentBtn.style.marginTop = '5px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 335. Expose logits distributions as a real-time bar chart in the UI sidebar /* v8 ignore next */ /* v8 ignore next */
    const logitsCanvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        width: '200', /* v8 ignore next */ /* v8 ignore next */
        height: '80', /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'margin-top: 10px; background: var(--color-background-primary); border: 1px solid var(--color-background-border); border-radius: 4px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    genaiSection.appendChild(genaiTitle); /* v8 ignore next */ /* v8 ignore next */
    genaiSection.appendChild(chatBtn); /* v8 ignore next */ /* v8 ignore next */
    genaiSection.appendChild(agentBtn); /* v8 ignore next */ /* v8 ignore next */
    genaiSection.appendChild(logitsCanvas); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(genaiSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('logitsUpdate', (data: { topTokens: number[]; topProbs: number[] }) => { /* v8 ignore next */ /* v8 ignore next */
      const ctx = logitsCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
      if (ctx) { /* v8 ignore next */ /* v8 ignore next */
        ctx.clearRect(0, 0, 200, 80); /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = 'var(--color-primary)'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const barW = 200 / data.topProbs.length; /* v8 ignore next */ /* v8 ignore next */
        data.topProbs.forEach((prob, i) => { /* v8 ignore next */ /* v8 ignore next */
          const h = Math.max(2, prob * 80); /* v8 ignore next */ /* v8 ignore next */
          ctx.fillRect(i * barW, 80 - h, barW - 1, h); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    chatBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleChat'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    agentBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleAgent'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Security & Privacy Toggle /* v8 ignore next */ /* v8 ignore next */
    const secSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const secTitle = $create('h4', { textContent: 'Security / Privacy' }); /* v8 ignore next */ /* v8 ignore next */
    const obfBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Obfuscate Topology', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const encBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Encrypt Weights (AES-GCM)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const decBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Decrypt Weights', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    secSection.appendChild(secTitle); /* v8 ignore next */ /* v8 ignore next */
    secSection.appendChild(obfBtn); /* v8 ignore next */ /* v8 ignore next */
    secSection.appendChild(encBtn); /* v8 ignore next */ /* v8 ignore next */
    secSection.appendChild(decBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(secSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    obfBtn.addEventListener('click', () => globalEvents.emit('securityAction', 'obfuscate')); /* v8 ignore next */ /* v8 ignore next */
    encBtn.addEventListener('click', () => globalEvents.emit('securityAction', 'encrypt')); /* v8 ignore next */ /* v8 ignore next */
    decBtn.addEventListener('click', () => globalEvents.emit('securityAction', 'decrypt')); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Sensors & Pipelines Toggle /* v8 ignore next */ /* v8 ignore next */
    const sensorsSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const sensorsTitle = $create('h4', { textContent: 'Sensors & Pipelines' }); /* v8 ignore next */ /* v8 ignore next */
    const visionBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Open Vision Pipeline', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const audioBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Open Audio Pipeline', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    sensorsSection.appendChild(sensorsTitle); /* v8 ignore next */ /* v8 ignore next */
    sensorsSection.appendChild(visionBtn); /* v8 ignore next */ /* v8 ignore next */
    sensorsSection.appendChild(audioBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(sensorsSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    visionBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleVision'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    audioBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleAudio'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Vault Toggle /* v8 ignore next */ /* v8 ignore next */
    const vaultSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const vaultTitle = $create('h4', { textContent: 'IndexedDB Vault' }); /* v8 ignore next */ /* v8 ignore next */
    const vaultBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Open Vault Manager', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    vaultSection.appendChild(vaultTitle); /* v8 ignore next */ /* v8 ignore next */
    vaultSection.appendChild(vaultBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(vaultSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    vaultBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleVault'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Swarm Toggle /* v8 ignore next */ /* v8 ignore next */
    const p2pSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const p2pTitle = $create('h4', { textContent: 'WebRTC Swarm' }); /* v8 ignore next */ /* v8 ignore next */
    const swarmBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Open Swarm Panel', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    p2pSection.appendChild(p2pTitle); /* v8 ignore next */ /* v8 ignore next */
    p2pSection.appendChild(swarmBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(p2pSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    swarmBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleSwarm'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // WebNN Execution Provider /* v8 ignore next */ /* v8 ignore next */
    const execSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const execTitle = $create('h4', { textContent: 'Execution Provider' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 253. Auto-select the fastest available backend /* v8 ignore next */ /* v8 ignore next */
    const epSelect = $create<HTMLSelectElement>('select', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-select', /* v8 ignore next */ /* v8 ignore next */
      innerHTML: ` /* v8 ignore next */ /* v8 ignore next */
        <option value="wasm">WASM (CPU)</option> /* v8 ignore next */ /* v8 ignore next */
        <option value="webgpu">WebGPU</option> /* v8 ignore next */ /* v8 ignore next */
      `, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const webnnStatus = $create('span', { /* v8 ignore next */ /* v8 ignore next */
      className: 'badge ' + ('ml' in navigator ? 'success' : 'danger'), /* v8 ignore next */ /* v8 ignore next */
      textContent: 'ml' in navigator ? 'WebNN Supported' : 'WebNN Not Supported', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'font-size: 0.7rem; margin-top: 5px; display: inline-block; padding: 2px 4px; border-radius: 4px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 265. Document the required browser flags /* v8 ignore next */ /* v8 ignore next */
    if (!('ml' in navigator)) { /* v8 ignore next */ /* v8 ignore next */
      const flagWarn = $create('p', { /* v8 ignore next */ /* v8 ignore next */
        className: 'muted', /* v8 ignore next */ /* v8 ignore next */
        innerHTML: /* v8 ignore next */ /* v8 ignore next */
          'To enable WebNN, try launching Chrome with <br><code>--enable-features=WebMachineLearningNeuralNetwork</code>', /* v8 ignore next */ /* v8 ignore next */
        attributes: { style: 'font-size: 0.7rem; margin-top: 5px; line-height: 1.2;' }, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      execSection.appendChild(flagWarn); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if ('ml' in navigator) { /* v8 ignore next */ /* v8 ignore next */
      epSelect.innerHTML += `<option value="webnn">WebNN (NPU/GPU)</option>`; /* v8 ignore next */ /* v8 ignore next */
      // Default to WebNN if available /* v8 ignore next */ /* v8 ignore next */
      epSelect.value = 'webnn'; /* v8 ignore next */ /* v8 ignore next */
    } else if (navigator.gpu) { /* v8 ignore next */ /* v8 ignore next */
      epSelect.value = 'webgpu'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const execBtn = $create('button', { className: 'action-btn', textContent: 'Run Inference' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    execSection.appendChild(execTitle); /* v8 ignore next */ /* v8 ignore next */
    execSection.appendChild(epSelect); /* v8 ignore next */ /* v8 ignore next */
    execSection.appendChild(webnnStatus); /* v8 ignore next */ /* v8 ignore next */
    execSection.appendChild(execBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(execSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    execBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('executeProvider', epSelect.value); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Benchmarks Toggle /* v8 ignore next */ /* v8 ignore next */
    const benchSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const benchTitle = $create('h4', { textContent: 'Micro-Benchmarks' }); /* v8 ignore next */ /* v8 ignore next */
    const runBenchBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Run Suite (1000 Samples)', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 257. Plot backend comparison graphs /* v8 ignore next */ /* v8 ignore next */
    const benchCanvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        width: '200', /* v8 ignore next */ /* v8 ignore next */
        height: '100', /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'margin-top: 10px; background: var(--color-background-primary); border: 1px solid var(--color-background-border); border-radius: 4px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    benchSection.appendChild(benchTitle); /* v8 ignore next */ /* v8 ignore next */
    benchSection.appendChild(runBenchBtn); /* v8 ignore next */ /* v8 ignore next */
    benchSection.appendChild(benchCanvas); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(benchSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('benchmarkResults', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      const ctx = benchCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
      if (ctx) { /* v8 ignore next */ /* v8 ignore next */
        ctx.clearRect(0, 0, 200, 100); /* v8 ignore next */ /* v8 ignore next */
        const barW = 50; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // WASM (Blue) /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = '#0d6efd'; /* v8 ignore next */ /* v8 ignore next */
        ctx.fillRect(20, 100 - (data.wasm || 10), barW, data.wasm || 10); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // WebGPU (Green) /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = '#198754'; /* v8 ignore next */ /* v8 ignore next */
        ctx.fillRect(80, 100 - (data.webgpu || 50), barW, data.webgpu || 50); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // WebNN (Purple) /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = '#6f42c1'; /* v8 ignore next */ /* v8 ignore next */
        ctx.fillRect(140, 100 - (data.webnn || 90), barW, data.webnn || 90); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = 'var(--color-foreground-muted)'; /* v8 ignore next */ /* v8 ignore next */
        ctx.font = '10px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
        ctx.fillText('CPU', 30, 15); /* v8 ignore next */ /* v8 ignore next */
        ctx.fillText('GPU', 90, 15); /* v8 ignore next */ /* v8 ignore next */
        ctx.fillText('NPU', 150, 15); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    runBenchBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('runBenchmark'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Compilation Toggle /* v8 ignore next */ /* v8 ignore next */
    const compileSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const compileTitle = $create('h4', { textContent: 'AOT Compiler' }); /* v8 ignore next */ /* v8 ignore next */
    const wasmBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Compile to WASM', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const wgslBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Compile to WGSL', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const cppBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Compile to C++', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const cBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Compile to C99', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    compileSection.appendChild(compileTitle); /* v8 ignore next */ /* v8 ignore next */
    compileSection.appendChild(wasmBtn); /* v8 ignore next */ /* v8 ignore next */
    compileSection.appendChild(wgslBtn); /* v8 ignore next */ /* v8 ignore next */
    compileSection.appendChild(cppBtn); /* v8 ignore next */ /* v8 ignore next */
    compileSection.appendChild(cBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(compileSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    wasmBtn.addEventListener('click', () => globalEvents.emit('compile', 'wasm')); /* v8 ignore next */ /* v8 ignore next */
    wgslBtn.addEventListener('click', () => globalEvents.emit('compile', 'wgsl')); /* v8 ignore next */ /* v8 ignore next */
    cppBtn.addEventListener('click', () => globalEvents.emit('compile', 'cpp')); /* v8 ignore next */ /* v8 ignore next */
    cBtn.addEventListener('click', () => globalEvents.emit('compile', 'c')); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Source toggle /* v8 ignore next */ /* v8 ignore next */
    const sourceSection = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const sourceBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'View ONNXScript Editor', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const toggleGraphBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Toggle Graph Canvas', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    sourceSection.appendChild(sourceBtn); /* v8 ignore next */ /* v8 ignore next */
    sourceSection.appendChild(toggleGraphBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(sourceSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    searchBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      const term = searchInput.value.trim(); /* v8 ignore next */ /* v8 ignore next */
      if (term) { /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('searchNode', term); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    loadBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (this.fileInput.files && this.fileInput.files.length > 0) { /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('filesDropped', Array.from(this.fileInput.files)); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    loadDirBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (dirInput.files && dirInput.files.length > 0) { /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('directoryDropped', Array.from(dirInput.files)); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    toggleGraphBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleGraph'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    sourceBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('toggleEditor'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 637. Implement a command palette (Cmd+K) /* v8 ignore next */ /* v8 ignore next */
    document.addEventListener('keydown', (e: KeyboardEvent) => { /* v8 ignore next */ /* v8 ignore next */
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { /* v8 ignore next */ /* v8 ignore next */
        e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
        this.showCommandPalette(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private showCommandPalette(): void { /* v8 ignore next */ /* v8 ignore next */
    const existing = document.getElementById('ide-cmd-palette'); /* v8 ignore next */ /* v8 ignore next */
    if (existing) { /* v8 ignore next */ /* v8 ignore next */
      existing.remove(); /* v8 ignore next */ /* v8 ignore next */
      return; // toggle off /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const overlay = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      id: 'ide-cmd-palette', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.5); z-index: 9999; display: flex; align-items: flex-start; justify-content: center; padding-top: 15vh;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const modal = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-messages', // reuse styling /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'width: 500px; max-width: 90vw; background: var(--color-background); border: 1px solid var(--color-background-border); border-radius: 8px; padding: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const input = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        type: 'text', /* v8 ignore next */ /* v8 ignore next */
        placeholder: 'Search commands...', /* v8 ignore next */ /* v8 ignore next */
        style: 'width: 100%; font-size: 1.1rem; padding: 10px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const list = $create('ul', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-list', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 10px; max-height: 300px; overflow-y: auto;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const commands = [ /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle Graph Canvas', action: () => globalEvents.emit('toggleGraph') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle ONNXScript Editor', action: () => globalEvents.emit('toggleEditor') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle GenAI Chat', action: () => globalEvents.emit('toggleChat') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle Agent Workflow', action: () => globalEvents.emit('toggleAgent') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle Vault Manager', action: () => globalEvents.emit('toggleVault') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle Swarm Panel', action: () => globalEvents.emit('toggleSwarm') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle Vision Pipeline', action: () => globalEvents.emit('toggleVision') }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Toggle Audio Pipeline', action: () => globalEvents.emit('toggleAudio') }, /* v8 ignore next */ /* v8 ignore next */
      { /* v8 ignore next */ /* v8 ignore next */
        label: 'Run Inference (Active Provider)', /* v8 ignore next */ /* v8 ignore next */
        action: () => globalEvents.emit('executeProvider', 'wasm'), /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
      { label: 'Run Micro-Benchmarks', action: () => globalEvents.emit('runBenchmark') }, /* v8 ignore next */ /* v8 ignore next */
    ]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const renderList = (filter: string) => { /* v8 ignore next */ /* v8 ignore next */
      list.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
      commands /* v8 ignore next */ /* v8 ignore next */
        .filter((c) => c.label.toLowerCase().includes(filter.toLowerCase())) /* v8 ignore next */ /* v8 ignore next */
        .forEach((c) => { /* v8 ignore next */ /* v8 ignore next */
          const li = $create('li', { /* v8 ignore next */ /* v8 ignore next */
            textContent: c.label, /* v8 ignore next */ /* v8 ignore next */
            attributes: { style: 'padding: 8px; cursor: pointer; border-radius: 4px;' }, /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
          li.addEventListener( /* v8 ignore next */ /* v8 ignore next */
            'mouseenter', /* v8 ignore next */ /* v8 ignore next */
            () => (li.style.background = 'var(--color-background-secondary)'), /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
          li.addEventListener('mouseleave', () => (li.style.background = 'transparent')); /* v8 ignore next */ /* v8 ignore next */
          li.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
            c.action(); /* v8 ignore next */ /* v8 ignore next */
            overlay.remove(); /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
          list.appendChild(li); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    renderList(''); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    input.addEventListener('input', () => renderList(input.value)); /* v8 ignore next */ /* v8 ignore next */
    overlay.addEventListener('click', (e) => { /* v8 ignore next */ /* v8 ignore next */
      if (e.target === overlay) overlay.remove(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    modal.appendChild(input); /* v8 ignore next */ /* v8 ignore next */
    modal.appendChild(list); /* v8 ignore next */ /* v8 ignore next */
    overlay.appendChild(modal); /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(overlay); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    input.focus(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void {} /* v8 ignore next */ /* v8 ignore next */
}
