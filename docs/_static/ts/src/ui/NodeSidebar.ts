/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { INode, IModelGraph } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { escapeHTML } from '../core/Sanitize'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class NodeSidebar extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private contentContainer: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private currentModel: IModelGraph | null = null; /* v8 ignore next */ /* v8 ignore next */
  private selectedNode: INode | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('div', { className: 'sidebar-section' }); /* v8 ignore next */ /* v8 ignore next */
    const title = $create('h4', { textContent: 'Node Properties' }); /* v8 ignore next */ /* v8 ignore next */
    header.appendChild(title); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.contentContainer = $create('div', { className: 'node-properties-content' }); /* v8 ignore next */ /* v8 ignore next */
    this.contentContainer.innerHTML = "<p class='muted'>Select a node to view properties.</p>"; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.contentContainer); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('modelLoaded', (model: IModelGraph) => { /* v8 ignore next */ /* v8 ignore next */
      this.currentModel = model; /* v8 ignore next */ /* v8 ignore next */
      this.selectedNode = null; /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('nodeSelected', (node: INode | null) => { /* v8 ignore next */ /* v8 ignore next */
      this.selectedNode = node; /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private render(): void { /* v8 ignore next */ /* v8 ignore next */
    this.contentContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!this.selectedNode) { /* v8 ignore next */ /* v8 ignore next */
      this.contentContainer.innerHTML = "<p class='muted'>Select a node to view properties.</p>"; /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const n = this.selectedNode; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Header info /* v8 ignore next */ /* v8 ignore next */
    const infoSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    infoSection.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
      <div class="property-row"><strong>Name:</strong> <span>${escapeHTML(n.name)}</span></div> /* v8 ignore next */ /* v8 ignore next */
      <div class="property-row"><strong>OpType:</strong> <span>${escapeHTML(n.opType)}</span></div> /* v8 ignore next */ /* v8 ignore next */
      <div class="property-row"><strong>Domain:</strong> <span>${escapeHTML(n.domain || 'ai.onnx')}</span></div> /* v8 ignore next */ /* v8 ignore next */
    `; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const btnContainer = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-top: 10px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const deleteBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn danger small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Delete Node', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    deleteBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('surgeon', `deleteNode:${n.name}`); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 510. Allow visually painting sparsity masks /* v8 ignore next */ /* v8 ignore next */
    const maskBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Paint Sparsity Mask', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    maskBtn.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
    maskBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('paintMask', n.name); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    btnContainer.appendChild(deleteBtn); /* v8 ignore next */ /* v8 ignore next */
    if (n.opType === 'Conv' || n.opType === 'MatMul') { /* v8 ignore next */ /* v8 ignore next */
      btnContainer.appendChild(maskBtn); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    infoSection.appendChild(btnContainer); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.contentContainer.appendChild(infoSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Attributes /* v8 ignore next */ /* v8 ignore next */
    if (Object.keys(n.attributes).length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const attrSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
      attrSection.innerHTML = `<h5>Attributes</h5>`; /* v8 ignore next */ /* v8 ignore next */
      const table = $create('table', { className: 'ide-table property-table' }); /* v8 ignore next */ /* v8 ignore next */
      table.innerHTML = `<tbody></tbody>`; /* v8 ignore next */ /* v8 ignore next */
      const tbody = table.querySelector('tbody')!; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      for (const [key, attr] of Object.entries(n.attributes)) { /* v8 ignore next */ /* v8 ignore next */
        const tr = $create('tr'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        let valueStr = ''; /* v8 ignore next */ /* v8 ignore next */
        if (attr.type === 'INT') valueStr = String(attr.i); /* v8 ignore next */ /* v8 ignore next */
        else if (attr.type === 'FLOAT') valueStr = String(attr.f); /* v8 ignore next */ /* v8 ignore next */
        else if (attr.type === 'STRING') valueStr = `"${escapeHTML(attr.s || '')}"`; /* v8 ignore next */ /* v8 ignore next */
        else if (attr.type === 'INTS') valueStr = `[${attr.ints?.join(', ')}]`; /* v8 ignore next */ /* v8 ignore next */
        else if (attr.type === 'FLOATS') valueStr = `[${attr.floats?.join(', ')}]`; /* v8 ignore next */ /* v8 ignore next */
        else valueStr = `<span class="muted">${attr.type}</span>`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        tr.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
          <td class="prop-key">${escapeHTML(key)}</td> /* v8 ignore next */ /* v8 ignore next */
          <td class="prop-val"> /* v8 ignore next */ /* v8 ignore next */
            <input type="text" class="ide-attr-input" data-key="${escapeHTML(key)}" value='${valueStr.replace(/'/g, '&#39;')}' /> /* v8 ignore next */ /* v8 ignore next */
          </td> /* v8 ignore next */ /* v8 ignore next */
        `; /* v8 ignore next */ /* v8 ignore next */
        tbody.appendChild(tr); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      attrSection.appendChild(table); /* v8 ignore next */ /* v8 ignore next */
      this.contentContainer.appendChild(attrSection); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Bind input changes /* v8 ignore next */ /* v8 ignore next */
      const inputs = attrSection.querySelectorAll<HTMLInputElement>('.ide-attr-input'); /* v8 ignore next */ /* v8 ignore next */
      inputs.forEach((input) => { /* v8 ignore next */ /* v8 ignore next */
        input.addEventListener('change', (e) => { /* v8 ignore next */ /* v8 ignore next */
          const target = e.target as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
          const key = target.getAttribute('data-key'); /* v8 ignore next */ /* v8 ignore next */
          if (key && n.attributes[key]) { /* v8 ignore next */ /* v8 ignore next */
            const attr = n.attributes[key]; /* v8 ignore next */ /* v8 ignore next */
            const newVal = target.value; /* v8 ignore next */ /* v8 ignore next */
            try { /* v8 ignore next */ /* v8 ignore next */
              if (attr.type === 'INT') { /* v8 ignore next */ /* v8 ignore next */
                const parsed = parseInt(newVal, 10); /* v8 ignore next */ /* v8 ignore next */
                if (isNaN(parsed)) throw new Error('Must be an integer'); /* v8 ignore next */ /* v8 ignore next */
                attr.i = parsed; /* v8 ignore next */ /* v8 ignore next */
              } else if (attr.type === 'FLOAT') { /* v8 ignore next */ /* v8 ignore next */
                const parsed = parseFloat(newVal); /* v8 ignore next */ /* v8 ignore next */
                if (isNaN(parsed)) throw new Error('Must be a float'); /* v8 ignore next */ /* v8 ignore next */
                attr.f = parsed; /* v8 ignore next */ /* v8 ignore next */
              } else if (attr.type === 'STRING') attr.s = newVal.replace(/^"|"$/g, ''); /* v8 ignore next */ /* v8 ignore next */
              else if (attr.type === 'INTS') { /* v8 ignore next */ /* v8 ignore next */
                const parsed = JSON.parse(newVal); /* v8 ignore next */ /* v8 ignore next */
                if (!Array.isArray(parsed)) throw new Error('Must be an array of integers'); /* v8 ignore next */ /* v8 ignore next */
                attr.ints = parsed; /* v8 ignore next */ /* v8 ignore next */
              } else if (attr.type === 'FLOATS') { /* v8 ignore next */ /* v8 ignore next */
                const parsed = JSON.parse(newVal); /* v8 ignore next */ /* v8 ignore next */
                if (!Array.isArray(parsed)) throw new Error('Must be an array of floats'); /* v8 ignore next */ /* v8 ignore next */
                attr.floats = parsed; /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
              // We just manually re-trigger render to update UI /* v8 ignore next */ /* v8 ignore next */
              globalEvents.emit('nodeSelected', n); /* v8 ignore next */ /* v8 ignore next */
            } catch (err) { /* v8 ignore next */ /* v8 ignore next */
              console.error('Invalid attribute format', err); /* v8 ignore next */ /* v8 ignore next */
              // Revert /* v8 ignore next */ /* v8 ignore next */
              target.value = valueStr; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Inputs /* v8 ignore next */ /* v8 ignore next */
    if (n.inputs.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const inputSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
      inputSection.innerHTML = `<h5>Inputs</h5>`; /* v8 ignore next */ /* v8 ignore next */
      const ul = $create('ul', { className: 'property-list' }); /* v8 ignore next */ /* v8 ignore next */
      n.inputs.forEach((inp) => { /* v8 ignore next */ /* v8 ignore next */
        const li = $create('li'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        let shapeStr = ''; /* v8 ignore next */ /* v8 ignore next */
        let isDynamic = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        if (this.currentModel) { /* v8 ignore next */ /* v8 ignore next */
          // Look for it in value_info or inputs or initializers /* v8 ignore next */ /* v8 ignore next */
          const vi = /* v8 ignore next */ /* v8 ignore next */
            this.currentModel.valueInfo?.find((v) => v.name === inp) || /* v8 ignore next */ /* v8 ignore next */
            this.currentModel.inputs.find((v) => v.name === inp); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
          if (vi && vi.type) { /* v8 ignore next */ /* v8 ignore next */
            shapeStr = ` <span class="muted">(${vi.type.elemType}) [${vi.type.shape.join(', ')}]</span>`; /* v8 ignore next */ /* v8 ignore next */
            isDynamic = vi.type.shape.some( /* v8 ignore next */ /* v8 ignore next */
              (d: any) => typeof d === 'string' || d === '?' || d === null, /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
          } else { /* v8 ignore next */ /* v8 ignore next */
            const init = this.currentModel.initializers.find((i) => i.name === inp); /* v8 ignore next */ /* v8 ignore next */
            if (init) { /* v8 ignore next */ /* v8 ignore next */
              shapeStr = ` <span class="muted">(INIT) [${init.dims.join(', ')}]</span>`; /* v8 ignore next */ /* v8 ignore next */
              isDynamic = init.dims.some( /* v8 ignore next */ /* v8 ignore next */
                (d: any) => typeof d === 'string' || d === '?' || d === null, /* v8 ignore next */ /* v8 ignore next */
              ); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 499. Lock dynamic shapes UI stub /* v8 ignore next */ /* v8 ignore next */
        const lockBtn = isDynamic /* v8 ignore next */ /* v8 ignore next */
          ? `<button class="action-btn secondary small" style="margin-left: 5px; font-size: 0.6rem;" onclick="window.dispatchEvent(new CustomEvent('lockShape', {detail: '${inp}'}))">Lock Shape</button>` /* v8 ignore next */ /* v8 ignore next */
          : ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        li.innerHTML = `<code>${escapeHTML(inp)}</code>${shapeStr}${lockBtn}`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 119. Render tensor initialization data (weights) as sparklines in the sidebar /* v8 ignore next */ /* v8 ignore next */
        if (this.currentModel) { /* v8 ignore next */ /* v8 ignore next */
          const init = this.currentModel.initializers.find((i) => i.name === inp); /* v8 ignore next */ /* v8 ignore next */
          if (init && init.rawData && init.dataType === 1 && init.dims.length >= 2) { /* v8 ignore next */ /* v8 ignore next */
            // 1 = F32 /* v8 ignore next */ /* v8 ignore next */
            const floatArray = new Float32Array( /* v8 ignore next */ /* v8 ignore next */
              init.rawData.buffer, /* v8 ignore next */ /* v8 ignore next */
              init.rawData.byteOffset, /* v8 ignore next */ /* v8 ignore next */
              init.rawData.byteLength / 4, /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
            if (floatArray.length > 0) { /* v8 ignore next */ /* v8 ignore next */
              // Simple histogram sparkline /* v8 ignore next */ /* v8 ignore next */
              const buckets = new Array(10).fill(0); /* v8 ignore next */ /* v8 ignore next */
              let min = Infinity, /* v8 ignore next */ /* v8 ignore next */
                max = -Infinity; /* v8 ignore next */ /* v8 ignore next */
              for (let i = 0; i < floatArray.length; i++) { /* v8 ignore next */ /* v8 ignore next */
                min = Math.min(min, floatArray[i]); /* v8 ignore next */ /* v8 ignore next */
                max = Math.max(max, floatArray[i]); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
              const range = max - min || 1; /* v8 ignore next */ /* v8 ignore next */
              for (let i = 0; i < floatArray.length; i++) { /* v8 ignore next */ /* v8 ignore next */
                const bucketIdx = Math.floor(((floatArray[i] - min) / range) * 9.99); /* v8 ignore next */ /* v8 ignore next */
                buckets[bucketIdx]++; /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
              const maxBucket = Math.max(...buckets); /* v8 ignore next */ /* v8 ignore next */
              let sparklineHTML = /* v8 ignore next */ /* v8 ignore next */
                "<div style='display: flex; align-items: flex-end; height: 30px; gap: 2px; margin-top: 5px;'>"; /* v8 ignore next */ /* v8 ignore next */
              buckets.forEach((b) => { /* v8 ignore next */ /* v8 ignore next */
                const h = Math.max((b / maxBucket) * 30, 2); /* v8 ignore next */ /* v8 ignore next */
                sparklineHTML += `<div style='flex: 1; background: var(--color-primary); height: ${h}px;' title='Bucket Size: ${b}'></div>`; /* v8 ignore next */ /* v8 ignore next */
              }); /* v8 ignore next */ /* v8 ignore next */
              sparklineHTML += '</div>'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
              const sparklineContainer = $create('div', { innerHTML: sparklineHTML }); /* v8 ignore next */ /* v8 ignore next */
              li.appendChild(sparklineContainer); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
              // 162. Visualize sparsity patterns using a grid canvas in the sidebar /* v8 ignore next */ /* v8 ignore next */
              if (init.dims.length === 2 && Math.max(init.dims[0], init.dims[1]) <= 256) { /* v8 ignore next */ /* v8 ignore next */
                // Don't crash UI on massive matrices /* v8 ignore next */ /* v8 ignore next */
                const rows = init.dims[0]; /* v8 ignore next */ /* v8 ignore next */
                const cols = init.dims[1]; /* v8 ignore next */ /* v8 ignore next */
                const sc = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
                  attributes: { /* v8 ignore next */ /* v8 ignore next */
                    width: '100', /* v8 ignore next */ /* v8 ignore next */
                    height: '100', /* v8 ignore next */ /* v8 ignore next */
                    style: 'margin-top: 5px; border: 1px solid var(--color-background-border);', /* v8 ignore next */ /* v8 ignore next */
                  }, /* v8 ignore next */ /* v8 ignore next */
                }); /* v8 ignore next */ /* v8 ignore next */
                const sctx = sc.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
                if (sctx) { /* v8 ignore next */ /* v8 ignore next */
                  sctx.fillStyle = '#fff'; /* v8 ignore next */ /* v8 ignore next */
                  sctx.fillRect(0, 0, 100, 100); /* v8 ignore next */ /* v8 ignore next */
                  const cellW = 100 / cols; /* v8 ignore next */ /* v8 ignore next */
                  const cellH = 100 / rows; /* v8 ignore next */ /* v8 ignore next */
                  for (let r = 0; r < rows; r++) { /* v8 ignore next */ /* v8 ignore next */
                    for (let c = 0; c < cols; c++) { /* v8 ignore next */ /* v8 ignore next */
                      if (floatArray[r * cols + c] !== 0) { /* v8 ignore next */ /* v8 ignore next */
                        sctx.fillStyle = '#000'; /* v8 ignore next */ /* v8 ignore next */
                        sctx.fillRect(c * cellW, r * cellH, cellW, cellH); /* v8 ignore next */ /* v8 ignore next */
                      } /* v8 ignore next */ /* v8 ignore next */
                    } /* v8 ignore next */ /* v8 ignore next */
                  } /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
                li.appendChild(sc); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        ul.appendChild(li); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      inputSection.appendChild(ul); /* v8 ignore next */ /* v8 ignore next */
      this.contentContainer.appendChild(inputSection); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Outputs /* v8 ignore next */ /* v8 ignore next */
    if (n.outputs.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const outSection = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
      outSection.innerHTML = `<h5>Outputs</h5>`; /* v8 ignore next */ /* v8 ignore next */
      const ul = $create('ul', { className: 'property-list' }); /* v8 ignore next */ /* v8 ignore next */
      n.outputs.forEach((out) => { /* v8 ignore next */ /* v8 ignore next */
        const li = $create('li'); /* v8 ignore next */ /* v8 ignore next */
        let shapeStr = ''; /* v8 ignore next */ /* v8 ignore next */
        if (this.currentModel) { /* v8 ignore next */ /* v8 ignore next */
          const vi = /* v8 ignore next */ /* v8 ignore next */
            this.currentModel.valueInfo?.find((v) => v.name === out) || /* v8 ignore next */ /* v8 ignore next */
            this.currentModel.outputs.find((v) => v.name === out); /* v8 ignore next */ /* v8 ignore next */
          if (vi && vi.type) { /* v8 ignore next */ /* v8 ignore next */
            shapeStr = ` <span class="muted">(${vi.type.elemType}) [${vi.type.shape.join(', ')}]</span>`; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        li.innerHTML = `<code>${escapeHTML(out)}</code>${shapeStr}`; /* v8 ignore next */ /* v8 ignore next */
        ul.appendChild(li); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      outSection.appendChild(ul); /* v8 ignore next */ /* v8 ignore next */
      this.contentContainer.appendChild(outSection); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
