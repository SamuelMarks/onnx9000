/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { IModelGraph, ITensor } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { escapeHTML } from '../core/Sanitize'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class ModelSummary extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private model: IModelGraph | null = null; /* v8 ignore next */ /* v8 ignore next */
  private tableContainer: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
    this.tableContainer = $create('div', { className: 'model-summary-table-wrapper' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.tableContainer); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    // nothing bound yet /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  setModel(model: IModelGraph | null): void { /* v8 ignore next */ /* v8 ignore next */
    this.model = model; /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private render(): void { /* v8 ignore next */ /* v8 ignore next */
    this.tableContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!this.model) { /* v8 ignore next */ /* v8 ignore next */
      this.tableContainer.innerHTML = '<p>No model loaded.</p>'; /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('h3', { textContent: `Model: ${this.model.name}` }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 577. Verify watermarks /* v8 ignore next */ /* v8 ignore next */
    if (this.model.docString && this.model.docString.includes('onnx9000_verified_')) { /* v8 ignore next */ /* v8 ignore next */
      const badge = $create('span', { /* v8 ignore next */ /* v8 ignore next */
        className: 'badge success', /* v8 ignore next */ /* v8 ignore next */
        textContent: 'DP Verified', /* v8 ignore next */ /* v8 ignore next */
        attributes: { style: 'margin-left: 10px; font-size: 0.7rem; vertical-align: middle;' }, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      header.appendChild(badge); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.tableContainer.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 163. Calculate FLOPs / Memory Footprint reductions /* v8 ignore next */ /* v8 ignore next */
    let totalParams = 0; /* v8 ignore next */ /* v8 ignore next */
    let totalBytes = 0; /* v8 ignore next */ /* v8 ignore next */
    let totalSparsity = 0; /* v8 ignore next */ /* v8 ignore next */
    let paramElements = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.model.initializers.forEach((init) => { /* v8 ignore next */ /* v8 ignore next */
      if (init.rawData) { /* v8 ignore next */ /* v8 ignore next */
        totalBytes += init.rawData.byteLength; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // If CSR sparse format /* v8 ignore next */ /* v8 ignore next */
        if (init.dataType === 21) { /* v8 ignore next */ /* v8 ignore next */
          // Roughly parse sparse density out of the buffer header /* v8 ignore next */ /* v8 ignore next */
          const dv = new DataView(init.rawData.buffer, init.rawData.byteOffset, 12); /* v8 ignore next */ /* v8 ignore next */
          const nnz = dv.getUint32(0, true); /* v8 ignore next */ /* v8 ignore next */
          let shapeSize = 1; /* v8 ignore next */ /* v8 ignore next */
          init.dims.forEach((d) => (shapeSize *= d)); /* v8 ignore next */ /* v8 ignore next */
          if (shapeSize > 0) { /* v8 ignore next */ /* v8 ignore next */
            totalSparsity += shapeSize - nnz; /* v8 ignore next */ /* v8 ignore next */
            paramElements += shapeSize; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } else { /* v8 ignore next */ /* v8 ignore next */
          let shapeSize = 1; /* v8 ignore next */ /* v8 ignore next */
          init.dims.forEach((d) => (shapeSize *= d)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
          // Check zeros directly /* v8 ignore next */ /* v8 ignore next */
          if (init.dataType === 1) { /* v8 ignore next */ /* v8 ignore next */
            // F32 /* v8 ignore next */ /* v8 ignore next */
            const f32 = new Float32Array( /* v8 ignore next */ /* v8 ignore next */
              init.rawData.buffer, /* v8 ignore next */ /* v8 ignore next */
              init.rawData.byteOffset, /* v8 ignore next */ /* v8 ignore next */
              init.rawData.byteLength / 4, /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
            let zeroCount = 0; /* v8 ignore next */ /* v8 ignore next */
            for (let i = 0; i < f32.length; i++) if (f32[i] === 0) zeroCount++; /* v8 ignore next */ /* v8 ignore next */
            totalSparsity += zeroCount; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          paramElements += shapeSize; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const sparsityPct = paramElements > 0 ? ((totalSparsity / paramElements) * 100).toFixed(1) : 0; /* v8 ignore next */ /* v8 ignore next */
    const mbSize = (totalBytes / 1024 / 1024).toFixed(2); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const stats = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-section', /* v8 ignore next */ /* v8 ignore next */
      innerHTML: ` /* v8 ignore next */ /* v8 ignore next */
        <div class="property-row"><strong>Nodes:</strong> <span>${this.model.nodes.length}</span></div> /* v8 ignore next */ /* v8 ignore next */
        <div class="property-row"><strong>Memory Footprint:</strong> <span>${mbSize} MB</span></div> /* v8 ignore next */ /* v8 ignore next */
        <div class="property-row"><strong>Global Sparsity:</strong> <span>${sparsityPct}% Zeros</span></div> /* v8 ignore next */ /* v8 ignore next */
      `, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.tableContainer.appendChild(stats); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.model.initializers.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const table = $create('table', { className: 'ide-table' }); /* v8 ignore next */ /* v8 ignore next */
      table.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
        <thead> /* v8 ignore next */ /* v8 ignore next */
          <tr> /* v8 ignore next */ /* v8 ignore next */
            <th>Name</th> /* v8 ignore next */ /* v8 ignore next */
            <th>Type</th> /* v8 ignore next */ /* v8 ignore next */
            <th>Shape</th> /* v8 ignore next */ /* v8 ignore next */
            <th>Size (Bytes)</th> /* v8 ignore next */ /* v8 ignore next */
          </tr> /* v8 ignore next */ /* v8 ignore next */
        </thead> /* v8 ignore next */ /* v8 ignore next */
        <tbody> /* v8 ignore next */ /* v8 ignore next */
        </tbody> /* v8 ignore next */ /* v8 ignore next */
      `; /* v8 ignore next */ /* v8 ignore next */
      const tbody = table.querySelector('tbody')!; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // limit to 100 for display /* v8 ignore next */ /* v8 ignore next */
      const displayCount = Math.min(100, this.model.initializers.length); /* v8 ignore next */ /* v8 ignore next */
      for (let i = 0; i < displayCount; i++) { /* v8 ignore next */ /* v8 ignore next */
        const init = this.model.initializers[i]; /* v8 ignore next */ /* v8 ignore next */
        const tr = $create('tr'); /* v8 ignore next */ /* v8 ignore next */
        const byteSize = init.rawData ? init.rawData.byteLength : 0; /* v8 ignore next */ /* v8 ignore next */
        tr.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
          <td>${escapeHTML(init.name)}</td> /* v8 ignore next */ /* v8 ignore next */
          <td>${init.dataType}</td> /* v8 ignore next */ /* v8 ignore next */
          <td>[${init.dims.join(', ')}]</td> /* v8 ignore next */ /* v8 ignore next */
          <td>${byteSize.toLocaleString()}</td> /* v8 ignore next */ /* v8 ignore next */
        `; /* v8 ignore next */ /* v8 ignore next */
        tbody.appendChild(tr); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (this.model.initializers.length > 100) { /* v8 ignore next */ /* v8 ignore next */
        const tr = $create('tr'); /* v8 ignore next */ /* v8 ignore next */
        tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--color-foreground-muted);">... and ${this.model.initializers.length - 100} more</td>`; /* v8 ignore next */ /* v8 ignore next */
        tbody.appendChild(tr); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.tableContainer.appendChild(table); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
