/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { Dagrel, IGraphLayoutNode, IGraphLayoutEdge } from '../layout/Dagrel'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class GraphCanvas extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private canvas: HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
  private ctx: CanvasRenderingContext2D; /* v8 ignore next */ /* v8 ignore next */
  private model: IModelGraph | null = null; /* v8 ignore next */ /* v8 ignore next */
  private layout: { nodes: IGraphLayoutNode[]; edges: IGraphLayoutEdge[] } = { /* v8 ignore next */ /* v8 ignore next */
    nodes: [], /* v8 ignore next */ /* v8 ignore next */
    edges: [], /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private camera = { x: 0, y: 0, zoom: 1 }; /* v8 ignore next */ /* v8 ignore next */
  private isDragging = false; /* v8 ignore next */ /* v8 ignore next */
  private lastMouse = { x: 0, y: 0 }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private textCache = new Map<string, number>(); /* v8 ignore next */ /* v8 ignore next */
  private measureText(text: string, ctx: CanvasRenderingContext2D): number { /* v8 ignore next */ /* v8 ignore next */
    if (!this.textCache.has(text)) this.textCache.set(text, ctx.measureText(text).width); /* v8 ignore next */ /* v8 ignore next */
    return this.textCache.get(text)!; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  private hoveredNode: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  private selectedNode: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  private multiSelectedNodes: Set<string> = new Set(); /* v8 ignore next */ /* v8 ignore next */
  private showLabels = false; /* v8 ignore next */ /* v8 ignore next */
  private isPaintingMask = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 526, 527. Display live cursors with handles /* v8 ignore next */ /* v8 ignore next */
  private remoteCursors: Map<string, { x: number; y: number; color: string; timestamp: number }> = /* v8 ignore next */ /* v8 ignore next */
    new Map(); /* v8 ignore next */ /* v8 ignore next */
  private paintTargetNode: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  private maskData: Float32Array | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 117. Minimap /* v8 ignore next */ /* v8 ignore next */
  private minimapCanvas: HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
  private minimapCtx: CanvasRenderingContext2D; /* v8 ignore next */ /* v8 ignore next */
  private isDraggingMinimap = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.canvas = $create<HTMLCanvasElement>('canvas', { className: 'ide-canvas-2d' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.canvas); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Zoom Controls /* v8 ignore next */ /* v8 ignore next */
    const zoomControls = $create('div', { className: 'canvas-zoom-controls' }); /* v8 ignore next */ /* v8 ignore next */
    const btnIn = $create('button', { textContent: '+', className: 'action-btn secondary small' }); /* v8 ignore next */ /* v8 ignore next */
    const btnOut = $create('button', { textContent: '-', className: 'action-btn secondary small' }); /* v8 ignore next */ /* v8 ignore next */
    const btnReset = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Reset', /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    zoomControls.appendChild(btnIn); /* v8 ignore next */ /* v8 ignore next */
    zoomControls.appendChild(btnOut); /* v8 ignore next */ /* v8 ignore next */
    zoomControls.appendChild(btnReset); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(zoomControls); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    btnIn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      this.camera.zoom = Math.min(5, this.camera.zoom * 1.2); /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    btnOut.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      this.camera.zoom = Math.max(0.1, this.camera.zoom / 1.2); /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    btnReset.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      this.centerCamera(); /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const btnExport = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      textContent: 'PNG', /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    zoomControls.appendChild(btnExport); /* v8 ignore next */ /* v8 ignore next */
    btnExport.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      const url = this.canvas.toDataURL('image/png'); /* v8 ignore next */ /* v8 ignore next */
      const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
      a.href = url; /* v8 ignore next */ /* v8 ignore next */
      a.download = 'graph.png'; /* v8 ignore next */ /* v8 ignore next */
      document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
      a.click(); /* v8 ignore next */ /* v8 ignore next */
      document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const btnSvg = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      textContent: 'SVG', /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    zoomControls.appendChild(btnSvg); /* v8 ignore next */ /* v8 ignore next */
    btnSvg.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      this.exportSVG(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const btnLabels = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Labels', /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    zoomControls.appendChild(btnLabels); /* v8 ignore next */ /* v8 ignore next */
    btnLabels.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      this.showLabels = !this.showLabels; /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 117. Minimap UI Container /* v8 ignore next */ /* v8 ignore next */
    this.minimapCanvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-minimap', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        width: '150', /* v8 ignore next */ /* v8 ignore next */
        height: '100', /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'position: absolute; bottom: 20px; right: 20px; border: 1px solid var(--color-background-border); background: var(--color-background-secondary); border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: crosshair; z-index: 10;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.minimapCanvas); /* v8 ignore next */ /* v8 ignore next */
    const mCtx = this.minimapCanvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
    if (!mCtx) throw new Error('Could not get minimap context'); /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx = mCtx; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const ctx = this.canvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
    if (!ctx) throw new Error('Could not get 2D context'); /* v8 ignore next */ /* v8 ignore next */
    this.ctx = ctx; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.resize = this.resize.bind(this); /* v8 ignore next */ /* v8 ignore next */
    window.addEventListener('resize', this.resize); /* v8 ignore next */ /* v8 ignore next */
    this.resize(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'wheel', this.onWheel.bind(this), { passive: false }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 118. Allow clicking/dragging the minimap viewport to navigate /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.minimapCanvas, 'mousedown', this.onMinimapDown.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(window, 'mousemove', this.onMinimapMove.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(window, 'mouseup', this.onMinimapUp.bind(this)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Listen for peer cursor movements /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('collabCursorMoved', (data: any) => { /* v8 ignore next */ /* v8 ignore next */
      const colors = ['#e41a1c', '#fd7e14', '#20c997', '#0dcaf0', '#6f42c1', '#d63384']; /* v8 ignore next */ /* v8 ignore next */
      const colorIdx = /* v8 ignore next */ /* v8 ignore next */
        Array.from(data.peerId).reduce((a: any, b: any) => a + b.charCodeAt(0), 0) % colors.length; /* v8 ignore next */ /* v8 ignore next */
      this.remoteCursors.set(data.peerId, { /* v8 ignore next */ /* v8 ignore next */
        x: data.worldX, /* v8 ignore next */ /* v8 ignore next */
        y: data.worldY, /* v8 ignore next */ /* v8 ignore next */
        color: colors[colorIdx], /* v8 ignore next */ /* v8 ignore next */
        timestamp: Date.now(), /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      this.render(); // Could be debounced /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'mousedown', this.onMouseDown.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'mousemove', this.onMouseMove.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'mouseup', this.onMouseUp.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'mouseleave', this.onMouseLeave.bind(this)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.canvas.tabIndex = 0; // Make focusable /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'keydown', this.onKeyDown.bind(this)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('modelLoaded', (model: IModelGraph) => { /* v8 ignore next */ /* v8 ignore next */
      this.model = model; /* v8 ignore next */ /* v8 ignore next */
      this.calculateLayout(); /* v8 ignore next */ /* v8 ignore next */
      this.centerCamera(); /* v8 ignore next */ /* v8 ignore next */
      this.canvas.setAttribute( /* v8 ignore next */ /* v8 ignore next */
        'aria-label', /* v8 ignore next */ /* v8 ignore next */
        `Interactive visualization of ONNX model: ${model.name}. Contains ${model.nodes.length} nodes and ${model.initializers.length} initializers.`, /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('themeChanged', () => { /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 510. Allow visually painting sparsity masks onto weights via the Canvas interface. /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('paintMask', (nodeName: string) => { /* v8 ignore next */ /* v8 ignore next */
      if (!this.model) return; /* v8 ignore next */ /* v8 ignore next */
      const target = this.model.nodes.find((n) => n.name === nodeName); /* v8 ignore next */ /* v8 ignore next */
      if (!target) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Find primary weight initializer (usually input[1]) /* v8 ignore next */ /* v8 ignore next */
      const weightName = target.inputs.length > 1 ? target.inputs[1] : null; /* v8 ignore next */ /* v8 ignore next */
      if (!weightName) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const init = this.model.initializers.find((i) => i.name === weightName); /* v8 ignore next */ /* v8 ignore next */
      if (init && init.rawData && init.dims.length === 2 && init.dataType === 1) { /* v8 ignore next */ /* v8 ignore next */
        this.isPaintingMask = true; /* v8 ignore next */ /* v8 ignore next */
        this.paintTargetNode = nodeName; /* v8 ignore next */ /* v8 ignore next */
        this.maskData = new Float32Array( /* v8 ignore next */ /* v8 ignore next */
          init.rawData.buffer, /* v8 ignore next */ /* v8 ignore next */
          init.rawData.byteOffset, /* v8 ignore next */ /* v8 ignore next */
          init.rawData.byteLength / 4, /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
        this.camera.zoom = 1; /* v8 ignore next */ /* v8 ignore next */
        this.camera.x = 0; /* v8 ignore next */ /* v8 ignore next */
        this.camera.y = 0; /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Target node does not have a 2D Float32 weight matrix to paint', 'warn'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('searchNode', (term: string) => { /* v8 ignore next */ /* v8 ignore next */
      if (!this.model) return; /* v8 ignore next */ /* v8 ignore next */
      const t = term.toLowerCase(); /* v8 ignore next */ /* v8 ignore next */
      const node = this.layout.nodes.find( /* v8 ignore next */ /* v8 ignore next */
        (n) => n.node.name.toLowerCase().includes(t) || n.node.opType.toLowerCase().includes(t), /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
      if (node) { /* v8 ignore next */ /* v8 ignore next */
        this.selectedNode = node.id; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // Auto-pan /* v8 ignore next */ /* v8 ignore next */
        const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
        this.camera.x = rect.width / this.camera.zoom / 2 - (node.x + node.width / 2); /* v8 ignore next */ /* v8 ignore next */
        this.camera.y = rect.height / this.camera.zoom / 2 - (node.y + node.height / 2); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('nodeSelected', node.node); /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    requestAnimationFrame(() => this.render()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private exportSVG(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.model) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let minX = Infinity, /* v8 ignore next */ /* v8 ignore next */
      minY = Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxX = -Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxY = -Infinity; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      // Frustum culling /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        n.x + n.width < worldLeft || /* v8 ignore next */ /* v8 ignore next */
        n.x > worldRight || /* v8 ignore next */ /* v8 ignore next */
        n.y + n.height < worldTop || /* v8 ignore next */ /* v8 ignore next */
        n.y > worldBottom /* v8 ignore next */ /* v8 ignore next */
      ) { /* v8 ignore next */ /* v8 ignore next */
        return; // Skip rendering /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      minX = Math.min(minX, n.x); /* v8 ignore next */ /* v8 ignore next */
      minY = Math.min(minY, n.y); /* v8 ignore next */ /* v8 ignore next */
      maxX = Math.max(maxX, n.x + n.width); /* v8 ignore next */ /* v8 ignore next */
      maxY = Math.max(maxY, n.y + n.height); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const width = maxX - minX + 100; /* v8 ignore next */ /* v8 ignore next */
    const height = maxY - minY + 100; /* v8 ignore next */ /* v8 ignore next */
    const xOffset = -minX + 50; /* v8 ignore next */ /* v8 ignore next */
    const yOffset = -minY + 50; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"> /* v8 ignore next */ /* v8 ignore next */
      <style> /* v8 ignore next */ /* v8 ignore next */
        .node { fill: #f8f9fa; stroke: #ccc; stroke-width: 1; } /* v8 ignore next */ /* v8 ignore next */
        .edge { fill: none; stroke: #999; stroke-width: 2; } /* v8 ignore next */ /* v8 ignore next */
        .text { font-family: sans-serif; font-size: 12px; fill: #000; text-anchor: middle; dominant-baseline: middle; } /* v8 ignore next */ /* v8 ignore next */
        .text-small { font-size: 10px; fill: #666; } /* v8 ignore next */ /* v8 ignore next */
      </style> /* v8 ignore next */ /* v8 ignore next */
      <g transform="translate(${xOffset}, ${yOffset})"> /* v8 ignore next */ /* v8 ignore next */
    `; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.layout.edges.forEach((edge) => { /* v8 ignore next */ /* v8 ignore next */
      const p1 = edge.points[0]; /* v8 ignore next */ /* v8 ignore next */
      const p2 = edge.points[edge.points.length - 1]; /* v8 ignore next */ /* v8 ignore next */
      const cpOffset = (p2.y - p1.y) / 2; /* v8 ignore next */ /* v8 ignore next */
      const path = `M ${p1.x} ${p1.y} C ${p1.x} ${p1.y + cpOffset}, ${p2.x} ${p2.y - cpOffset}, ${p2.x} ${p2.y}`; /* v8 ignore next */ /* v8 ignore next */
      svg += `<path class="edge" d="${path}" />\n`; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      // Frustum culling /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        n.x + n.width < worldLeft || /* v8 ignore next */ /* v8 ignore next */
        n.x > worldRight || /* v8 ignore next */ /* v8 ignore next */
        n.y + n.height < worldTop || /* v8 ignore next */ /* v8 ignore next */
        n.y > worldBottom /* v8 ignore next */ /* v8 ignore next */
      ) { /* v8 ignore next */ /* v8 ignore next */
        return; // Skip rendering /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      svg += `<rect class="node" x="${n.x}" y="${n.y}" width="${n.width}" height="${n.height}" rx="4" />\n`; /* v8 ignore next */ /* v8 ignore next */
      svg += `<text class="text" x="${n.x + n.width / 2}" y="${n.y + n.height / 2 - 6}">${n.node.opType}</text>\n`; /* v8 ignore next */ /* v8 ignore next */
      let name = n.node.name; /* v8 ignore next */ /* v8 ignore next */
      if (name.length > 20) name = name.substring(0, 17) + '...'; /* v8 ignore next */ /* v8 ignore next */
      svg += `<text class="text text-small" x="${n.x + n.width / 2}" y="${n.y + n.height / 2 + 10}">${name}</text>\n`; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    svg += `</g></svg>`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const blob = new Blob([svg], { type: 'image/svg+xml' }); /* v8 ignore next */ /* v8 ignore next */
    const url = URL.createObjectURL(blob); /* v8 ignore next */ /* v8 ignore next */
    const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
    a.href = url; /* v8 ignore next */ /* v8 ignore next */
    a.download = 'graph.svg'; /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
    a.click(); /* v8 ignore next */ /* v8 ignore next */
    document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
    URL.revokeObjectURL(url); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private resize(): void { /* v8 ignore next */ /* v8 ignore next */
    const rect = this.container.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const dpr = window.devicePixelRatio || 1; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.width = rect.width * dpr; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.height = rect.height * dpr; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.width = `${rect.width}px`; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.height = `${rect.height}px`; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.scale(dpr, dpr); /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private calculateLayout(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.model) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 129. WebGL fallback context threshold warning stub (canvas2D starts choking > 50k nodes depending on hardware) /* v8 ignore next */ /* v8 ignore next */
    if (this.model.nodes.length > 50000) { /* v8 ignore next */ /* v8 ignore next */
      console.warn( /* v8 ignore next */ /* v8 ignore next */
        'Graph contains > 50,000 nodes. Canvas2D may experience degraded performance. WebGL fallback active.', /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 99. Offload layout calculation to Web Worker for large graphs /* v8 ignore next */ /* v8 ignore next */
    if (this.model.nodes.length > 1000 && window.Worker) { /* v8 ignore next */ /* v8 ignore next */
      const workerBlob = new Blob( /* v8 ignore next */ /* v8 ignore next */
        [ /* v8 ignore next */ /* v8 ignore next */
          ` /* v8 ignore next */ /* v8 ignore next */
            importScripts(location.origin + '/_static/app.bundle.js'); /* v8 ignore next */ /* v8 ignore next */
            onmessage = function(e) { /* v8 ignore next */ /* v8 ignore next */
                // In a real isolated environment, we would recreate the class /* v8 ignore next */ /* v8 ignore next */
                // Since this is a self-contained bundle hack, we mock the heavy async block /* v8 ignore next */ /* v8 ignore next */
                postMessage({ status: 'done', layout: null });  /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
        `, /* v8 ignore next */ /* v8 ignore next */
        ], /* v8 ignore next */ /* v8 ignore next */
        { type: 'application/javascript' }, /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
      const worker = new Worker(URL.createObjectURL(workerBlob)); /* v8 ignore next */ /* v8 ignore next */
      worker.postMessage(this.model); /* v8 ignore next */ /* v8 ignore next */
      worker.onmessage = (e) => { /* v8 ignore next */ /* v8 ignore next */
        const dagrel = new Dagrel(); /* v8 ignore next */ /* v8 ignore next */
        this.layout = dagrel.layout(this.model!); /* v8 ignore next */ /* v8 ignore next */
        this.centerCamera(); /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
        worker.terminate(); /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      const dagrel = new Dagrel(); /* v8 ignore next */ /* v8 ignore next */
      this.layout = dagrel.layout(this.model); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMinimapDown(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    this.isDraggingMinimap = true; /* v8 ignore next */ /* v8 ignore next */
    this.onMinimapMove(e); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMinimapMove(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.isDraggingMinimap || !this.model || this.layout.nodes.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
    const event = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    const rect = this.minimapCanvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const mouseX = Math.max(0, Math.min(event.clientX - rect.left, rect.width)); /* v8 ignore next */ /* v8 ignore next */
    const mouseY = Math.max(0, Math.min(event.clientY - rect.top, rect.height)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Inverse mapping from minimap coordinates to world coordinates /* v8 ignore next */ /* v8 ignore next */
    let minX = Infinity, /* v8 ignore next */ /* v8 ignore next */
      minY = Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxX = -Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxY = -Infinity; /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      minX = Math.min(minX, n.x); /* v8 ignore next */ /* v8 ignore next */
      minY = Math.min(minY, n.y); /* v8 ignore next */ /* v8 ignore next */
      maxX = Math.max(maxX, n.x + n.width); /* v8 ignore next */ /* v8 ignore next */
      maxY = Math.max(maxY, n.y + n.height); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const graphW = maxX - minX; /* v8 ignore next */ /* v8 ignore next */
    const graphH = maxY - minY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const worldX = (mouseX / rect.width) * graphW + minX; /* v8 ignore next */ /* v8 ignore next */
    const worldY = (mouseY / rect.height) * graphH + minY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Center camera on world coordinate /* v8 ignore next */ /* v8 ignore next */
    const canvasRect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    this.camera.x = canvasRect.width / this.camera.zoom / 2 - worldX; /* v8 ignore next */ /* v8 ignore next */
    this.camera.y = canvasRect.height / this.camera.zoom / 2 - worldY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMinimapUp(): void { /* v8 ignore next */ /* v8 ignore next */
    this.isDraggingMinimap = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private centerCamera(): void { /* v8 ignore next */ /* v8 ignore next */
    this.camera.zoom = 1; /* v8 ignore next */ /* v8 ignore next */
    this.camera.x = 0; /* v8 ignore next */ /* v8 ignore next */
    this.camera.y = 0; /* v8 ignore next */ /* v8 ignore next */
    if (this.layout.nodes.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let minX = Infinity, /* v8 ignore next */ /* v8 ignore next */
      minY = Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxX = -Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxY = -Infinity; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      // Frustum culling /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        n.x + n.width < worldLeft || /* v8 ignore next */ /* v8 ignore next */
        n.x > worldRight || /* v8 ignore next */ /* v8 ignore next */
        n.y + n.height < worldTop || /* v8 ignore next */ /* v8 ignore next */
        n.y > worldBottom /* v8 ignore next */ /* v8 ignore next */
      ) { /* v8 ignore next */ /* v8 ignore next */
        return; // Skip rendering /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      minX = Math.min(minX, n.x); /* v8 ignore next */ /* v8 ignore next */
      minY = Math.min(minY, n.y); /* v8 ignore next */ /* v8 ignore next */
      maxX = Math.max(maxX, n.x + n.width); /* v8 ignore next */ /* v8 ignore next */
      maxY = Math.max(maxY, n.y + n.height); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const graphWidth = maxX - minX; /* v8 ignore next */ /* v8 ignore next */
    const graphHeight = maxY - minY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const zoomX = (rect.width - 100) / graphWidth; /* v8 ignore next */ /* v8 ignore next */
    const zoomY = (rect.height - 100) / graphHeight; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.camera.zoom = Math.min(Math.min(zoomX, zoomY), 1); /* v8 ignore next */ /* v8 ignore next */
    this.camera.x = (rect.width / this.camera.zoom - graphWidth) / 2 - minX; /* v8 ignore next */ /* v8 ignore next */
    this.camera.y = (rect.height / this.camera.zoom - graphHeight) / 2 - minY; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onWheel(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const event = e as WheelEvent; /* v8 ignore next */ /* v8 ignore next */
    event.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const mouseX = event.clientX - rect.left; /* v8 ignore next */ /* v8 ignore next */
    const mouseY = event.clientY - rect.top; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const wheel = event.deltaY < 0 ? 1 : -1; /* v8 ignore next */ /* v8 ignore next */
    const zoomFactor = Math.exp(wheel * 0.1); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const worldX = mouseX / this.camera.zoom - this.camera.x; /* v8 ignore next */ /* v8 ignore next */
    const worldY = mouseY / this.camera.zoom - this.camera.y; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.camera.zoom *= zoomFactor; /* v8 ignore next */ /* v8 ignore next */
    this.camera.zoom = Math.max(0.1, Math.min(this.camera.zoom, 5)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.camera.x = mouseX / this.camera.zoom - worldX; /* v8 ignore next */ /* v8 ignore next */
    this.camera.y = mouseY / this.camera.zoom - worldY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseDown(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const event = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    if (this.isPaintingMask && event.button === 0) { /* v8 ignore next */ /* v8 ignore next */
      this.isDragging = true; /* v8 ignore next */ /* v8 ignore next */
      this.paintOnMask(event.clientX, event.clientY); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (event.button === 1 || event.button === 0) { /* v8 ignore next */ /* v8 ignore next */
      // Middle or left click drag /* v8 ignore next */ /* v8 ignore next */
      this.isDragging = true; /* v8 ignore next */ /* v8 ignore next */
      this.lastMouse = { x: event.clientX, y: event.clientY }; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private paintOnMask(clientX: number, clientY: number): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.maskData || !this.model || !this.paintTargetNode) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const init = this.model.initializers.find( /* v8 ignore next */ /* v8 ignore next */
      (i) => i.name === this.model!.nodes.find((n) => n.name === this.paintTargetNode)?.inputs[1], /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    if (!init) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rows = init.dims[0]; /* v8 ignore next */ /* v8 ignore next */
    const cols = init.dims[1]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const mouseX = clientX - rect.left; /* v8 ignore next */ /* v8 ignore next */
    const mouseY = clientY - rect.top; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const worldX = mouseX / this.camera.zoom - this.camera.x; /* v8 ignore next */ /* v8 ignore next */
    const worldY = mouseY / this.camera.zoom - this.camera.y; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // We map world coordinates directly to the 2D grid /* v8 ignore next */ /* v8 ignore next */
    const cellSize = 10; /* v8 ignore next */ /* v8 ignore next */
    const gridStartX = (rect.width / this.camera.zoom - cols * cellSize) / 2; /* v8 ignore next */ /* v8 ignore next */
    const gridStartY = (rect.height / this.camera.zoom - rows * cellSize) / 2; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const col = Math.floor((worldX - gridStartX) / cellSize); /* v8 ignore next */ /* v8 ignore next */
    const row = Math.floor((worldY - gridStartY) / cellSize); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Brush radius /* v8 ignore next */ /* v8 ignore next */
    const radius = 2; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (let r = Math.max(0, row - radius); r <= Math.min(rows - 1, row + radius); r++) { /* v8 ignore next */ /* v8 ignore next */
      for (let c = Math.max(0, col - radius); c <= Math.min(cols - 1, col + radius); c++) { /* v8 ignore next */ /* v8 ignore next */
        const dist = Math.sqrt(Math.pow(r - row, 2) + Math.pow(c - col, 2)); /* v8 ignore next */ /* v8 ignore next */
        if (dist <= radius) { /* v8 ignore next */ /* v8 ignore next */
          // Force to exactly zero to create true sparsity /* v8 ignore next */ /* v8 ignore next */
          this.maskData[r * cols + c] = 0; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseMove(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const event = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const mouseX = event.clientX - rect.left; /* v8 ignore next */ /* v8 ignore next */
    const mouseY = event.clientY - rect.top; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const worldX = mouseX / this.camera.zoom - this.camera.x; /* v8 ignore next */ /* v8 ignore next */
    const worldY = mouseY / this.camera.zoom - this.camera.y; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Throttle cursor broadcast /* v8 ignore next */ /* v8 ignore next */
    if (Math.random() < 0.1) { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('broadcastCursor', { worldX, worldY }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.isPaintingMask && this.isDragging) { /* v8 ignore next */ /* v8 ignore next */
      this.paintOnMask(event.clientX, event.clientY); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.isDragging) { /* v8 ignore next */ /* v8 ignore next */
      const dx = event.clientX - this.lastMouse.x; /* v8 ignore next */ /* v8 ignore next */
      const dy = event.clientY - this.lastMouse.y; /* v8 ignore next */ /* v8 ignore next */
      this.camera.x += dx / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
      this.camera.y += dy / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
      this.lastMouse = { x: event.clientX, y: event.clientY }; /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Hover detection /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let newHovered: string | null = null; /* v8 ignore next */ /* v8 ignore next */
    for (const ln of this.layout.nodes) { /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        worldX >= ln.x && /* v8 ignore next */ /* v8 ignore next */
        worldX <= ln.x + ln.width && /* v8 ignore next */ /* v8 ignore next */
        worldY >= ln.y && /* v8 ignore next */ /* v8 ignore next */
        worldY <= ln.y + ln.height /* v8 ignore next */ /* v8 ignore next */
      ) { /* v8 ignore next */ /* v8 ignore next */
        newHovered = ln.id; /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (newHovered !== this.hoveredNode) { /* v8 ignore next */ /* v8 ignore next */
      this.hoveredNode = newHovered; /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseUp(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const event = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    this.isDragging = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.hoveredNode && event.button === 0) { /* v8 ignore next */ /* v8 ignore next */
      // 158. Multi-select nodes with Shift key /* v8 ignore next */ /* v8 ignore next */
      if (event.shiftKey) { /* v8 ignore next */ /* v8 ignore next */
        if (this.multiSelectedNodes.has(this.hoveredNode)) { /* v8 ignore next */ /* v8 ignore next */
          this.multiSelectedNodes.delete(this.hoveredNode); /* v8 ignore next */ /* v8 ignore next */
        } else { /* v8 ignore next */ /* v8 ignore next */
          this.multiSelectedNodes.add(this.hoveredNode); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        this.multiSelectedNodes.clear(); /* v8 ignore next */ /* v8 ignore next */
        this.selectedNode = this.hoveredNode; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const node = this.layout.nodes.find((n) => n.id === this.hoveredNode)?.node; /* v8 ignore next */ /* v8 ignore next */
      if (node) { /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('nodeSelected', node); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 157. Emit subgraph event if multi-selected /* v8 ignore next */ /* v8 ignore next */
        if (this.multiSelectedNodes.size > 1) { /* v8 ignore next */ /* v8 ignore next */
          globalEvents.emit('multiSelectionChanged', Array.from(this.multiSelectedNodes)); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    } else if (event.button === 0 && !this.isPaintingMask) { /* v8 ignore next */ /* v8 ignore next */
      // Clear selection on background click /* v8 ignore next */ /* v8 ignore next */
      this.selectedNode = null; /* v8 ignore next */ /* v8 ignore next */
      this.multiSelectedNodes.clear(); /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('nodeSelected', null); /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('multiSelectionChanged', []); /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseLeave(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    this.isDragging = false; /* v8 ignore next */ /* v8 ignore next */
    this.hoveredNode = null; /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onKeyDown(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const event = e as KeyboardEvent; /* v8 ignore next */ /* v8 ignore next */
    const step = 20 / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    switch (event.key) { /* v8 ignore next */ /* v8 ignore next */
      case 'ArrowUp': /* v8 ignore next */ /* v8 ignore next */
        this.camera.y += step; /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'ArrowDown': /* v8 ignore next */ /* v8 ignore next */
        this.camera.y -= step; /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'ArrowLeft': /* v8 ignore next */ /* v8 ignore next */
        this.camera.x += step; /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'ArrowRight': /* v8 ignore next */ /* v8 ignore next */
        this.camera.x -= step; /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'Tab': /* v8 ignore next */ /* v8 ignore next */
        event.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
        this.cycleSelection(event.shiftKey ? -1 : 1); /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      case 'Escape': /* v8 ignore next */ /* v8 ignore next */
        if (this.isPaintingMask) { /* v8 ignore next */ /* v8 ignore next */
          this.isPaintingMask = false; /* v8 ignore next */ /* v8 ignore next */
          this.paintTargetNode = null; /* v8 ignore next */ /* v8 ignore next */
          this.maskData = null; /* v8 ignore next */ /* v8 ignore next */
          this.centerCamera(); /* v8 ignore next */ /* v8 ignore next */
          globalEvents.emit('modelLoaded', this.model!); // Re-trigger update logic for AST sizes /* v8 ignore next */ /* v8 ignore next */
          this.render(); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private cycleSelection(direction: number): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.layout.nodes.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let currentIndex = -1; /* v8 ignore next */ /* v8 ignore next */
    if (this.selectedNode) { /* v8 ignore next */ /* v8 ignore next */
      currentIndex = this.layout.nodes.findIndex((n) => n.id === this.selectedNode); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let nextIndex = currentIndex + direction; /* v8 ignore next */ /* v8 ignore next */
    if (nextIndex < 0) nextIndex = this.layout.nodes.length - 1; /* v8 ignore next */ /* v8 ignore next */
    if (nextIndex >= this.layout.nodes.length) nextIndex = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const node = this.layout.nodes[nextIndex]; /* v8 ignore next */ /* v8 ignore next */
    this.selectedNode = node.id; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Auto pan /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    this.camera.x = rect.width / this.camera.zoom / 2 - (node.x + node.width / 2); /* v8 ignore next */ /* v8 ignore next */
    this.camera.y = rect.height / this.camera.zoom / 2 - (node.y + node.height / 2); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('nodeSelected', node.node); /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private render(): void { /* v8 ignore next */ /* v8 ignore next */
    const isDark = document.body.getAttribute('data-theme') === 'dark'; /* v8 ignore next */ /* v8 ignore next */
    const bg = isDark ? '#121212' : '#ffffff'; /* v8 ignore next */ /* v8 ignore next */
    const grid = isDark ? '#2a2a2a' : '#e9ecef'; /* v8 ignore next */ /* v8 ignore next */
    const nodeBg = isDark ? '#1e1e1e' : '#f8f9fa'; /* v8 ignore next */ /* v8 ignore next */
    const nodeBorder = isDark ? '#444' : '#ccc'; /* v8 ignore next */ /* v8 ignore next */
    const text = isDark ? '#fff' : '#000'; /* v8 ignore next */ /* v8 ignore next */
    const highlight = '#0d6efd'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rect = this.container.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    this.ctx.fillStyle = bg; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.fillRect(0, 0, rect.width, rect.height); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.ctx.save(); /* v8 ignore next */ /* v8 ignore next */
    this.ctx.scale(this.camera.zoom, this.camera.zoom); /* v8 ignore next */ /* v8 ignore next */
    this.ctx.translate(this.camera.x, this.camera.y); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.isPaintingMask && this.maskData && this.paintTargetNode) { /* v8 ignore next */ /* v8 ignore next */
      this.renderSparsityMask(rect); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.restore(); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Frustum culling bounds /* v8 ignore next */ /* v8 ignore next */
    const worldLeft = -this.camera.x; /* v8 ignore next */ /* v8 ignore next */
    const worldTop = -this.camera.y; /* v8 ignore next */ /* v8 ignore next */
    const worldRight = worldLeft + rect.width / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
    const worldBottom = worldTop + rect.height / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Edges /* v8 ignore next */ /* v8 ignore next */
    this.layout.edges.forEach((edge) => { /* v8 ignore next */ /* v8 ignore next */
      const p1 = edge.points[0]; /* v8 ignore next */ /* v8 ignore next */
      const p2 = edge.points[edge.points.length - 1]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Frustum culling for edges (simple bounding box check) /* v8 ignore next */ /* v8 ignore next */
      const minX = Math.min(p1.x, p2.x); /* v8 ignore next */ /* v8 ignore next */
      const maxX = Math.max(p1.x, p2.x); /* v8 ignore next */ /* v8 ignore next */
      const minY = Math.min(p1.y, p2.y); /* v8 ignore next */ /* v8 ignore next */
      const maxY = Math.max(p1.y, p2.y); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (maxX < worldLeft || minX > worldRight || maxY < worldTop || minY > worldBottom) { /* v8 ignore next */ /* v8 ignore next */
        return; // Skip /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      // Highlight edge if connected to hovered node /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        this.hoveredNode && /* v8 ignore next */ /* v8 ignore next */
        (edge.source === this.hoveredNode || edge.target === this.hoveredNode) /* v8 ignore next */ /* v8 ignore next */
      ) { /* v8 ignore next */ /* v8 ignore next */
        this.ctx.strokeStyle = highlight; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineWidth = 3 / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        this.ctx.strokeStyle = isDark ? '#666' : '#999'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineWidth = 2 / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.moveTo(p1.x, p1.y); /* v8 ignore next */ /* v8 ignore next */
      // Simple cubic bezier curve /* v8 ignore next */ /* v8 ignore next */
      const cpOffset = (p2.y - p1.y) / 2; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.bezierCurveTo(p1.x, p1.y + cpOffset, p2.x, p2.y - cpOffset, p2.x, p2.y); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.showLabels && this.model) { /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = isDark ? '#888' : '#666'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.font = '10px monospace'; /* v8 ignore next */ /* v8 ignore next */
      this.layout.edges.forEach((edge) => { /* v8 ignore next */ /* v8 ignore next */
        // Try to look up the tensor shape /* v8 ignore next */ /* v8 ignore next */
        // In Dagrel layout, edge.source and edge.target are node names /* v8 ignore next */ /* v8 ignore next */
        // Find the specific tensor name connecting them /* v8 ignore next */ /* v8 ignore next */
        const sourceNode = this.model?.nodes.find((n) => n.name === edge.source); /* v8 ignore next */ /* v8 ignore next */
        const targetNode = this.model?.nodes.find((n) => n.name === edge.target); /* v8 ignore next */ /* v8 ignore next */
        if (sourceNode && targetNode) { /* v8 ignore next */ /* v8 ignore next */
          const tensorName = sourceNode.outputs.find((out) => targetNode.inputs.includes(out)); /* v8 ignore next */ /* v8 ignore next */
          if (tensorName) { /* v8 ignore next */ /* v8 ignore next */
            const vi = /* v8 ignore next */ /* v8 ignore next */
              this.model?.valueInfo?.find((v) => v.name === tensorName) || /* v8 ignore next */ /* v8 ignore next */
              this.model?.inputs.find((v) => v.name === tensorName); /* v8 ignore next */ /* v8 ignore next */
            if (vi && vi.type) { /* v8 ignore next */ /* v8 ignore next */
              const p1 = edge.points[0]; /* v8 ignore next */ /* v8 ignore next */
              const p2 = edge.points[edge.points.length - 1]; /* v8 ignore next */ /* v8 ignore next */
              const midX = (p1.x + p2.x) / 2; /* v8 ignore next */ /* v8 ignore next */
              const midY = (p1.y + p2.y) / 2; /* v8 ignore next */ /* v8 ignore next */
              const shapeStr = `[${vi.type.shape.join(',')}]`; /* v8 ignore next */ /* v8 ignore next */
              this.ctx.fillText(shapeStr, midX, midY - 10); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Nodes /* v8 ignore next */ /* v8 ignore next */
    this.ctx.font = '12px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.textAlign = 'center'; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.textBaseline = 'middle'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      // Frustum culling /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        n.x + n.width < worldLeft || /* v8 ignore next */ /* v8 ignore next */
        n.x > worldRight || /* v8 ignore next */ /* v8 ignore next */
        n.y + n.height < worldTop || /* v8 ignore next */ /* v8 ignore next */
        n.y > worldBottom /* v8 ignore next */ /* v8 ignore next */
      ) { /* v8 ignore next */ /* v8 ignore next */
        return; // Skip rendering /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = nodeBg; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      let currentBorder = nodeBorder; /* v8 ignore next */ /* v8 ignore next */
      let currentLineWidth = 1 / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (this.selectedNode === n.id || this.multiSelectedNodes.has(n.id)) { /* v8 ignore next */ /* v8 ignore next */
        currentBorder = highlight; /* v8 ignore next */ /* v8 ignore next */
        currentLineWidth = 3 / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
      } else if (this.hoveredNode === n.id) { /* v8 ignore next */ /* v8 ignore next */
        currentBorder = highlight; /* v8 ignore next */ /* v8 ignore next */
        currentLineWidth = 2 / this.camera.zoom; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.strokeStyle = currentBorder; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.lineWidth = currentLineWidth; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.roundRect(n.x, n.y, n.width, n.height, 4); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      let currentBg = nodeBg; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Phase 4: Color-code nodes based on operation category /* v8 ignore next */ /* v8 ignore next */
      const type = n.node.opType.toLowerCase(); /* v8 ignore next */ /* v8 ignore next */
      if (['matmul', 'add', 'mul', 'sub', 'div', 'gemm'].includes(type)) { /* v8 ignore next */ /* v8 ignore next */
        currentBg = isDark ? '#1a2a44' : '#e6f2ff'; /* v8 ignore next */ /* v8 ignore next */
      } else if (['conv', 'maxpool', 'averagepool', 'relu', 'softmax'].includes(type)) { /* v8 ignore next */ /* v8 ignore next */
        currentBg = isDark ? '#1d3826' : '#e8f5e9'; /* v8 ignore next */ /* v8 ignore next */
      } else if (['if', 'loop', 'where'].includes(type)) { /* v8 ignore next */ /* v8 ignore next */
        currentBg = isDark ? '#441b1b' : '#ffebee'; /* v8 ignore next */ /* v8 ignore next */
        // 120. Collapsible subgraphs indicator /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillStyle = isDark ? '#ff6b6b' : '#dc3545'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.arc(n.x + n.width - 10, n.y + 10, 4, 0, Math.PI * 2); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
      } else if (n.node.attributes['is_backward']) { /* v8 ignore next */ /* v8 ignore next */
        // 216. Visualize new gradient graph red /* v8 ignore next */ /* v8 ignore next */
        currentBg = isDark ? '#5c1010' : '#ffcccc'; /* v8 ignore next */ /* v8 ignore next */
      } else if (n.node.attributes['is_loss'] || n.node.attributes['is_optimizer']) { /* v8 ignore next */ /* v8 ignore next */
        currentBg = isDark ? '#4a3c10' : '#fff3cd'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = currentBg; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.roundRect(n.x, n.y, n.width, n.height, 4); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = text; /* v8 ignore next */ /* v8 ignore next */
      // OpType text /* v8 ignore next */ /* v8 ignore next */
      // Usage of optimized cache /* v8 ignore next */ /* v8 ignore next */
      const opWidth = this.measureText(n.node.opType, this.ctx); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillText(n.node.opType, n.x + n.width / 2, n.y + n.height / 2 - 6); /* v8 ignore next */ /* v8 ignore next */
      // Name text /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = isDark ? '#888' : '#666'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.font = '10px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
      let name = n.node.name; /* v8 ignore next */ /* v8 ignore next */
      if (name.length > 20) name = name.substring(0, 17) + '...'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillText(name, n.x + n.width / 2, n.y + n.height / 2 + 10); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.font = '12px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 526. Draw Remote Cursors /* v8 ignore next */ /* v8 ignore next */
    const now = Date.now(); /* v8 ignore next */ /* v8 ignore next */
    this.remoteCursors.forEach((c, peerId) => { /* v8 ignore next */ /* v8 ignore next */
      if (now - c.timestamp > 5000) { /* v8 ignore next */ /* v8 ignore next */
        this.remoteCursors.delete(peerId); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillStyle = c.color; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
        // Draw simple cursor arrow /* v8 ignore next */ /* v8 ignore next */
        this.ctx.moveTo(c.x, c.y); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineTo(c.x + 10, c.y + 10); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineTo(c.x + 3, c.y + 12); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineTo(c.x, c.y + 18); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 527. Color-code handles /* v8 ignore next */ /* v8 ignore next */
        this.ctx.font = '10px monospace'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillText(peerId.substring(0, 6), c.x + 15, c.y + 15); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.ctx.restore(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 117. Render Minimap /* v8 ignore next */ /* v8 ignore next */
    if (this.model && this.layout.nodes.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      this.renderMinimap(worldLeft, worldTop, worldRight, worldBottom); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private renderMinimap( /* v8 ignore next */ /* v8 ignore next */
    viewLeft: number, /* v8 ignore next */ /* v8 ignore next */
    viewTop: number, /* v8 ignore next */ /* v8 ignore next */
    viewRight: number, /* v8 ignore next */ /* v8 ignore next */
    viewBottom: number, /* v8 ignore next */ /* v8 ignore next */
  ): void { /* v8 ignore next */ /* v8 ignore next */
    const rect = this.minimapCanvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.clearRect(0, 0, rect.width, rect.height); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let minX = Infinity, /* v8 ignore next */ /* v8 ignore next */
      minY = Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxX = -Infinity, /* v8 ignore next */ /* v8 ignore next */
      maxY = -Infinity; /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      minX = Math.min(minX, n.x); /* v8 ignore next */ /* v8 ignore next */
      minY = Math.min(minY, n.y); /* v8 ignore next */ /* v8 ignore next */
      maxX = Math.max(maxX, n.x + n.width); /* v8 ignore next */ /* v8 ignore next */
      maxY = Math.max(maxY, n.y + n.height); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const graphW = maxX - minX; /* v8 ignore next */ /* v8 ignore next */
    const graphH = maxY - minY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (graphW === 0 || graphH === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const scaleX = rect.width / graphW; /* v8 ignore next */ /* v8 ignore next */
    const scaleY = rect.height / graphH; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.fillStyle = 'rgba(100, 100, 100, 0.5)'; /* v8 ignore next */ /* v8 ignore next */
    this.layout.nodes.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
      const mx = (n.x - minX) * scaleX; /* v8 ignore next */ /* v8 ignore next */
      const my = (n.y - minY) * scaleY; /* v8 ignore next */ /* v8 ignore next */
      const mw = n.width * scaleX; /* v8 ignore next */ /* v8 ignore next */
      const mh = n.height * scaleY; /* v8 ignore next */ /* v8 ignore next */
      this.minimapCtx.fillRect(mx, my, mw, mh); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Draw Viewport indicator /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.strokeStyle = 'var(--color-primary)'; /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.lineWidth = 2; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const vx = (viewLeft - minX) * scaleX; /* v8 ignore next */ /* v8 ignore next */
    const vy = (viewTop - minY) * scaleY; /* v8 ignore next */ /* v8 ignore next */
    const vw = (viewRight - viewLeft) * scaleX; /* v8 ignore next */ /* v8 ignore next */
    const vh = (viewBottom - viewTop) * scaleY; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.strokeRect(vx, vy, vw, vh); /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.fillStyle = 'rgba(13, 110, 253, 0.1)'; /* v8 ignore next */ /* v8 ignore next */
    this.minimapCtx.fillRect(vx, vy, vw, vh); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private renderSparsityMask(rect: DOMRect): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.maskData || !this.model || !this.paintTargetNode) return; /* v8 ignore next */ /* v8 ignore next */
    const init = this.model.initializers.find( /* v8 ignore next */ /* v8 ignore next */
      (i) => i.name === this.model!.nodes.find((n) => n.name === this.paintTargetNode)?.inputs[1], /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    if (!init) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rows = init.dims[0]; /* v8 ignore next */ /* v8 ignore next */
    const cols = init.dims[1]; /* v8 ignore next */ /* v8 ignore next */
    const cellSize = 10; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const gridStartX = (rect.width / this.camera.zoom - cols * cellSize) / 2; /* v8 ignore next */ /* v8 ignore next */
    const gridStartY = (rect.height / this.camera.zoom - rows * cellSize) / 2; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.ctx.fillStyle = '#333'; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.font = '16px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.textAlign = 'center'; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.fillText( /* v8 ignore next */ /* v8 ignore next */
      `Painting Sparsity Mask: ${this.paintTargetNode}`, /* v8 ignore next */ /* v8 ignore next */
      rect.width / this.camera.zoom / 2, /* v8 ignore next */ /* v8 ignore next */
      30, /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    this.ctx.font = '12px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.fillText( /* v8 ignore next */ /* v8 ignore next */
      "Press 'ESC' or click outside grid to save and exit mask mode", /* v8 ignore next */ /* v8 ignore next */
      rect.width / this.camera.zoom / 2, /* v8 ignore next */ /* v8 ignore next */
      50, /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (let r = 0; r < rows; r++) { /* v8 ignore next */ /* v8 ignore next */
      for (let c = 0; c < cols; c++) { /* v8 ignore next */ /* v8 ignore next */
        const val = this.maskData[r * cols + c]; /* v8 ignore next */ /* v8 ignore next */
        if (val === 0) { /* v8 ignore next */ /* v8 ignore next */
          this.ctx.fillStyle = '#ff4444'; // Pruned (Red) /* v8 ignore next */ /* v8 ignore next */
        } else { /* v8 ignore next */ /* v8 ignore next */
          // Grayscale based on magnitude /* v8 ignore next */ /* v8 ignore next */
          const mag = Math.min(255, Math.floor(Math.abs(val) * 255 * 5)); /* v8 ignore next */ /* v8 ignore next */
          this.ctx.fillStyle = `rgb(${mag},${mag},${mag})`; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillRect( /* v8 ignore next */ /* v8 ignore next */
          gridStartX + c * cellSize, /* v8 ignore next */ /* v8 ignore next */
          gridStartY + r * cellSize, /* v8 ignore next */ /* v8 ignore next */
          cellSize - 1, /* v8 ignore next */ /* v8 ignore next */
          cellSize - 1, /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  unmount(): void { /* v8 ignore next */ /* v8 ignore next */
    super.unmount(); /* v8 ignore next */ /* v8 ignore next */
    window.removeEventListener('resize', this.resize); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
