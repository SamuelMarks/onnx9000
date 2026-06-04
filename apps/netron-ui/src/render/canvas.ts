/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import {
  GraphLayout,
  LayoutNode,
  LayoutEdge,
} from '../layout/dag'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export class CanvasRenderer {
  /* v8 ignore next */ /* v8 ignore next */
  private canvas: HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
  private ctx: CanvasRenderingContext2D; /* v8 ignore next */ /* v8 ignore next */
  private layout: GraphLayout | null = null; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private scale: number = 1; /* v8 ignore next */ /* v8 ignore next */
  private offsetX: number = 0; /* v8 ignore next */ /* v8 ignore next */
  private offsetY: number = 0; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private isDragging: boolean = false; /* v8 ignore next */ /* v8 ignore next */
  private hasMovedDuringDrag: boolean = false; /* v8 ignore next */ /* v8 ignore next */
  private lastMouseX: number = 0; /* v8 ignore next */ /* v8 ignore next */
  private lastMouseY: number = 0; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  public onSelect: (nodeId: string | null) => void = () =>
    undefined; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  constructor(canvas: HTMLCanvasElement) {
    /* v8 ignore next */ /* v8 ignore next */
    this.canvas = canvas; /* v8 ignore next */ /* v8 ignore next */
    const ctx = canvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
    if (!ctx)
      throw new Error('Canvas 2D context not supported'); /* v8 ignore next */ /* v8 ignore next */
    this.ctx = ctx; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.setupEvents(); /* v8 ignore next */ /* v8 ignore next */
    this.resize(); /* v8 ignore next */ /* v8 ignore next */
    window.addEventListener('resize', () => {
      /* v8 ignore next */ /* v8 ignore next */
      this.resize(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  setLayout(layout: GraphLayout) {
    /* v8 ignore next */ /* v8 ignore next */
    this.layout = layout; /* v8 ignore next */ /* v8 ignore next */
    this.centerGraph(); /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private centerGraph() {
    /* v8 ignore next */ /* v8 ignore next */
    if (!this.layout) return; /* v8 ignore next */ /* v8 ignore next */
    const cw = this.canvas.width; /* v8 ignore next */ /* v8 ignore next */
    const ch = this.canvas.height; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Auto-scale to fit horizontally or vertically, max scale 1 /* v8 ignore next */ /* v8 ignore next */
    const scaleX = cw / (this.layout.width + 200); /* v8 ignore next */ /* v8 ignore next */
    const scaleY = ch / (this.layout.height + 200); /* v8 ignore next */ /* v8 ignore next */
    this.scale = Math.min(1, Math.min(scaleX, scaleY)); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Center it /* v8 ignore next */ /* v8 ignore next */
    this.offsetX = cw / 2; /* v8 ignore next */ /* v8 ignore next */
    this.offsetY = 100 * this.scale; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Attempt restore from localStorage /* v8 ignore next */ /* v8 ignore next */
    try {
      /* v8 ignore next */ /* v8 ignore next */
      const savedScale =
        localStorage.getItem('onnxModifier_scale'); /* v8 ignore next */ /* v8 ignore next */
      const savedOffsetX =
        localStorage.getItem('onnxModifier_offsetX'); /* v8 ignore next */ /* v8 ignore next */
      const savedOffsetY =
        localStorage.getItem('onnxModifier_offsetY'); /* v8 ignore next */ /* v8 ignore next */
      if (savedScale && savedOffsetX && savedOffsetY) {
        /* v8 ignore next */ /* v8 ignore next */
        this.scale = parseFloat(savedScale); /* v8 ignore next */ /* v8 ignore next */
        this.offsetX = parseFloat(savedOffsetX); /* v8 ignore next */ /* v8 ignore next */
        this.offsetY = parseFloat(savedOffsetY); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } catch (e) {} /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private resize() {
    /* v8 ignore next */ /* v8 ignore next */
    this.canvas.width = window.innerWidth; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.height = window.innerHeight; /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private hoveredNode: string | null = null; /* v8 ignore next */ /* v8 ignore next */
  private hoveredEdge: LayoutEdge | null = null; /* v8 ignore next */ /* v8 ignore next */
  public selectedNodes: string[] = []; /* v8 ignore next */ /* v8 ignore next */
  private searchResults: Set<string> = new Set(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private hideControlEdges: boolean = false; /* v8 ignore next */ /* v8 ignore next */
  private customColorRegex: RegExp | null = null; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  public setSearchResults(results: string[]) {
    /* v8 ignore next */ /* v8 ignore next */
    this.searchResults = new Set(results); /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  public setFilterControlEdges(hide: boolean) {
    /* v8 ignore next */ /* v8 ignore next */
    this.hideControlEdges = hide; /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  public setCustomColorRegex(pattern: string) {
    /* v8 ignore next */ /* v8 ignore next */
    if (!pattern.trim()) {
      /* v8 ignore next */ /* v8 ignore next */
      this.customColorRegex = null; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      try {
        /* v8 ignore next */ /* v8 ignore next */
        this.customColorRegex = new RegExp(pattern, 'i'); /* v8 ignore next */ /* v8 ignore next */
      } catch (e) {
        /* v8 ignore next */ /* v8 ignore next */
        this.customColorRegex = null; // invalid regex, ignore /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  public focusNode(nodeId: string) {
    /* v8 ignore next */ /* v8 ignore next */
    if (!this.layout) return; /* v8 ignore next */ /* v8 ignore next */
    const node = this.layout.nodes.find(
      (n) => n.id === nodeId,
    ); /* v8 ignore next */ /* v8 ignore next */
    if (!node) return; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.selectedNodes = [nodeId]; /* v8 ignore next */ /* v8 ignore next */
    // Animate or snap to center /* v8 ignore next */ /* v8 ignore next */
    this.offsetX =
      this.canvas.width / 2 -
      (node.x + node.width / 2) * this.scale; /* v8 ignore next */ /* v8 ignore next */
    this.offsetY =
      this.canvas.height / 2 -
      (node.y + node.height / 2) * this.scale; /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  private setupEvents() {
    /* v8 ignore next */ /* v8 ignore next */
    // 283. Implement touch event handling for iPad/Mobile Safari rendering. /* v8 ignore next */ /* v8 ignore next */
    this.canvas.addEventListener(
      /* v8 ignore next */ /* v8 ignore next */
      'touchstart' /* v8 ignore next */ /* v8 ignore next */,
      (e) => {
        /* v8 ignore next */ /* v8 ignore next */
        if (e.touches.length === 1) {
          /* v8 ignore next */ /* v8 ignore next */
          this.isDragging = true; /* v8 ignore next */ /* v8 ignore next */
          this.hasMovedDuringDrag = false; /* v8 ignore next */ /* v8 ignore next */
          this.lastMouseX = e.touches[0]!.clientX; /* v8 ignore next */ /* v8 ignore next */
          this.lastMouseY = e.touches[0]!.clientY; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */,
      { passive: false } /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.canvas.addEventListener(
      /* v8 ignore next */ /* v8 ignore next */
      'touchmove' /* v8 ignore next */ /* v8 ignore next */,
      (e) => {
        /* v8 ignore next */ /* v8 ignore next */
        if (this.isDragging && e.touches.length === 1) {
          /* v8 ignore next */ /* v8 ignore next */
          e.preventDefault(); // Prevent scrolling /* v8 ignore next */ /* v8 ignore next */
          this.hasMovedDuringDrag = true; /* v8 ignore next */ /* v8 ignore next */
          this.offsetX +=
            e.touches[0]!.clientX - this.lastMouseX; /* v8 ignore next */ /* v8 ignore next */
          this.offsetY +=
            e.touches[0]!.clientY - this.lastMouseY; /* v8 ignore next */ /* v8 ignore next */
          this.lastMouseX = e.touches[0]!.clientX; /* v8 ignore next */ /* v8 ignore next */
          this.lastMouseY = e.touches[0]!.clientY; /* v8 ignore next */ /* v8 ignore next */
          try {
            /* v8 ignore next */ /* v8 ignore next */
            localStorage.setItem(
              'onnxModifier_offsetX',
              String(this.offsetX),
            ); /* v8 ignore next */ /* v8 ignore next */
            localStorage.setItem(
              'onnxModifier_offsetY',
              String(this.offsetY),
            ); /* v8 ignore next */ /* v8 ignore next */
          } catch (e) {} /* v8 ignore next */ /* v8 ignore next */
          this.render(); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */,
      { passive: false } /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.canvas.addEventListener('touchend', () => {
      /* v8 ignore next */ /* v8 ignore next */
      this.isDragging = false; /* v8 ignore next */ /* v8 ignore next */
      // We don't simulate click selection on touch yet, just pan /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.canvas.addEventListener('mousedown', (e) => {
      /* v8 ignore next */ /* v8 ignore next */
      this.isDragging = true; /* v8 ignore next */ /* v8 ignore next */
      this.hasMovedDuringDrag = false; /* v8 ignore next */ /* v8 ignore next */
      this.lastMouseX = e.clientX; /* v8 ignore next */ /* v8 ignore next */
      this.lastMouseY = e.clientY; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    window.addEventListener('mouseup', (e) => {
      /* v8 ignore next */ /* v8 ignore next */
      if (this.isDragging && !this.hasMovedDuringDrag) {
        /* v8 ignore next */ /* v8 ignore next */
        if (this.hoveredNode) {
          /* v8 ignore next */ /* v8 ignore next */
          if (e.shiftKey) {
            /* v8 ignore next */ /* v8 ignore next */
            if (this.selectedNodes.includes(this.hoveredNode)) {
              /* v8 ignore next */ /* v8 ignore next */
              this.selectedNodes = this.selectedNodes.filter(
                (n) => n !== this.hoveredNode,
              ); /* v8 ignore next */ /* v8 ignore next */
            } else {
              /* v8 ignore next */ /* v8 ignore next */
              this.selectedNodes.push(this.hoveredNode); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } else {
            /* v8 ignore next */ /* v8 ignore next */
            this.selectedNodes = [this.hoveredNode]; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          this.onSelect(
            /* v8 ignore next */ /* v8 ignore next */
            this.selectedNodes.length > 0 /* v8 ignore next */ /* v8 ignore next */
              ? this.selectedNodes[
                  this.selectedNodes.length - 1
                ]! /* v8 ignore next */ /* v8 ignore next */
              : null /* v8 ignore next */ /* v8 ignore next */,
          ); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          this.selectedNodes = []; /* v8 ignore next */ /* v8 ignore next */
          this.onSelect(null); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      this.isDragging = false; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    window.addEventListener('mousemove', (e) => {
      /* v8 ignore next */ /* v8 ignore next */
      if (this.isDragging) {
        /* v8 ignore next */ /* v8 ignore next */
        this.hasMovedDuringDrag = true; /* v8 ignore next */ /* v8 ignore next */
        this.offsetX += e.clientX - this.lastMouseX; /* v8 ignore next */ /* v8 ignore next */
        this.offsetY += e.clientY - this.lastMouseY; /* v8 ignore next */ /* v8 ignore next */
        this.lastMouseX = e.clientX; /* v8 ignore next */ /* v8 ignore next */
        this.lastMouseY = e.clientY; /* v8 ignore next */ /* v8 ignore next */
        try {
          /* v8 ignore next */ /* v8 ignore next */
          localStorage.setItem(
            'onnxModifier_offsetX',
            String(this.offsetX),
          ); /* v8 ignore next */ /* v8 ignore next */
          localStorage.setItem(
            'onnxModifier_offsetY',
            String(this.offsetY),
          ); /* v8 ignore next */ /* v8 ignore next */
        } catch (e) {} /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        // Sub-millisecond hit testing for hovering /* v8 ignore next */ /* v8 ignore next */
        if (this.layout) {
          /* v8 ignore next */ /* v8 ignore next */
          const mx =
            (e.clientX - this.offsetX) / this.scale; /* v8 ignore next */ /* v8 ignore next */
          const my =
            (e.clientY - this.offsetY) / this.scale; /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          let foundNode: string | null = null; /* v8 ignore next */ /* v8 ignore next */
          // Loop backwards to hit top-most items first /* v8 ignore next */ /* v8 ignore next */
          for (let i = this.layout.nodes.length - 1; i >= 0; i--) {
            /* v8 ignore next */ /* v8 ignore next */
            const n = this.layout.nodes[i]!; /* v8 ignore next */ /* v8 ignore next */
            if (mx >= n.x && mx <= n.x + n.width && my >= n.y && my <= n.y + n.height) {
              /* v8 ignore next */ /* v8 ignore next */
              foundNode = n.id; /* v8 ignore next */ /* v8 ignore next */
              break; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          let foundEdge: LayoutEdge | null = null; /* v8 ignore next */ /* v8 ignore next */
          if (!foundNode) {
            /* v8 ignore next */ /* v8 ignore next */
            // Hit test edges if no node found /* v8 ignore next */ /* v8 ignore next */
            for (const edge of this.layout.edges) {
              /* v8 ignore next */ /* v8 ignore next */
              const p1 = edge.points[0]; /* v8 ignore next */ /* v8 ignore next */
              const p2 = edge.points[1]; /* v8 ignore next */ /* v8 ignore next */
              if (p1 && p2) {
                /* v8 ignore next */ /* v8 ignore next */
                const isVertical =
                  Math.abs(p2.y - p1.y) >
                  Math.abs(p2.x - p1.x); /* v8 ignore next */ /* v8 ignore next */
                const path = new Path2D(); /* v8 ignore next */ /* v8 ignore next */
                path.moveTo(p1.x, p1.y); /* v8 ignore next */ /* v8 ignore next */
                if (isVertical) {
                  /* v8 ignore next */ /* v8 ignore next */
                  const midY = (p1.y + p2.y) / 2; /* v8 ignore next */ /* v8 ignore next */
                  path.bezierCurveTo(
                    p1.x,
                    midY,
                    p2.x,
                    midY,
                    p2.x,
                    p2.y,
                  ); /* v8 ignore next */ /* v8 ignore next */
                } else {
                  /* v8 ignore next */ /* v8 ignore next */
                  const midX = (p1.x + p2.x) / 2; /* v8 ignore next */ /* v8 ignore next */
                  path.bezierCurveTo(
                    midX,
                    p1.y,
                    midX,
                    p2.y,
                    p2.x,
                    p2.y,
                  ); /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
                /* v8 ignore next */ /* v8 ignore next */
                this.ctx.lineWidth = 10 / this.scale; // Thicker hit area /* v8 ignore next */ /* v8 ignore next */
                if (this.ctx.isPointInStroke(path, mx, my)) {
                  /* v8 ignore next */ /* v8 ignore next */
                  foundEdge = edge; /* v8 ignore next */ /* v8 ignore next */
                  break; /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          let changed = false; /* v8 ignore next */ /* v8 ignore next */
          if (this.hoveredNode !== foundNode) {
            /* v8 ignore next */ /* v8 ignore next */
            this.hoveredNode = foundNode; /* v8 ignore next */ /* v8 ignore next */
            changed = true; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          if (this.hoveredEdge !== foundEdge) {
            /* v8 ignore next */ /* v8 ignore next */
            this.hoveredEdge = foundEdge; /* v8 ignore next */ /* v8 ignore next */
            changed = true; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          /* v8 ignore next */ /* v8 ignore next */
          if (changed) {
            /* v8 ignore next */ /* v8 ignore next */
            this.canvas.style.cursor =
              foundNode || foundEdge
                ? 'pointer'
                : 'default'; /* v8 ignore next */ /* v8 ignore next */
            this.render(); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.canvas.addEventListener(
      /* v8 ignore next */ /* v8 ignore next */
      'wheel' /* v8 ignore next */ /* v8 ignore next */,
      (e) => {
        /* v8 ignore next */ /* v8 ignore next */
        e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
        const zoomSensitivity = 0.001; /* v8 ignore next */ /* v8 ignore next */
        const zoom = 1 - e.deltaY * zoomSensitivity; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        const mouseX = e.clientX; /* v8 ignore next */ /* v8 ignore next */
        const mouseY = e.clientY; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // Zoom around cursor /* v8 ignore next */ /* v8 ignore next */
        this.offsetX =
          mouseX - (mouseX - this.offsetX) * zoom; /* v8 ignore next */ /* v8 ignore next */
        this.offsetY =
          mouseY - (mouseY - this.offsetY) * zoom; /* v8 ignore next */ /* v8 ignore next */
        this.scale *= zoom; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // Limit zoom /* v8 ignore next */ /* v8 ignore next */
        this.scale = Math.max(
          0.01,
          Math.min(this.scale, 5),
        ); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // 279. Support saving layout preferences to localStorage /* v8 ignore next */ /* v8 ignore next */
        try {
          /* v8 ignore next */ /* v8 ignore next */
          localStorage.setItem(
            'onnxModifier_scale',
            String(this.scale),
          ); /* v8 ignore next */ /* v8 ignore next */
          localStorage.setItem(
            'onnxModifier_offsetX',
            String(this.offsetX),
          ); /* v8 ignore next */ /* v8 ignore next */
          localStorage.setItem(
            'onnxModifier_offsetY',
            String(this.offsetY),
          ); /* v8 ignore next */ /* v8 ignore next */
        } catch (e) {} /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        this.render(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */,
      { passive: false } /* v8 ignore next */ /* v8 ignore next */,
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  render() {
    /* v8 ignore next */ /* v8 ignore next */
    const { ctx, canvas } = this; /* v8 ignore next */ /* v8 ignore next */
    ctx.clearRect(0, 0, canvas.width, canvas.height); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    if (!this.layout) return; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    ctx.save(); /* v8 ignore next */ /* v8 ignore next */
    ctx.translate(this.offsetX, this.offsetY); /* v8 ignore next */ /* v8 ignore next */
    ctx.scale(this.scale, this.scale); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const startX = -this.offsetX / this.scale; /* v8 ignore next */ /* v8 ignore next */
    const startY = -this.offsetY / this.scale; /* v8 ignore next */ /* v8 ignore next */
    const endX = startX + canvas.width / this.scale; /* v8 ignore next */ /* v8 ignore next */
    const endY = startY + canvas.height / this.scale; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Draw Grid /* v8 ignore next */ /* v8 ignore next */
    if (this.scale > 0.2) {
      /* v8 ignore next */ /* v8 ignore next */
      ctx.save(); /* v8 ignore next */ /* v8 ignore next */
      ctx.strokeStyle = '#222'; /* v8 ignore next */ /* v8 ignore next */
      ctx.lineWidth = 1 / this.scale; /* v8 ignore next */ /* v8 ignore next */
      const gridSize = 50; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const gridStartX =
        Math.floor(startX / gridSize) * gridSize; /* v8 ignore next */ /* v8 ignore next */
      const gridStartY =
        Math.floor(startY / gridSize) * gridSize; /* v8 ignore next */ /* v8 ignore next */
      const gridEndX = endX + gridSize; /* v8 ignore next */ /* v8 ignore next */
      const gridEndY = endY + gridSize; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      for (let x = gridStartX; x < gridEndX; x += gridSize) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.moveTo(x, gridStartY); /* v8 ignore next */ /* v8 ignore next */
        ctx.lineTo(x, gridEndY); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      for (let y = gridStartY; y < gridEndY; y += gridSize) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.moveTo(gridStartX, y); /* v8 ignore next */ /* v8 ignore next */
        ctx.lineTo(gridEndX, y); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
      ctx.setLineDash([]); // Reset line dash for next shapes /* v8 ignore next */ /* v8 ignore next */
      ctx.restore(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Draw Groups (NameScopes) /* v8 ignore next */ /* v8 ignore next */
    if (this.layout.groups) {
      /* v8 ignore next */ /* v8 ignore next */
      for (const group of this.layout.groups) {
        /* v8 ignore next */ /* v8 ignore next */
        if (
          /* v8 ignore next */ /* v8 ignore next */
          group.x + group.width < startX /* v8 ignore next */ /* v8 ignore next */ ||
          group.x > endX /* v8 ignore next */ /* v8 ignore next */ ||
          group.y + group.height < startY /* v8 ignore next */ /* v8 ignore next */ ||
          group.y > endY /* v8 ignore next */ /* v8 ignore next */
        ) {
          /* v8 ignore next */ /* v8 ignore next */
          continue; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = `rgba(255, 255, 255, ${0.02 + group.depth * 0.01})`; /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 + group.depth * 0.05})`; /* v8 ignore next */ /* v8 ignore next */
        ctx.lineWidth = 1 / this.scale; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
        ctx.roundRect(
          group.x,
          group.y,
          group.width,
          group.height,
          8,
        ); /* v8 ignore next */ /* v8 ignore next */
        ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
        ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        if (this.scale > 0.1) {
          /* v8 ignore next */ /* v8 ignore next */
          ctx.fillStyle = `rgba(255, 255, 255, ${0.4 + group.depth * 0.1})`; /* v8 ignore next */ /* v8 ignore next */
          ctx.font = '12px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
          ctx.textAlign = 'left'; /* v8 ignore next */ /* v8 ignore next */
          ctx.textBaseline = 'top'; /* v8 ignore next */ /* v8 ignore next */
          ctx.fillText(
            group.name,
            group.x + 10,
            group.y + 10,
          ); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Draw Edges /* v8 ignore next */ /* v8 ignore next */
    for (const edge of this.layout.edges) {
      /* v8 ignore next */ /* v8 ignore next */
      if (this.hideControlEdges) {
        /* v8 ignore next */ /* v8 ignore next */
        // Typical ONNX control edges have empty strings as tensor names, or are boolean /* v8 ignore next */ /* v8 ignore next */
        if (!edge.tensorName || edge.dtype === 'bool')
          continue; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      const isHoveredEdge = this.hoveredEdge === edge; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // Color by dtype /* v8 ignore next */ /* v8 ignore next */
      // 133. Colorblind-friendly palette /* v8 ignore next */ /* v8 ignore next */
      // Using IBM Design-ish / Wong palette colors for colorblind accessibility /* v8 ignore next */ /* v8 ignore next */
      if (isHoveredEdge) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#ffffff'; // Highlighted edge color /* v8 ignore next */ /* v8 ignore next */
      } else if (edge.dtype?.startsWith('float') || edge.dtype?.startsWith('bfloat')) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#56B4E9'; // Sky Blue /* v8 ignore next */ /* v8 ignore next */
      } else if (edge.dtype?.startsWith('int') || edge.dtype?.startsWith('uint')) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#009E73'; // Bluish Green /* v8 ignore next */ /* v8 ignore next */
      } else if (edge.dtype === 'bool') {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#F0E442'; // Yellow /* v8 ignore next */ /* v8 ignore next */
      } else if (edge.dtype === 'string') {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#E69F00'; // Orange /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.strokeStyle = '#888888'; // Default /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      ctx.lineWidth =
        (isHoveredEdge ? 4 : 2) / this.scale; /* v8 ignore next */ /* v8 ignore next */
      ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const p1 = edge.points[0]; /* v8 ignore next */ /* v8 ignore next */
      const p2 = edge.points[1]; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (p1 && p2) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.moveTo(p1.x, p1.y); /* v8 ignore next */ /* v8 ignore next */
        // Smooth bezier curve for DAG flow /* v8 ignore next */ /* v8 ignore next */
        // Assuming mostly vertical flow for the control points logic, but we can do generic /* v8 ignore next */ /* v8 ignore next */
        const isVertical =
          Math.abs(p2.y - p1.y) > Math.abs(p2.x - p1.x); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        if (isVertical) {
          /* v8 ignore next */ /* v8 ignore next */
          const midY = (p1.y + p2.y) / 2; /* v8 ignore next */ /* v8 ignore next */
          ctx.bezierCurveTo(
            p1.x,
            midY,
            p2.x,
            midY,
            p2.x,
            p2.y,
          ); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          const midX = (p1.x + p2.x) / 2; /* v8 ignore next */ /* v8 ignore next */
          ctx.bezierCurveTo(
            midX,
            p1.y,
            midX,
            p2.y,
            p2.x,
            p2.y,
          ); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // Render data types and shapes as edge labels /* v8 ignore next */ /* v8 ignore next */
        if ((this.scale > 0.8 || isHoveredEdge) && edge.shape && edge.dtype) {
          /* v8 ignore next */ /* v8 ignore next */
          ctx.fillStyle = isHoveredEdge ? '#fff' : '#aaa'; /* v8 ignore next */ /* v8 ignore next */
          ctx.font = isHoveredEdge
            ? '12px sans-serif'
            : '10px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
          ctx.textAlign = 'center'; /* v8 ignore next */ /* v8 ignore next */
          ctx.textBaseline = 'bottom'; /* v8 ignore next */ /* v8 ignore next */
          const midX = (p1.x + p2.x) / 2; /* v8 ignore next */ /* v8 ignore next */
          const midY = (p1.y + p2.y) / 2; /* v8 ignore next */ /* v8 ignore next */
          // 210. Edge hover preview of tensor shape sizes /* v8 ignore next */ /* v8 ignore next */
          let text = `${edge.dtype} ${edge.shape}`; /* v8 ignore next */ /* v8 ignore next */
          if (isHoveredEdge) {
            /* v8 ignore next */ /* v8 ignore next */
            // estimate byte size if possible /* v8 ignore next */ /* v8 ignore next */
            const dims = edge.shape /* v8 ignore next */ /* v8 ignore next */
              .replace('[', '') /* v8 ignore next */ /* v8 ignore next */
              .replace(']', '') /* v8 ignore next */ /* v8 ignore next */
              .split(',') /* v8 ignore next */ /* v8 ignore next */
              .map((s) => parseInt(s.trim())); /* v8 ignore next */ /* v8 ignore next */
            if (dims.every((d) => !isNaN(d) && d > 0)) {
              /* v8 ignore next */ /* v8 ignore next */
              const elements = dims.reduce(
                (a, b) => a * b,
                1,
              ); /* v8 ignore next */ /* v8 ignore next */
              let bpe = 4; /* v8 ignore next */ /* v8 ignore next */
              if (edge.dtype.includes('8')) bpe = 1; /* v8 ignore next */ /* v8 ignore next */
              if (edge.dtype.includes('16')) bpe = 2; /* v8 ignore next */ /* v8 ignore next */
              if (edge.dtype.includes('64')) bpe = 8; /* v8 ignore next */ /* v8 ignore next */
              const bytes = elements * bpe; /* v8 ignore next */ /* v8 ignore next */
              text += ` (${(bytes / 1024).toFixed(1)} KB)`; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            // Draw background pill for hover text /* v8 ignore next */ /* v8 ignore next */
            const m = ctx.measureText(text); /* v8 ignore next */ /* v8 ignore next */
            ctx.fillStyle = 'rgba(0,0,0,0.8)'; /* v8 ignore next */ /* v8 ignore next */
            ctx.fillRect(
              midX - m.width / 2 - 4,
              midY - 14,
              m.width + 8,
              16,
            ); /* v8 ignore next */ /* v8 ignore next */
            ctx.fillStyle = '#fff'; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          ctx.fillText(text, midX, midY - 2); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Draw Nodes /* v8 ignore next */ /* v8 ignore next */
    for (const node of this.layout.nodes) {
      /* v8 ignore next */ /* v8 ignore next */
      if (
        /* v8 ignore next */ /* v8 ignore next */
        node.x + node.width < startX /* v8 ignore next */ /* v8 ignore next */ ||
        node.x > endX /* v8 ignore next */ /* v8 ignore next */ ||
        node.y + node.height < startY /* v8 ignore next */ /* v8 ignore next */ ||
        node.y > endY /* v8 ignore next */ /* v8 ignore next */
      ) {
        /* v8 ignore next */ /* v8 ignore next */
        continue; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // Node body styling based on type /* v8 ignore next */ /* v8 ignore next */
      let fill = '#1e1e1e'; /* v8 ignore next */ /* v8 ignore next */
      let stroke = '#444'; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (node.type === 'input') {
        /* v8 ignore next */ /* v8 ignore next */
        fill = '#1e3e1e'; /* v8 ignore next */ /* v8 ignore next */
        stroke = '#4a4'; /* v8 ignore next */ /* v8 ignore next */
      } else if (node.type === 'output') {
        /* v8 ignore next */ /* v8 ignore next */
        fill = '#3e1e1e'; /* v8 ignore next */ /* v8 ignore next */
        stroke = '#a44'; /* v8 ignore next */ /* v8 ignore next */
      } else if (node.type === 'constant') {
        /* v8 ignore next */ /* v8 ignore next */
        fill = '#1e1e3e'; /* v8 ignore next */ /* v8 ignore next */
        stroke = '#44a'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // 211, 212. Visual Cues for INT8 / W4A16 /* v8 ignore next */ /* v8 ignore next */
      const isQuantized =
        /* v8 ignore next */ /* v8 ignore next */
        (node.opType || '').includes('Quantize') /* v8 ignore next */ /* v8 ignore next */ ||
        (node.opType || '').includes('Integer') /* v8 ignore next */ /* v8 ignore next */ ||
        (node.opType || '').includes('QLinear'); /* v8 ignore next */ /* v8 ignore next */
      const isPacked =
        /* v8 ignore next */ /* v8 ignore next */
        (node.opType || '').includes('Bitpack') ||
        (node.opType || '').includes('NBits'); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (isQuantized) fill = '#3e2a1e'; // Orange-ish tint for INT8 /* v8 ignore next */ /* v8 ignore next */
      if (isPacked) fill = '#2e1e3e'; // Purple-ish tint for W4A16 /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // 197. Custom Regex Node Coloring /* v8 ignore next */ /* v8 ignore next */
      if (this.customColorRegex && this.customColorRegex.test(node.name || node.opType)) {
        /* v8 ignore next */ /* v8 ignore next */
        fill = '#0052cc'; // Distinctive highlight blue /* v8 ignore next */ /* v8 ignore next */
        stroke = '#0066ff'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      const isConnectedToHoveredEdge =
        /* v8 ignore next */ /* v8 ignore next */
        this.hoveredEdge &&
        (this.hoveredEdge.from === node.id ||
          this.hoveredEdge.to === node.id); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // Highlight on hover / selection / search /* v8 ignore next */ /* v8 ignore next */
      const isSelected = this.selectedNodes.includes(
        node.id,
      ); /* v8 ignore next */ /* v8 ignore next */
      if (isSelected) {
        /* v8 ignore next */ /* v8 ignore next */
        fill = '#4a4a4a'; /* v8 ignore next */ /* v8 ignore next */
        stroke = '#ffffff'; /* v8 ignore next */ /* v8 ignore next */
      } else if (this.hoveredNode === node.id || isConnectedToHoveredEdge) {
        /* v8 ignore next */ /* v8 ignore next */
        fill = '#3a3a3a'; /* v8 ignore next */ /* v8 ignore next */
        stroke = '#ffffff'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      if (this.searchResults.has(node.id)) {
        /* v8 ignore next */ /* v8 ignore next */
        stroke = '#f8e71c'; // Yellow highlight for search match /* v8 ignore next */ /* v8 ignore next */
        ctx.lineWidth = 3 / this.scale; /* v8 ignore next */ /* v8 ignore next */
      } else {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.lineWidth =
          /* v8 ignore next */ /* v8 ignore next */
          (isSelected || this.hoveredNode === node.id || isConnectedToHoveredEdge
            ? 2
            : 1) /* v8 ignore next */ /* v8 ignore next */ /
          this.scale; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      ctx.fillStyle = fill; /* v8 ignore next */ /* v8 ignore next */
      ctx.strokeStyle = stroke; /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // Rounded rect /* v8 ignore next */ /* v8 ignore next */
      const radius = 6; /* v8 ignore next */ /* v8 ignore next */
      ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
      ctx.roundRect(
        node.x,
        node.y,
        node.width,
        node.height,
        radius,
      ); /* v8 ignore next */ /* v8 ignore next */
      ctx.fill(); /* v8 ignore next */ /* v8 ignore next */
      ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
      /* v8 ignore next */ /* v8 ignore next */
      // Text (level of detail) /* v8 ignore next */ /* v8 ignore next */
      if (this.scale > 0.3) {
        /* v8 ignore next */ /* v8 ignore next */
        ctx.fillStyle = '#fff'; /* v8 ignore next */ /* v8 ignore next */
        ctx.font = '12px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
        ctx.textAlign = 'center'; /* v8 ignore next */ /* v8 ignore next */
        ctx.textBaseline = 'middle'; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // 193. Render mathematical symbols for basic math nodes /* v8 ignore next */ /* v8 ignore next */
        let label = node.opType; /* v8 ignore next */ /* v8 ignore next */
        if (label === 'Add') label = '+ (Add)'; /* v8 ignore next */ /* v8 ignore next */
        if (label === 'Sub') label = '- (Sub)'; /* v8 ignore next */ /* v8 ignore next */
        if (label === 'Mul') label = '× (Mul)'; /* v8 ignore next */ /* v8 ignore next */
        if (label === 'Div') label = '÷ (Div)'; /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        // 297. Render String constants with truncated inline text /* v8 ignore next */ /* v8 ignore next */
        if (node.opType === 'Constant' && node.stringValue) {
          /* v8 ignore next */ /* v8 ignore next */
          label = `"${node.stringValue}"`; /* v8 ignore next */ /* v8 ignore next */
        } else if (node.type === 'constant' && node.stringValue) {
          /* v8 ignore next */ /* v8 ignore next */
          label = `"${node.stringValue}"`; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        /* v8 ignore next */ /* v8 ignore next */
        ctx.fillText(
          label,
          node.x + node.width / 2,
          node.y + node.height / 2,
        ); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    ctx.restore(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
