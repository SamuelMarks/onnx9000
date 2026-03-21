import { GraphLayout, LayoutNode, LayoutEdge } from '../layout/dag';

export class CanvasRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private layout: GraphLayout | null = null;

  private scale: number = 1;
  private offsetX: number = 0;
  private offsetY: number = 0;

  private isDragging: boolean = false;
  private hasMovedDuringDrag: boolean = false;
  private lastMouseX: number = 0;
  private lastMouseY: number = 0;

  public onSelect: (nodeId: string | null) => void = () => {};

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context not supported');
    this.ctx = ctx;

    this.setupEvents();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  setLayout(layout: GraphLayout) {
    this.layout = layout;
    this.centerGraph();
    this.render();
  }

  private centerGraph() {
    if (!this.layout) return;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    // Auto-scale to fit horizontally or vertically, max scale 1
    const scaleX = cw / (this.layout.width + 200);
    const scaleY = ch / (this.layout.height + 200);
    this.scale = Math.min(1, Math.min(scaleX, scaleY));

    // Center it
    this.offsetX = cw / 2;
    this.offsetY = 100 * this.scale;

    // Attempt restore from localStorage
    try {
      const savedScale = localStorage.getItem('onnxModifier_scale');
      const savedOffsetX = localStorage.getItem('onnxModifier_offsetX');
      const savedOffsetY = localStorage.getItem('onnxModifier_offsetY');
      if (savedScale && savedOffsetX && savedOffsetY) {
        this.scale = parseFloat(savedScale);
        this.offsetX = parseFloat(savedOffsetX);
        this.offsetY = parseFloat(savedOffsetY);
      }
    } catch (e) {}
  }

  private resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.render();
  }

  private hoveredNode: string | null = null;
  private hoveredEdge: LayoutEdge | null = null;
  public selectedNodes: string[] = [];
  private searchResults: Set<string> = new Set();

  private hideControlEdges: boolean = false;
  private customColorRegex: RegExp | null = null;

  public setSearchResults(results: string[]) {
    this.searchResults = new Set(results);
    this.render();
  }

  public setFilterControlEdges(hide: boolean) {
    this.hideControlEdges = hide;
    this.render();
  }

  public setCustomColorRegex(pattern: string) {
    if (!pattern.trim()) {
      this.customColorRegex = null;
    } else {
      try {
        this.customColorRegex = new RegExp(pattern, 'i');
      } catch (e) {
        this.customColorRegex = null; // invalid regex, ignore
      }
    }
    this.render();
  }

  public focusNode(nodeId: string) {
    if (!this.layout) return;
    const node = this.layout.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    this.selectedNodes = [nodeId];
    // Animate or snap to center
    this.offsetX = this.canvas.width / 2 - (node.x + node.width / 2) * this.scale;
    this.offsetY = this.canvas.height / 2 - (node.y + node.height / 2) * this.scale;
    this.render();
  }

  private setupEvents() {
    // 283. Implement touch event handling for iPad/Mobile Safari rendering.
    this.canvas.addEventListener(
      'touchstart',
      (e) => {
        if (e.touches.length === 1) {
          this.isDragging = true;
          this.hasMovedDuringDrag = false;
          this.lastMouseX = e.touches[0]!.clientX;
          this.lastMouseY = e.touches[0]!.clientY;
        }
      },
      { passive: false },
    );

    this.canvas.addEventListener(
      'touchmove',
      (e) => {
        if (this.isDragging && e.touches.length === 1) {
          e.preventDefault(); // Prevent scrolling
          this.hasMovedDuringDrag = true;
          this.offsetX += e.touches[0]!.clientX - this.lastMouseX;
          this.offsetY += e.touches[0]!.clientY - this.lastMouseY;
          this.lastMouseX = e.touches[0]!.clientX;
          this.lastMouseY = e.touches[0]!.clientY;
          try {
            localStorage.setItem('onnxModifier_offsetX', String(this.offsetX));
            localStorage.setItem('onnxModifier_offsetY', String(this.offsetY));
          } catch (e) {}
          this.render();
        }
      },
      { passive: false },
    );

    this.canvas.addEventListener('touchend', () => {
      this.isDragging = false;
      // We don't simulate click selection on touch yet, just pan
    });

    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.hasMovedDuringDrag = false;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mouseup', (e) => {
      if (this.isDragging && !this.hasMovedDuringDrag) {
        if (this.hoveredNode) {
          if (e.shiftKey) {
            if (this.selectedNodes.includes(this.hoveredNode)) {
              this.selectedNodes = this.selectedNodes.filter((n) => n !== this.hoveredNode);
            } else {
              this.selectedNodes.push(this.hoveredNode);
            }
          } else {
            this.selectedNodes = [this.hoveredNode];
          }
          this.onSelect(
            this.selectedNodes.length > 0
              ? this.selectedNodes[this.selectedNodes.length - 1]!
              : null,
          );
        } else {
          this.selectedNodes = [];
          this.onSelect(null);
        }
        this.render();
      }
      this.isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.hasMovedDuringDrag = true;
        this.offsetX += e.clientX - this.lastMouseX;
        this.offsetY += e.clientY - this.lastMouseY;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        try {
          localStorage.setItem('onnxModifier_offsetX', String(this.offsetX));
          localStorage.setItem('onnxModifier_offsetY', String(this.offsetY));
        } catch (e) {}
        this.render();
      } else {
        // Sub-millisecond hit testing for hovering
        if (this.layout) {
          const mx = (e.clientX - this.offsetX) / this.scale;
          const my = (e.clientY - this.offsetY) / this.scale;

          let foundNode: string | null = null;
          // Loop backwards to hit top-most items first
          for (let i = this.layout.nodes.length - 1; i >= 0; i--) {
            const n = this.layout.nodes[i]!;
            if (mx >= n.x && mx <= n.x + n.width && my >= n.y && my <= n.y + n.height) {
              foundNode = n.id;
              break;
            }
          }

          let foundEdge: LayoutEdge | null = null;
          if (!foundNode) {
            // Hit test edges if no node found
            for (const edge of this.layout.edges) {
              const p1 = edge.points[0];
              const p2 = edge.points[1];
              if (p1 && p2) {
                const isVertical = Math.abs(p2.y - p1.y) > Math.abs(p2.x - p1.x);
                const path = new Path2D();
                path.moveTo(p1.x, p1.y);
                if (isVertical) {
                  const midY = (p1.y + p2.y) / 2;
                  path.bezierCurveTo(p1.x, midY, p2.x, midY, p2.x, p2.y);
                } else {
                  const midX = (p1.x + p2.x) / 2;
                  path.bezierCurveTo(midX, p1.y, midX, p2.y, p2.x, p2.y);
                }

                this.ctx.lineWidth = 10 / this.scale; // Thicker hit area
                if (this.ctx.isPointInStroke(path, mx, my)) {
                  foundEdge = edge;
                  break;
                }
              }
            }
          }

          let changed = false;
          if (this.hoveredNode !== foundNode) {
            this.hoveredNode = foundNode;
            changed = true;
          }
          if (this.hoveredEdge !== foundEdge) {
            this.hoveredEdge = foundEdge;
            changed = true;
          }

          if (changed) {
            this.canvas.style.cursor = foundNode || foundEdge ? 'pointer' : 'default';
            this.render();
          }
        }
      }
    });

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const zoomSensitivity = 0.001;
        const zoom = 1 - e.deltaY * zoomSensitivity;

        const mouseX = e.clientX;
        const mouseY = e.clientY;

        // Zoom around cursor
        this.offsetX = mouseX - (mouseX - this.offsetX) * zoom;
        this.offsetY = mouseY - (mouseY - this.offsetY) * zoom;
        this.scale *= zoom;

        // Limit zoom
        this.scale = Math.max(0.01, Math.min(this.scale, 5));

        // 279. Support saving layout preferences to localStorage
        try {
          localStorage.setItem('onnxModifier_scale', String(this.scale));
          localStorage.setItem('onnxModifier_offsetX', String(this.offsetX));
          localStorage.setItem('onnxModifier_offsetY', String(this.offsetY));
        } catch (e) {}

        this.render();
      },
      { passive: false },
    );
  }

  render() {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!this.layout) return;

    ctx.save();
    ctx.translate(this.offsetX, this.offsetY);
    ctx.scale(this.scale, this.scale);

    const startX = -this.offsetX / this.scale;
    const startY = -this.offsetY / this.scale;
    const endX = startX + canvas.width / this.scale;
    const endY = startY + canvas.height / this.scale;

    // Draw Grid
    if (this.scale > 0.2) {
      ctx.save();
      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1 / this.scale;
      const gridSize = 50;

      const gridStartX = Math.floor(startX / gridSize) * gridSize;
      const gridStartY = Math.floor(startY / gridSize) * gridSize;
      const gridEndX = endX + gridSize;
      const gridEndY = endY + gridSize;

      ctx.beginPath();
      for (let x = gridStartX; x < gridEndX; x += gridSize) {
        ctx.moveTo(x, gridStartY);
        ctx.lineTo(x, gridEndY);
      }
      for (let y = gridStartY; y < gridEndY; y += gridSize) {
        ctx.moveTo(gridStartX, y);
        ctx.lineTo(gridEndX, y);
      }
      ctx.stroke();
      ctx.setLineDash([]); // Reset line dash for next shapes
      ctx.restore();
    }

    // Draw Groups (NameScopes)
    if (this.layout.groups) {
      for (const group of this.layout.groups) {
        if (
          group.x + group.width < startX ||
          group.x > endX ||
          group.y + group.height < startY ||
          group.y > endY
        ) {
          continue;
        }

        ctx.fillStyle = `rgba(255, 255, 255, ${0.02 + group.depth * 0.01})`;
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 + group.depth * 0.05})`;
        ctx.lineWidth = 1 / this.scale;

        ctx.beginPath();
        ctx.roundRect(group.x, group.y, group.width, group.height, 8);
        ctx.fill();
        ctx.stroke();

        if (this.scale > 0.1) {
          ctx.fillStyle = `rgba(255, 255, 255, ${0.4 + group.depth * 0.1})`;
          ctx.font = '12px sans-serif';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          ctx.fillText(group.name, group.x + 10, group.y + 10);
        }
      }
    }

    // Draw Edges
    for (const edge of this.layout.edges) {
      if (this.hideControlEdges) {
        // Typical ONNX control edges have empty strings as tensor names, or are boolean
        if (!edge.tensorName || edge.dtype === 'bool') continue;
      }
      const isHoveredEdge = this.hoveredEdge === edge;

      // Color by dtype
      // 133. Colorblind-friendly palette
      // Using IBM Design-ish / Wong palette colors for colorblind accessibility
      if (isHoveredEdge) {
        ctx.strokeStyle = '#ffffff'; // Highlighted edge color
      } else if (edge.dtype?.startsWith('float') || edge.dtype?.startsWith('bfloat')) {
        ctx.strokeStyle = '#56B4E9'; // Sky Blue
      } else if (edge.dtype?.startsWith('int') || edge.dtype?.startsWith('uint')) {
        ctx.strokeStyle = '#009E73'; // Bluish Green
      } else if (edge.dtype === 'bool') {
        ctx.strokeStyle = '#F0E442'; // Yellow
      } else if (edge.dtype === 'string') {
        ctx.strokeStyle = '#E69F00'; // Orange
      } else {
        ctx.strokeStyle = '#888888'; // Default
      }

      ctx.lineWidth = (isHoveredEdge ? 4 : 2) / this.scale;
      ctx.beginPath();

      const p1 = edge.points[0];
      const p2 = edge.points[1];

      if (p1 && p2) {
        ctx.moveTo(p1.x, p1.y);
        // Smooth bezier curve for DAG flow
        // Assuming mostly vertical flow for the control points logic, but we can do generic
        const isVertical = Math.abs(p2.y - p1.y) > Math.abs(p2.x - p1.x);

        if (isVertical) {
          const midY = (p1.y + p2.y) / 2;
          ctx.bezierCurveTo(p1.x, midY, p2.x, midY, p2.x, p2.y);
        } else {
          const midX = (p1.x + p2.x) / 2;
          ctx.bezierCurveTo(midX, p1.y, midX, p2.y, p2.x, p2.y);
        }

        ctx.stroke();

        // Render data types and shapes as edge labels
        if ((this.scale > 0.8 || isHoveredEdge) && edge.shape && edge.dtype) {
          ctx.fillStyle = isHoveredEdge ? '#fff' : '#aaa';
          ctx.font = isHoveredEdge ? '12px sans-serif' : '10px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          const midX = (p1.x + p2.x) / 2;
          const midY = (p1.y + p2.y) / 2;
          // 210. Edge hover preview of tensor shape sizes
          let text = `${edge.dtype} ${edge.shape}`;
          if (isHoveredEdge) {
            // estimate byte size if possible
            const dims = edge.shape
              .replace('[', '')
              .replace(']', '')
              .split(',')
              .map((s) => parseInt(s.trim()));
            if (dims.every((d) => !isNaN(d) && d > 0)) {
              const elements = dims.reduce((a, b) => a * b, 1);
              let bpe = 4;
              if (edge.dtype.includes('8')) bpe = 1;
              if (edge.dtype.includes('16')) bpe = 2;
              if (edge.dtype.includes('64')) bpe = 8;
              const bytes = elements * bpe;
              text += ` (${(bytes / 1024).toFixed(1)} KB)`;
            }
            // Draw background pill for hover text
            const m = ctx.measureText(text);
            ctx.fillStyle = 'rgba(0,0,0,0.8)';
            ctx.fillRect(midX - m.width / 2 - 4, midY - 14, m.width + 8, 16);
            ctx.fillStyle = '#fff';
          }
          ctx.fillText(text, midX, midY - 2);
        }
      }
    }

    // Draw Nodes
    for (const node of this.layout.nodes) {
      if (
        node.x + node.width < startX ||
        node.x > endX ||
        node.y + node.height < startY ||
        node.y > endY
      ) {
        continue;
      }

      // Node body styling based on type
      let fill = '#1e1e1e';
      let stroke = '#444';

      if (node.type === 'input') {
        fill = '#1e3e1e';
        stroke = '#4a4';
      } else if (node.type === 'output') {
        fill = '#3e1e1e';
        stroke = '#a44';
      } else if (node.type === 'constant') {
        fill = '#1e1e3e';
        stroke = '#44a';
      }

      // 211, 212. Visual Cues for INT8 / W4A16
      const isQuantized =
        (node.opType || '').includes('Quantize') ||
        (node.opType || '').includes('Integer') ||
        (node.opType || '').includes('QLinear');
      const isPacked =
        (node.opType || '').includes('Bitpack') || (node.opType || '').includes('NBits');

      if (isQuantized) fill = '#3e2a1e'; // Orange-ish tint for INT8
      if (isPacked) fill = '#2e1e3e'; // Purple-ish tint for W4A16

      // 197. Custom Regex Node Coloring
      if (this.customColorRegex && this.customColorRegex.test(node.name || node.opType)) {
        fill = '#0052cc'; // Distinctive highlight blue
        stroke = '#0066ff';
      }

      const isConnectedToHoveredEdge =
        this.hoveredEdge && (this.hoveredEdge.from === node.id || this.hoveredEdge.to === node.id);

      // Highlight on hover / selection / search
      const isSelected = this.selectedNodes.includes(node.id);
      if (isSelected) {
        fill = '#4a4a4a';
        stroke = '#ffffff';
      } else if (this.hoveredNode === node.id || isConnectedToHoveredEdge) {
        fill = '#3a3a3a';
        stroke = '#ffffff';
      }

      if (this.searchResults.has(node.id)) {
        stroke = '#f8e71c'; // Yellow highlight for search match
        ctx.lineWidth = 3 / this.scale;
      } else {
        ctx.lineWidth =
          (isSelected || this.hoveredNode === node.id || isConnectedToHoveredEdge ? 2 : 1) /
          this.scale;
      }

      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;

      // Rounded rect
      const radius = 6;
      ctx.beginPath();
      ctx.roundRect(node.x, node.y, node.width, node.height, radius);
      ctx.fill();
      ctx.stroke();

      // Text (level of detail)
      if (this.scale > 0.3) {
        ctx.fillStyle = '#fff';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // 193. Render mathematical symbols for basic math nodes
        let label = node.opType;
        if (label === 'Add') label = '+ (Add)';
        if (label === 'Sub') label = '- (Sub)';
        if (label === 'Mul') label = '× (Mul)';
        if (label === 'Div') label = '÷ (Div)';

        // 297. Render String constants with truncated inline text
        if (node.opType === 'Constant' && node.stringValue) {
          label = `"${node.stringValue}"`;
        } else if (node.type === 'constant' && node.stringValue) {
          label = `"${node.stringValue}"`;
        }

        ctx.fillText(label, node.x + node.width / 2, node.y + node.height / 2);
      }
    }

    ctx.restore();
  }
}
