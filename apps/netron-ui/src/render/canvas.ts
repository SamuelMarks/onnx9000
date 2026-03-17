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
    // this.offsetY = ch / 2 - (this.layout.height * this.scale) / 2;
    this.offsetY = 100 * this.scale;
  }

  private resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.render();
  }

  private hoveredNode: string | null = null;
  public selectedNode: string | null = null;
  private searchResults: Set<string> = new Set();

  public setSearchResults(results: string[]) {
    this.searchResults = new Set(results);
    this.render();
  }

  public focusNode(nodeId: string) {
    if (!this.layout) return;
    const node = this.layout.nodes.find((n) => n.id === nodeId);
    if (!node) return;

    // Animate or snap to center
    this.offsetX = this.canvas.width / 2 - (node.x + node.width / 2) * this.scale;
    this.offsetY = this.canvas.height / 2 - (node.y + node.height / 2) * this.scale;
    this.render();
  }

  private setupEvents() {
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.hasMovedDuringDrag = false;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => {
      if (this.isDragging && !this.hasMovedDuringDrag) {
        this.selectedNode = this.hoveredNode;
        this.onSelect(this.selectedNode);
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
        this.render();
      } else {
        // Sub-millisecond hit testing for hovering
        if (this.layout) {
          const mx = (e.clientX - this.offsetX) / this.scale;
          const my = (e.clientY - this.offsetY) / this.scale;

          let found: string | null = null;
          // Loop backwards to hit top-most items first
          for (let i = this.layout.nodes.length - 1; i >= 0; i--) {
            const n = this.layout.nodes[i]!;
            if (mx >= n.x && mx <= n.x + n.width && my >= n.y && my <= n.y + n.height) {
              found = n.id;
              break;
            }
          }

          if (this.hoveredNode !== found) {
            this.hoveredNode = found;
            this.canvas.style.cursor = found ? 'pointer' : 'default';
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

    // Draw grid lines
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
      ctx.restore();
    }

    // Draw Edges
    for (const edge of this.layout.edges) {
      // Color by dtype
      if (edge.dtype?.startsWith('float') || edge.dtype?.startsWith('bfloat')) {
        ctx.strokeStyle = '#4A90E2'; // Blue
      } else if (edge.dtype?.startsWith('int') || edge.dtype?.startsWith('uint')) {
        ctx.strokeStyle = '#7ED321'; // Green
      } else if (edge.dtype === 'bool') {
        ctx.strokeStyle = '#F8E71C'; // Yellow
      } else if (edge.dtype === 'string') {
        ctx.strokeStyle = '#F5A623'; // Orange
      } else {
        ctx.strokeStyle = '#888888'; // Default
      }

      ctx.lineWidth = 2 / this.scale;
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
        if (this.scale > 0.8 && edge.shape && edge.dtype) {
          ctx.fillStyle = '#aaa';
          ctx.font = '10px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          const midX = (p1.x + p2.x) / 2;
          const midY = (p1.y + p2.y) / 2;
          ctx.fillText(`${edge.dtype} ${edge.shape}`, midX, midY - 2);
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

      // Highlight on hover / selection / search
      if (this.selectedNode === node.id) {
        fill = '#4a4a4a';
        stroke = '#ffffff';
      } else if (this.hoveredNode === node.id) {
        fill = '#3a3a3a';
        stroke = '#aaaaaa';
      }

      if (this.searchResults.has(node.id)) {
        stroke = '#f8e71c'; // Yellow highlight for search match
        ctx.lineWidth = 3 / this.scale;
      } else {
        ctx.lineWidth =
          (this.selectedNode === node.id || this.hoveredNode === node.id ? 2 : 1) / this.scale;
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
        ctx.fillText(node.opType, node.x + node.width / 2, node.y + node.height / 2);
      }
    }

    ctx.restore();
  }
}
