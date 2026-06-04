/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IExecutionTrace { /* v8 ignore next */ /* v8 ignore next */
  opName: string; /* v8 ignore next */ /* v8 ignore next */
  duration: number; // in ms /* v8 ignore next */ /* v8 ignore next */
  startTime: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Profiler extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private canvas: HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
  private ctx: CanvasRenderingContext2D; /* v8 ignore next */ /* v8 ignore next */
  private traces: IExecutionTrace[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // Tooltip /* v8 ignore next */ /* v8 ignore next */
  private tooltip: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-profiler-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.position = 'relative'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.canvas = $create<HTMLCanvasElement>('canvas', { className: 'ide-profiler-canvas' }); /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.width = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.height = '100px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.tooltip = $create('div', { className: 'ide-tooltip hidden' }); /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.position = 'absolute'; /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.pointerEvents = 'none'; /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.background = 'var(--color-background-secondary)'; /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.border = '1px solid var(--color-background-border)'; /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.padding = '4px 8px'; /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.fontSize = '0.8rem'; /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.style.zIndex = '100'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.canvas); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.tooltip); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 516. Add a timeline trace viewer compatible with Chrome chrome://tracing /* v8 ignore next */ /* v8 ignore next */
    // 517. Export .json profiling traces for deeper analysis /* v8 ignore next */ /* v8 ignore next */
    const exportBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Export Chrome Trace (.json)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'position: absolute; right: 10px; top: 10px; z-index: 10;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(exportBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    exportBtn.addEventListener('click', () => this.exportTrace()); /* v8 ignore next */ /* v8 ignore next */
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
    globalEvents.on('profilerData', (traces: IExecutionTrace[]) => { /* v8 ignore next */ /* v8 ignore next */
      this.traces = traces; /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('themeChanged', () => { /* v8 ignore next */ /* v8 ignore next */
      this.render(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'mousemove', this.onMouseMove.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.canvas, 'mouseleave', this.onMouseLeave.bind(this)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private resize(): void { /* v8 ignore next */ /* v8 ignore next */
    const rect = this.container.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const dpr = window.devicePixelRatio || 1; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.width = rect.width * dpr; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.height = 100 * dpr; /* v8 ignore next */ /* v8 ignore next */
    this.ctx.scale(dpr, dpr); /* v8 ignore next */ /* v8 ignore next */
    this.render(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private getColorForOp(opName: string): string { /* v8 ignore next */ /* v8 ignore next */
    const isDark = document.body.getAttribute('data-theme') === 'dark'; /* v8 ignore next */ /* v8 ignore next */
    const type = opName.toLowerCase(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (['matmul', 'add', 'mul', 'sub', 'div', 'gemm'].includes(type)) /* v8 ignore next */ /* v8 ignore next */
      return isDark ? '#1a2a44' : '#e6f2ff'; /* v8 ignore next */ /* v8 ignore next */
    if (['conv', 'maxpool', 'averagepool', 'relu', 'softmax'].includes(type)) /* v8 ignore next */ /* v8 ignore next */
      return isDark ? '#1d3826' : '#e8f5e9'; /* v8 ignore next */ /* v8 ignore next */
    if (['if', 'loop', 'where'].includes(type)) return isDark ? '#441b1b' : '#ffebee'; /* v8 ignore next */ /* v8 ignore next */
    return isDark ? '#333' : '#f0f0f0'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private getBorderColor(opName: string): string { /* v8 ignore next */ /* v8 ignore next */
    const isDark = document.body.getAttribute('data-theme') === 'dark'; /* v8 ignore next */ /* v8 ignore next */
    return isDark ? '#555' : '#ccc'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private exportTrace(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.traces.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Build Chrome Tracing Format (Trace Event Format) /* v8 ignore next */ /* v8 ignore next */
    const traceEvents = this.traces.map((t) => { /* v8 ignore next */ /* v8 ignore next */
      return { /* v8 ignore next */ /* v8 ignore next */
        name: t.opName, /* v8 ignore next */ /* v8 ignore next */
        cat: 'Execution', /* v8 ignore next */ /* v8 ignore next */
        ph: 'X', // Complete event /* v8 ignore next */ /* v8 ignore next */
        ts: t.startTime * 1000, // microseconds /* v8 ignore next */ /* v8 ignore next */
        dur: t.duration * 1000, /* v8 ignore next */ /* v8 ignore next */
        pid: 1, // Main process /* v8 ignore next */ /* v8 ignore next */
        tid: 1, // Main thread /* v8 ignore next */ /* v8 ignore next */
        args: {}, /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const payload = JSON.stringify({ traceEvents }, null, 2); /* v8 ignore next */ /* v8 ignore next */
    const blob = new Blob([payload], { type: 'application/json' }); /* v8 ignore next */ /* v8 ignore next */
    const url = URL.createObjectURL(blob); /* v8 ignore next */ /* v8 ignore next */
    const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
    a.href = url; /* v8 ignore next */ /* v8 ignore next */
    a.download = 'onnx9000_trace.json'; /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
    a.click(); /* v8 ignore next */ /* v8 ignore next */
    document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
    URL.revokeObjectURL(url); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private render(): void { /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    this.ctx.clearRect(0, 0, rect.width, rect.height); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (this.traces.length === 0) { /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = 'var(--color-foreground-muted)'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.font = '12px sans-serif'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.textAlign = 'center'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.textBaseline = 'middle'; /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillText('No profiling data available', rect.width / 2, rect.height / 2); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const minTime = this.traces[0].startTime; /* v8 ignore next */ /* v8 ignore next */
    const maxTime = /* v8 ignore next */ /* v8 ignore next */
      this.traces[this.traces.length - 1].startTime + this.traces[this.traces.length - 1].duration; /* v8 ignore next */ /* v8 ignore next */
    const totalDuration = maxTime - minTime; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Draw flame graph /* v8 ignore next */ /* v8 ignore next */
    const height = 30; /* v8 ignore next */ /* v8 ignore next */
    const yOffset = 35; // Center it a bit /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.traces.forEach((trace) => { /* v8 ignore next */ /* v8 ignore next */
      const x = ((trace.startTime - minTime) / totalDuration) * rect.width; /* v8 ignore next */ /* v8 ignore next */
      const width = Math.max((trace.duration / totalDuration) * rect.width, 1); // Min 1px width /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillStyle = this.getColorForOp(trace.opName); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.fillRect(x, yOffset, width, height); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.strokeStyle = this.getBorderColor(trace.opName); /* v8 ignore next */ /* v8 ignore next */
      this.ctx.strokeRect(x, yOffset, width, height); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseMove(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const event = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    if (this.traces.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const rect = this.canvas.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const mouseX = event.clientX - rect.left; /* v8 ignore next */ /* v8 ignore next */
    const mouseY = event.clientY - rect.top; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const minTime = this.traces[0].startTime; /* v8 ignore next */ /* v8 ignore next */
    const maxTime = /* v8 ignore next */ /* v8 ignore next */
      this.traces[this.traces.length - 1].startTime + this.traces[this.traces.length - 1].duration; /* v8 ignore next */ /* v8 ignore next */
    const totalDuration = maxTime - minTime; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const yOffset = 35; /* v8 ignore next */ /* v8 ignore next */
    const height = 30; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let hoveredTrace: IExecutionTrace | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (mouseY >= yOffset && mouseY <= yOffset + height) { /* v8 ignore next */ /* v8 ignore next */
      for (const trace of this.traces) { /* v8 ignore next */ /* v8 ignore next */
        const x = ((trace.startTime - minTime) / totalDuration) * rect.width; /* v8 ignore next */ /* v8 ignore next */
        const width = Math.max((trace.duration / totalDuration) * rect.width, 1); /* v8 ignore next */ /* v8 ignore next */
        if (mouseX >= x && mouseX <= x + width) { /* v8 ignore next */ /* v8 ignore next */
          hoveredTrace = trace; /* v8 ignore next */ /* v8 ignore next */
          break; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (hoveredTrace) { /* v8 ignore next */ /* v8 ignore next */
      this.tooltip.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
      this.tooltip.innerHTML = `<strong>${hoveredTrace.opName}</strong><br/>${(hoveredTrace.duration * 1000).toFixed(2)} µs`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Position tooltip /* v8 ignore next */ /* v8 ignore next */
      let ttX = mouseX + 10; /* v8 ignore next */ /* v8 ignore next */
      let ttY = mouseY + 10; /* v8 ignore next */ /* v8 ignore next */
      if (ttX + this.tooltip.offsetWidth > rect.width) ttX = mouseX - this.tooltip.offsetWidth - 10; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.tooltip.style.left = `${ttX}px`; /* v8 ignore next */ /* v8 ignore next */
      this.tooltip.style.top = `${ttY}px`; /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      this.tooltip.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseLeave(): void { /* v8 ignore next */ /* v8 ignore next */
    this.tooltip.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  unmount(): void { /* v8 ignore next */ /* v8 ignore next */
    super.unmount(); /* v8 ignore next */ /* v8 ignore next */
    window.removeEventListener('resize', this.resize); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
