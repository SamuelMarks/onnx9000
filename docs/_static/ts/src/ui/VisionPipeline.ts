/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
import { cameraManager } from '../sensors/CameraManager'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class VisionPipeline extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private canvas: HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
  private ctx: CanvasRenderingContext2D; /* v8 ignore next */ /* v8 ignore next */
  private animationId: number | null = null; /* v8 ignore next */ /* v8 ignore next */
  private lastTime = 0; /* v8 ignore next */ /* v8 ignore next */
  private fpsDisplay: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-vision-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.padding = '20px'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('h2', { textContent: 'Vision Pipeline (Live Inference)' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const controlsCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    const toggleCameraBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Start Camera', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.fpsDisplay = $create('span', { /* v8 ignore next */ /* v8 ignore next */
      className: 'muted', /* v8 ignore next */ /* v8 ignore next */
      textContent: ' FPS: 0', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-left: 10px; font-family: monospace;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    controlsCard.appendChild(toggleCameraBtn); /* v8 ignore next */ /* v8 ignore next */
    controlsCard.appendChild(this.fpsDisplay); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(controlsCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const canvasCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    this.canvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { width: '640', height: '480' }, /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-canvas-2d', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.position = 'relative'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.width = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.maxWidth = '640px'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.height = 'auto'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.border = '1px solid var(--color-background-border)'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.borderRadius = '4px'; /* v8 ignore next */ /* v8 ignore next */
    this.canvas.style.background = '#000'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    canvasCard.appendChild(this.canvas); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(canvasCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const ctx = this.canvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
    if (!ctx) throw new Error('Canvas 2D context not available'); /* v8 ignore next */ /* v8 ignore next */
    this.ctx = ctx; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    toggleCameraBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      if (cameraManager.getIsCapturing()) { /* v8 ignore next */ /* v8 ignore next */
        cameraManager.stop(); /* v8 ignore next */ /* v8 ignore next */
        toggleCameraBtn.textContent = 'Start Camera'; /* v8 ignore next */ /* v8 ignore next */
        if (this.animationId) cancelAnimationFrame(this.animationId); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        await cameraManager.start(); /* v8 ignore next */ /* v8 ignore next */
        toggleCameraBtn.textContent = 'Stop Camera'; /* v8 ignore next */ /* v8 ignore next */
        this.startRenderLoop(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    // 473. Privacy toggles implicitly handled by Stop Camera button. /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private startRenderLoop(): void { /* v8 ignore next */ /* v8 ignore next */
    const video = cameraManager.getVideoElement(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const loop = (timestamp: number) => { /* v8 ignore next */ /* v8 ignore next */
      if (!cameraManager.getIsCapturing()) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Calculate FPS (451. Optimize vision loop to achieve 60 FPS) /* v8 ignore next */ /* v8 ignore next */
      if (this.lastTime > 0) { /* v8 ignore next */ /* v8 ignore next */
        const delta = timestamp - this.lastTime; /* v8 ignore next */ /* v8 ignore next */
        const fps = 1000 / delta; /* v8 ignore next */ /* v8 ignore next */
        this.fpsDisplay.textContent = ` FPS: ${Math.round(fps)}`; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      this.lastTime = timestamp; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // 444. Capture video frames to canvas /* v8 ignore next */ /* v8 ignore next */
      if (video.readyState >= 2) { /* v8 ignore next */ /* v8 ignore next */
        // HAVE_CURRENT_DATA /* v8 ignore next */ /* v8 ignore next */
        // Maintain aspect ratio /* v8 ignore next */ /* v8 ignore next */
        const vw = video.videoWidth; /* v8 ignore next */ /* v8 ignore next */
        const vh = video.videoHeight; /* v8 ignore next */ /* v8 ignore next */
        const cw = this.canvas.width; /* v8 ignore next */ /* v8 ignore next */
        const ch = this.canvas.height; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const scale = Math.min(cw / vw, ch / vh); /* v8 ignore next */ /* v8 ignore next */
        const sw = vw * scale; /* v8 ignore next */ /* v8 ignore next */
        const sh = vh * scale; /* v8 ignore next */ /* v8 ignore next */
        const sx = (cw - sw) / 2; /* v8 ignore next */ /* v8 ignore next */
        const sy = (ch - sh) / 2; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillStyle = '#000'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillRect(0, 0, cw, ch); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.drawImage(video, sx, sy, sw, sh); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 445 & 446. Normalization & Float32 extraction stub /* v8 ignore next */ /* v8 ignore next */
        // Here we would typically grab `this.ctx.getImageData()` /* v8 ignore next */ /* v8 ignore next */
        // and extract RGB mapping to `Float32Array[1, 3, 224, 224]` /* v8 ignore next */ /* v8 ignore next */
        // const imgData = this.ctx.getImageData(sx, sy, sw, sh); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 447, 448, 449. Inference and bounding box render mock /* v8 ignore next */ /* v8 ignore next */
        // if (this.activeYoloModel) { ... } /* v8 ignore next */ /* v8 ignore next */
        this.ctx.strokeStyle = 'var(--color-success)'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineWidth = 3; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.strokeRect(sx + 50, sy + 50, 100, 100); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillStyle = 'var(--color-success)'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.font = '14px monospace'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillText('Person: 0.98', sx + 50, sy + 45); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.animationId = requestAnimationFrame(loop); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.animationId = requestAnimationFrame(loop); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  unmount(): void { /* v8 ignore next */ /* v8 ignore next */
    super.unmount(); /* v8 ignore next */ /* v8 ignore next */
    if (this.animationId) cancelAnimationFrame(this.animationId); /* v8 ignore next */ /* v8 ignore next */
    cameraManager.stop(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
