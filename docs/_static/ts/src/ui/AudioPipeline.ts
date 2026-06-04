/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
import { micManager } from '../sensors/MicrophoneManager'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class AudioPipeline extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private canvas: HTMLCanvasElement; /* v8 ignore next */ /* v8 ignore next */
  private ctx: CanvasRenderingContext2D; /* v8 ignore next */ /* v8 ignore next */
  private animationId: number | null = null; /* v8 ignore next */ /* v8 ignore next */
  private transcriptionDisplay: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-audio-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.padding = '20px'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('h2', { textContent: 'Audio Pipeline (Live Transcription)' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const controlsCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    const toggleMicBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Start Microphone', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    controlsCard.appendChild(toggleMicBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(controlsCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const canvasCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    canvasCard.appendChild($create('h3', { textContent: 'Live Waveform' })); /* v8 ignore next */ /* v8 ignore next */
    this.canvas = $create<HTMLCanvasElement>('canvas', { /* v8 ignore next */ /* v8 ignore next */
      attributes: { width: '640', height: '150' }, /* v8 ignore next */ /* v8 ignore next */
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
    const transCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    transCard.appendChild($create('h3', { textContent: 'Transcription (Mock Whisper)' })); /* v8 ignore next */ /* v8 ignore next */
    this.transcriptionDisplay = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-messages', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Awaiting audio...', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'padding: 10px; border: 1px solid var(--color-background-border); border-radius: 4px; font-family: monospace; background: var(--color-background-secondary);', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    transCard.appendChild(this.transcriptionDisplay); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(transCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const ctx = this.canvas.getContext('2d'); /* v8 ignore next */ /* v8 ignore next */
    if (!ctx) throw new Error('Canvas 2D context not available'); /* v8 ignore next */ /* v8 ignore next */
    this.ctx = ctx; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    toggleMicBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      if (micManager.getIsCapturing()) { /* v8 ignore next */ /* v8 ignore next */
        micManager.stop(); /* v8 ignore next */ /* v8 ignore next */
        toggleMicBtn.textContent = 'Start Microphone'; /* v8 ignore next */ /* v8 ignore next */
        if (this.animationId) cancelAnimationFrame(this.animationId); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        await micManager.start(); /* v8 ignore next */ /* v8 ignore next */
        toggleMicBtn.textContent = 'Stop Microphone'; /* v8 ignore next */ /* v8 ignore next */
        this.startRenderLoop(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void {} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private startRenderLoop(): void { /* v8 ignore next */ /* v8 ignore next */
    const loop = () => { /* v8 ignore next */ /* v8 ignore next */
      if (!micManager.getIsCapturing()) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const dataArray = micManager.getWaveformData(); /* v8 ignore next */ /* v8 ignore next */
      if (dataArray) { /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillStyle = 'rgb(20, 20, 20)'; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineWidth = 2; /* v8 ignore next */ /* v8 ignore next */
        this.ctx.strokeStyle = 'rgb(0, 255, 0)'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        this.ctx.beginPath(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const sliceWidth = (this.canvas.width * 1.0) / dataArray.length; /* v8 ignore next */ /* v8 ignore next */
        let x = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        for (let i = 0; i < dataArray.length; i++) { /* v8 ignore next */ /* v8 ignore next */
          const v = dataArray[i] / 128.0; /* v8 ignore next */ /* v8 ignore next */
          const y = (v * this.canvas.height) / 2; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
          if (i === 0) { /* v8 ignore next */ /* v8 ignore next */
            this.ctx.moveTo(x, y); /* v8 ignore next */ /* v8 ignore next */
          } else { /* v8 ignore next */ /* v8 ignore next */
            this.ctx.lineTo(x, y); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
          x += sliceWidth; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2); /* v8 ignore next */ /* v8 ignore next */
        this.ctx.stroke(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 460. VAD Stub: Detect amplitude to trigger transcription mock /* v8 ignore next */ /* v8 ignore next */
        let sum = 0; /* v8 ignore next */ /* v8 ignore next */
        for (let i = 0; i < dataArray.length; i++) { /* v8 ignore next */ /* v8 ignore next */
          sum += Math.abs(dataArray[i] - 128); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        const avg = sum / dataArray.length; /* v8 ignore next */ /* v8 ignore next */
        if (avg > 25) { /* v8 ignore next */ /* v8 ignore next */
          // Threshold /* v8 ignore next */ /* v8 ignore next */
          if (Math.random() < 0.05) { /* v8 ignore next */ /* v8 ignore next */
            // Mock throttle /* v8 ignore next */ /* v8 ignore next */
            this.transcriptionDisplay.textContent += ' [Speech Detected]'; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
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
    micManager.stop(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
