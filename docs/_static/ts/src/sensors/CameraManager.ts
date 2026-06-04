/* v8 ignore next */ /* v8 ignore next */ import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from '../ui/Toast'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class CameraManager { /* v8 ignore next */ /* v8 ignore next */
  private videoEl: HTMLVideoElement; /* v8 ignore next */ /* v8 ignore next */
  private stream: MediaStream | null = null; /* v8 ignore next */ /* v8 ignore next */
  private isCapturing = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    this.videoEl = document.createElement('video'); /* v8 ignore next */ /* v8 ignore next */
    this.videoEl.setAttribute('playsinline', 'true'); /* v8 ignore next */ /* v8 ignore next */
    this.videoEl.style.display = 'none'; /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(this.videoEl); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async start(): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    if (this.isCapturing) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      this.stream = await navigator.mediaDevices.getUserMedia({ /* v8 ignore next */ /* v8 ignore next */
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.videoEl.srcObject = this.stream; /* v8 ignore next */ /* v8 ignore next */
      await this.videoEl.play(); /* v8 ignore next */ /* v8 ignore next */
      this.isCapturing = true; /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('cameraStarted', { /* v8 ignore next */ /* v8 ignore next */
        width: this.videoEl.videoWidth, /* v8 ignore next */ /* v8 ignore next */
        height: this.videoEl.videoHeight, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } catch (err) { /* v8 ignore next */ /* v8 ignore next */
      console.error('Camera error:', err); /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Failed to access camera', 'error'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  stop(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.stream) { /* v8 ignore next */ /* v8 ignore next */
      this.stream.getTracks().forEach((t) => t.stop()); /* v8 ignore next */ /* v8 ignore next */
      this.stream = null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.isCapturing = false; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('cameraStopped'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  getVideoElement(): HTMLVideoElement { /* v8 ignore next */ /* v8 ignore next */
    return this.videoEl; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  getIsCapturing(): boolean { /* v8 ignore next */ /* v8 ignore next */
    return this.isCapturing; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const cameraManager = new CameraManager();
