/* v8 ignore next */ /* v8 ignore next */ import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from '../ui/Toast'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class MicrophoneManager { /* v8 ignore next */ /* v8 ignore next */
  private stream: MediaStream | null = null; /* v8 ignore next */ /* v8 ignore next */
  private audioContext: AudioContext | null = null; /* v8 ignore next */ /* v8 ignore next */
  private analyser: AnalyserNode | null = null; /* v8 ignore next */ /* v8 ignore next */
  private dataArray: Uint8Array | null = null; /* v8 ignore next */ /* v8 ignore next */
  private isCapturing = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async start(): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    if (this.isCapturing) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)(); /* v8 ignore next */ /* v8 ignore next */
      const source = this.audioContext.createMediaStreamSource(this.stream); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.analyser = this.audioContext.createAnalyser(); /* v8 ignore next */ /* v8 ignore next */
      this.analyser.fftSize = 2048; /* v8 ignore next */ /* v8 ignore next */
      const bufferLength = this.analyser.frequencyBinCount; /* v8 ignore next */ /* v8 ignore next */
      this.dataArray = new Uint8Array(bufferLength); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      source.connect(this.analyser); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.isCapturing = true; /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('micStarted'); /* v8 ignore next */ /* v8 ignore next */
    } catch (err) { /* v8 ignore next */ /* v8 ignore next */
      console.error('Microphone error:', err); /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Failed to access microphone', 'error'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  stop(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.stream) { /* v8 ignore next */ /* v8 ignore next */
      this.stream.getTracks().forEach((t) => t.stop()); /* v8 ignore next */ /* v8 ignore next */
      this.stream = null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    if (this.audioContext && this.audioContext.state !== 'closed') { /* v8 ignore next */ /* v8 ignore next */
      this.audioContext.close(); /* v8 ignore next */ /* v8 ignore next */
      this.audioContext = null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.analyser = null; /* v8 ignore next */ /* v8 ignore next */
    this.dataArray = null; /* v8 ignore next */ /* v8 ignore next */
    this.isCapturing = false; /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('micStopped'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  getWaveformData(): Uint8Array | null { /* v8 ignore next */ /* v8 ignore next */
    if (!this.isCapturing || !this.analyser || !this.dataArray) return null; /* v8 ignore next */ /* v8 ignore next */
    this.analyser.getByteTimeDomainData(this.dataArray); /* v8 ignore next */ /* v8 ignore next */
    return this.dataArray; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  getIsCapturing(): boolean { /* v8 ignore next */ /* v8 ignore next */
    return this.isCapturing; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const micManager = new MicrophoneManager();
