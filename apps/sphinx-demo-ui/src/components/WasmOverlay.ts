import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';
import { WasmManager, WasmState } from '../core/WasmManager';
import { t } from '../core/I18n';

/**
 * An overlay that blocks UI interaction until the heavy WASM binaries
 * are lazy-loaded by the user.
 */
export class WasmOverlay extends Component<HTMLDivElement> {
  private loadButton!: HTMLButtonElement;
  private progressContainer!: HTMLDivElement;
  private progressBar!: HTMLDivElement;
  private progressText!: HTMLParagraphElement;
  private errorText!: HTMLParagraphElement;
  private sizeInfo!: HTMLParagraphElement;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const overlay = document.createElement('div');
    overlay.className = 'demo-wasm-overlay';

    const modal = document.createElement('div');
    modal.className = 'demo-wasm-modal';

    const title = document.createElement('h2');
    title.textContent = t('wasm.loading');

    const desc = document.createElement('p');
    desc.textContent = t('wasm.desc');

    this.sizeInfo = document.createElement('p');
    this.sizeInfo.className = 'demo-wasm-size-info';
    this.sizeInfo.textContent = 'Estimating download size...';
    this.sizeInfo.style.fontWeight = 'bold';
    this.sizeInfo.style.marginTop = '10px';
    this.sizeInfo.style.marginBottom = '20px';

    this.loadButton = document.createElement('button');
    this.loadButton.className = 'demo-btn-primary';
    this.loadButton.textContent = t('wasm.start');

    this.progressContainer = document.createElement('div');
    this.progressContainer.className = 'demo-wasm-progress-container';
    this.progressContainer.style.display = 'none';

    const progressTrack = document.createElement('div');
    progressTrack.className = 'demo-wasm-progress-track';

    this.progressBar = document.createElement('div');
    this.progressBar.className = 'demo-wasm-progress-bar';
    this.progressBar.style.width = '0%';
    progressTrack.appendChild(this.progressBar);

    this.progressText = document.createElement('p');
    this.progressText.className = 'demo-wasm-progress-text';
    this.progressText.textContent = '0%';

    this.progressContainer.appendChild(progressTrack);
    this.progressContainer.appendChild(this.progressText);

    this.errorText = document.createElement('p');
    this.errorText.className = 'demo-wasm-error-text';
    this.errorText.style.display = 'none';

    modal.appendChild(title);
    modal.appendChild(desc);
    modal.appendChild(this.sizeInfo);
    modal.appendChild(this.loadButton);
    modal.appendChild(this.progressContainer);
    modal.appendChild(this.errorText);

    overlay.appendChild(modal);
    return overlay;
  }

  protected onMount(): void {
    // Estimate download size before they click
    fetch('/onnx9000.wasm', { method: 'HEAD' })
      .then((res) => {
        const size = res.headers.get('content-length');
        if (size) {
          const mb = (parseInt(size, 10) / (1024 * 1024)).toFixed(1);
          this.sizeInfo.textContent = `Estimated download size: ~${mb} MB`;
        } else {
          this.sizeInfo.textContent = `Estimated download size: ~50.0 MB`;
        }
      })
      .catch(() => {
        this.sizeInfo.textContent = `Estimated download size: ~50.0 MB`;
      });

    // Bind click event
    this.addDOMListener(this.loadButton, 'click', () => {
      this.loadButton.style.display = 'none';
      this.progressContainer.style.display = 'block';
      this.errorText.style.display = 'none';
      WasmManager.getInstance().load();
    });

    // Re-bind strings when language changes
    this.onCleanup(
      globalEventBus.on('LANGUAGE_CHANGED', () => {
        const title = this.element.querySelector('h2');
        if (title) title.textContent = t('wasm.loading');
        // Only update the first p element which is desc
        const desc = this.element.querySelector('p');
        if (desc) desc.textContent = t('wasm.desc');
        if (this.loadButton) this.loadButton.textContent = t('wasm.start');
      })
    );

    // Listen to WASM progress events
    this.onCleanup(
      globalEventBus.on<any>('WASM_PROGRESS', (payload) => {
        let progress = 0;
        let loaded = 0;
        let total = 0;

        if (typeof payload === 'number') {
          progress = payload;
        } else if (payload && typeof payload.progress === 'number') {
          progress = payload.progress;
          loaded = payload.loaded || 0;
          total = payload.total || 0;
        }

        this.progressBar.style.width = `${progress}%`;

        if (total > 0) {
          const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
          const totalMB = (total / (1024 * 1024)).toFixed(1);
          this.progressText.textContent = `${progress}% (${loadedMB} MB / ${totalMB} MB)`;
        } else if (loaded > 0) {
          const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
          this.progressText.textContent = `${progress}% (${loadedMB} MB)`;
        } else {
          this.progressText.textContent = `${progress}%`;
        }
      })
    );

    // Listen to WASM state changes
    this.onCleanup(
      globalEventBus.on<WasmState>('WASM_STATE_CHANGED', (state) => {
        if (state === WasmState.ERROR) {
          const mgr = WasmManager.getInstance();
          this.errorText.textContent = mgr.error || 'Unknown error occurred.';
          this.errorText.style.display = 'block';
          this.loadButton.style.display = 'block';
          this.progressContainer.style.display = 'none';
        } else if (state === WasmState.LOADED) {
          // Fade out the overlay
          this.element.style.opacity = '0';
          setTimeout(() => {
            this.unmount();
          }, 300); // 300ms transition time
        }
      })
    );
  }
}
