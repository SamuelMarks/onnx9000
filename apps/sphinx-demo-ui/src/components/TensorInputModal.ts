/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';
import { t } from '../core/I18n';

export interface TensorShape {
  name: string;
  type: string; // e.g., 'float32', 'int64'
  dims: (number | string)[]; // e.g., ['N', 3, 224, 224]
}

/**
 * A modal component dynamically generated from ONNX model inputs
 * to capture inference payload data.
 */
export class TensorInputModal extends Component<HTMLDivElement> {
  private inputs: TensorShape[] = [];
  private inputsContainer!: HTMLDivElement;
  private generateBtn!: HTMLButtonElement;
  private submitBtn!: HTMLButtonElement;
  private closeBtn!: HTMLButtonElement;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const overlay = document.createElement('div');
    overlay.className = 'demo-tensor-modal-overlay';
    overlay.style.display = 'none'; // Hidden by default

    const modal = document.createElement('div');
    modal.className = 'demo-tensor-modal';

    const header = document.createElement('div');
    header.className = 'demo-tensor-modal-header';
    header.innerHTML = `<h3>Configure Inference Inputs</h3>`;

    this.closeBtn = document.createElement('button');
    this.closeBtn.className = 'demo-btn-close';
    this.closeBtn.innerHTML = '&times;';
    header.appendChild(this.closeBtn);

    this.inputsContainer = document.createElement('div');
    this.inputsContainer.className = 'demo-tensor-inputs-container';

    const footer = document.createElement('div');
    footer.className = 'demo-tensor-modal-footer';

    this.generateBtn = document.createElement('button');
    this.generateBtn.className = 'demo-btn-secondary';
    this.generateBtn.textContent = t('tensor.generate');

    this.submitBtn = document.createElement('button');
    this.submitBtn.className = 'demo-btn-primary';
    this.submitBtn.textContent = t('tensor.submit');

    footer.appendChild(this.generateBtn);
    footer.appendChild(this.submitBtn);

    modal.appendChild(header);
    modal.appendChild(this.inputsContainer);
    modal.appendChild(footer);

    overlay.appendChild(modal);
    return overlay;
  }

  protected onMount(): void {
    this.addDOMListener(this.closeBtn, 'click', () => this.hide());
    this.addDOMListener(this.generateBtn, 'click', () => this.fillRandomData());
    this.addDOMListener(this.submitBtn, 'click', () => this.submit());

    // Hide if clicking on overlay background
    this.addDOMListener(this.element, 'click', (e) => {
      if (e.target === this.element) this.hide();
    });

    this.onCleanup(
      globalEventBus.on('LANGUAGE_CHANGED', () => {
        if (this.generateBtn) this.generateBtn.textContent = t('tensor.generate');
        if (this.submitBtn) this.submitBtn.textContent = t('tensor.submit');
        if (this.closeBtn) this.closeBtn.setAttribute('aria-label', t('tensor.close'));
      })
    );
  }

  public show(inputs: TensorShape[]): void {
    this.inputs = inputs;
    this.renderInputs();
    this.element.style.display = 'flex';
  }

  public hide(): void {
    this.element.style.display = 'none';
  }

  private renderInputs(): void {
    this.inputsContainer.innerHTML = '';

    if (this.inputs.length === 0) {
      this.inputsContainer.innerHTML = '<p>No inputs required.</p>';
      return;
    }

    this.inputs.forEach((input) => {
      const group = document.createElement('div');
      group.className = 'demo-tensor-input-group';

      const label = document.createElement('label');
      label.innerHTML = `<strong>${input.name}</strong> <span>(${input.type}, shape: [${input.dims.join(', ')}])</span>`;

      const fileInput = document.createElement('input');
      fileInput.type = 'file';
      fileInput.accept = 'image/jpeg, image/png';
      fileInput.className = 'demo-tensor-file-input';
      fileInput.setAttribute('data-name', input.name);

      const preview = document.createElement('canvas');
      preview.className = 'demo-tensor-preview';
      preview.style.display = 'none';

      group.appendChild(label);
      group.appendChild(fileInput);
      group.appendChild(preview);

      this.inputsContainer.appendChild(group);

      // Handle image upload and resize via Canvas
      this.addDOMListener(fileInput, 'change', (e) => this.handleImageUpload(e, preview, input));
    });
  }

  private handleImageUpload(e: Event, canvas: HTMLCanvasElement, shape: TensorShape): void {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      // Find expected dims, defaulting to 224 if dynamic or missing
      let width = 224;
      let height = 224;

      if (shape.dims.length >= 4) {
        const w = shape.dims[3];
        const h = shape.dims[2];
        if (typeof w === 'number') width = w;
        if (typeof h === 'number') height = h;
      }

      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(img, 0, 0, width, height);
      }
      canvas.style.display = 'block';
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  public fillRandomData(): void {
    // We just emit the event indicating we want random data without touching the DOM forms
    // The Web Worker will construct normal distribution floats for the inputs.
    globalEventBus.emit('SHOW_TOAST', {
      message: 'Random data configured for execution.',
      type: 'info'
    });
  }

  private submit(): void {
    // Collect data or pass a flag to generate
    this.hide();
    globalEventBus.emit('EXECUTE_INFERENCE_REQUEST', { useRandom: true }); // Mock logic
  }
}
