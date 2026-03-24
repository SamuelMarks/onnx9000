import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';

export interface OliveConfig {
  quantizationLevel: 'FP16' | 'INT8' | 'None';
  enableStaticShapeInference: boolean;
  enableTransformerFusion: boolean;
}

export class OliveConfigPanel extends Component<HTMLDivElement> {
  private config: OliveConfig = {
    quantizationLevel: 'None',
    enableStaticShapeInference: true,
    enableTransformerFusion: false
  };

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-olive-config-panel';
    container.style.padding = '12px';
    container.style.borderTop = '1px solid var(--border-color)';
    container.style.backgroundColor = 'var(--bg-color-panel)';
    container.style.fontSize = '0.85rem';

    const title = document.createElement('h3');
    title.textContent = 'Olive Optimization Settings';
    title.style.marginTop = '0';
    title.style.marginBottom = '12px';
    title.style.fontSize = '0.95rem';
    container.appendChild(title);

    // Quantization Dropdown
    const quantLabel = document.createElement('label');
    quantLabel.textContent = 'Quantization: ';
    quantLabel.style.display = 'block';
    quantLabel.style.marginBottom = '8px';

    const quantSelect = document.createElement('select');
    quantSelect.className = 'demo-olive-quant-select';
    ['None', 'FP16', 'INT8'].forEach((opt) => {
      const option = document.createElement('option');
      option.value = opt;
      option.textContent = opt;
      quantSelect.appendChild(option);
    });
    quantLabel.appendChild(quantSelect);
    container.appendChild(quantLabel);

    // Static Shape Inference Toggle
    const shapeLabel = document.createElement('label');
    shapeLabel.style.display = 'block';
    shapeLabel.style.marginBottom = '8px';

    const shapeCheckbox = document.createElement('input');
    shapeCheckbox.type = 'checkbox';
    shapeCheckbox.className = 'demo-olive-shape-checkbox';
    shapeCheckbox.checked = this.config.enableStaticShapeInference;

    shapeLabel.appendChild(shapeCheckbox);
    shapeLabel.appendChild(document.createTextNode(' Enable Static Shape Inference'));
    container.appendChild(shapeLabel);

    // Transformer Fusion Toggle
    const fusionLabel = document.createElement('label');
    fusionLabel.style.display = 'block';
    fusionLabel.style.marginBottom = '8px';

    const fusionCheckbox = document.createElement('input');
    fusionCheckbox.type = 'checkbox';
    fusionCheckbox.className = 'demo-olive-fusion-checkbox';
    fusionCheckbox.checked = this.config.enableTransformerFusion;

    fusionLabel.appendChild(fusionCheckbox);
    fusionLabel.appendChild(document.createTextNode(' Enable Transformer Fusion'));
    container.appendChild(fusionLabel);

    return container;
  }

  protected onMount(): void {
    const quantSelect = this.element.querySelector('.demo-olive-quant-select') as HTMLSelectElement;
    const shapeCheckbox = this.element.querySelector(
      '.demo-olive-shape-checkbox'
    ) as HTMLInputElement;
    const fusionCheckbox = this.element.querySelector(
      '.demo-olive-fusion-checkbox'
    ) as HTMLInputElement;

    this.addDOMListener(quantSelect, 'change', (e) => {
      this.config.quantizationLevel = (e.target as HTMLSelectElement).value as any;
      this.emitConfig();
    });

    this.addDOMListener(shapeCheckbox, 'change', (e) => {
      this.config.enableStaticShapeInference = (e.target as HTMLInputElement).checked;
      this.emitConfig();
    });

    this.addDOMListener(fusionCheckbox, 'change', (e) => {
      this.config.enableTransformerFusion = (e.target as HTMLInputElement).checked;
      this.emitConfig();
    });
  }

  private emitConfig(): void {
    globalEventBus.emit('OLIVE_CONFIG_CHANGED', this.config);
  }

  public getConfig(): OliveConfig {
    return { ...this.config };
  }
}
