/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';
import { PipelineNode } from '../core/PipelineNode';
import { globalEventBus } from '../core/EventBus';
import { i18n, Language } from '../core/I18n';

export class Breadcrumbs extends Component<HTMLDivElement> {
  private items: PipelineNode[] = [];
  private langSelect!: HTMLSelectElement;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-breadcrumbs-container';
    container.setAttribute('aria-label', 'Pipeline History');
    container.style.display = 'flex';
    container.style.justifyContent = 'space-between';
    container.style.alignItems = 'center';

    const crumbsDiv = document.createElement('div');
    crumbsDiv.className = 'demo-breadcrumbs-list';
    crumbsDiv.style.display = 'flex';
    crumbsDiv.style.alignItems = 'center';
    crumbsDiv.style.flex = '1';

    // Initial render
    this.renderItems(crumbsDiv);

    // Language selector
    this.langSelect = document.createElement('select');
    this.langSelect.className = 'demo-lang-select';
    this.langSelect.setAttribute('aria-label', 'Select Language');
    this.langSelect.style.marginLeft = '16px';
    this.langSelect.style.padding = '4px 8px';
    this.langSelect.style.borderRadius = '4px';
    this.langSelect.style.border = '1px solid var(--border-color)';
    this.langSelect.style.backgroundColor = 'var(--md-sys-color-surface)';
    this.langSelect.style.color = 'var(--md-sys-color-on-surface)';

    const langs: { code: Language; label: string }[] = [
      { code: 'en', label: 'English' },
      { code: 'es', label: 'Español' },
      { code: 'fr', label: 'Français' },
      { code: 'de', label: 'Deutsch' },
      { code: 'ja', label: '日本語' }
    ];

    langs.forEach((lang) => {
      const option = document.createElement('option');
      option.value = lang.code;
      option.textContent = lang.label;
      if (lang.code === i18n.getLanguage()) {
        option.selected = true;
      }
      this.langSelect.appendChild(option);
    });

    this.langSelect.addEventListener('change', (e) => {
      const target = e.target as HTMLSelectElement;
      i18n.setLanguage(target.value as Language);
    });

    container.appendChild(crumbsDiv);
    container.appendChild(this.langSelect);

    return container;
  }

  public renderItems(container: HTMLDivElement): void {
    container.innerHTML = '';

    if (this.items.length === 0) {
      const placeholder = document.createElement('span');
      placeholder.className = 'demo-breadcrumb-placeholder';
      placeholder.textContent = 'Pipeline is empty. Select a source and generate a target.';
      container.appendChild(placeholder);
    }

    this.items.forEach((node, index) => {
      const itemBtn = document.createElement('button');
      itemBtn.className = 'demo-breadcrumb-item';
      itemBtn.textContent = node.description;
      itemBtn.title = `Revert back to: ${node.description}`;

      // If it's the last item, it's active
      if (index === this.items.length - 1) {
        itemBtn.classList.add('active');
        itemBtn.disabled = true; // Can't revert to current
      }

      this.addDOMListener(itemBtn, 'click', () => {
        globalEventBus.emit('PIPELINE_REVERT_REQUESTED', node.id);
      });

      container.appendChild(itemBtn);

      // Add separator if not last
      if (index < this.items.length - 1) {
        const sep = document.createElement('span');
        sep.className = 'demo-breadcrumb-separator';
        sep.innerHTML = '&gt;';
        container.appendChild(sep);
      }
    });
  }

  protected onMount(): void {
    const crumbsDiv = this.element.querySelector('.demo-breadcrumbs-list') as HTMLDivElement;

    this.onCleanup(
      globalEventBus.on<PipelineNode>('PIPELINE_STEP_ADDED', (node) => {
        this.items.push(node);
        this.renderItems(crumbsDiv);
      })
    );

    this.onCleanup(
      globalEventBus.on<PipelineNode>('PIPELINE_STEP_REMOVED', (node) => {
        // Find and remove all nodes after and including this one
        const idx = this.items.findIndex((n) => n.id === node.id);
        if (idx !== -1) {
          this.items = this.items.slice(0, idx);
          this.renderItems(crumbsDiv);
        }
      })
    );
  }
}
