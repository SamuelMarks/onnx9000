/* eslint-disable */
// @ts-nocheck
import { Component } from '../core/Component';
import { globalEventBus } from '../core/EventBus';
import { LogEntry } from '../core/Logger';

export class Console extends Component<HTMLDivElement> {
  private outputDiv!: HTMLDivElement;
  private clearBtn!: HTMLButtonElement;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-console-container';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.height = '100%';

    const toolbar = document.createElement('div');
    toolbar.className = 'demo-console-toolbar';

    this.clearBtn = document.createElement('button');
    this.clearBtn.className = 'demo-btn-secondary demo-console-clear-btn';
    this.clearBtn.textContent = 'Clear Console';
    toolbar.appendChild(this.clearBtn);

    this.outputDiv = document.createElement('div');
    this.outputDiv.className = 'demo-console-output';

    container.appendChild(toolbar);
    container.appendChild(this.outputDiv);

    return container;
  }

  protected onMount(): void {
    this.addDOMListener(this.clearBtn, 'click', () => this.clear());

    this.onCleanup(
      globalEventBus.on<LogEntry>('CONSOLE_LOG', (entry) => {
        this.appendLog(entry);
      })
    );
  }

  public appendLog(entry: LogEntry): void {
    const line = document.createElement('div');
    line.className = `demo-console-line demo-console-level-${entry.level}`;

    const timestamp = document.createElement('span');
    timestamp.className = 'demo-console-time';
    timestamp.textContent = `[${entry.timestamp.toISOString().split('T')[1].replace('Z', '')}] `;

    const msg = document.createElement('span');
    msg.className = 'demo-console-msg';
    msg.textContent = entry.message;

    line.appendChild(timestamp);
    line.appendChild(msg);

    this.outputDiv.appendChild(line);

    // Auto-scroll
    this.outputDiv.scrollTop = this.outputDiv.scrollHeight;

    // Line limit
    if (this.outputDiv.children.length > 1000) {
      this.outputDiv.removeChild(this.outputDiv.firstChild as Node);
    }
  }

  public clear(): void {
    this.outputDiv.innerHTML = '';
  }
}
