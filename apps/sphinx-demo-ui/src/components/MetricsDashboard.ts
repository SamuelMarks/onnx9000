import { Component } from '../core/Component';

export interface MetricsData {
  ttftMs: number; // Time to First Token
  tps: number; // Tokens per second
  totalLatencyMs: number;
}

export class MetricsDashboard extends Component<HTMLDivElement> {
  private ttftValue!: HTMLSpanElement;
  private tpsValue!: HTMLSpanElement;
  private totalValue!: HTMLSpanElement;

  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-metrics-dashboard';
    container.style.display = 'flex';
    container.style.gap = '16px';
    container.style.padding = '12px';
    container.style.backgroundColor = 'var(--bg-color-header)';
    container.style.borderBottom = '1px solid var(--border-color)';

    // TTFT
    const ttftCard = this.createMetricCard('TTFT', 'demo-metric-ttft');
    this.ttftValue = ttftCard.querySelector('.demo-metric-value') as HTMLSpanElement;
    container.appendChild(ttftCard);

    // TPS
    const tpsCard = this.createMetricCard('Tokens/Sec', 'demo-metric-tps');
    this.tpsValue = tpsCard.querySelector('.demo-metric-value') as HTMLSpanElement;
    container.appendChild(tpsCard);

    // Total Latency
    const totalCard = this.createMetricCard('Total Latency', 'demo-metric-total');
    this.totalValue = totalCard.querySelector('.demo-metric-value') as HTMLSpanElement;
    container.appendChild(totalCard);

    return container;
  }

  private createMetricCard(label: string, className: string): HTMLDivElement {
    const card = document.createElement('div');
    card.className = `demo-metric-card ${className}`;
    card.style.flex = '1';
    card.style.padding = '8px';
    card.style.backgroundColor = 'var(--bg-color-main)';
    card.style.borderRadius = '4px';
    card.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
    card.style.textAlign = 'center';

    const lbl = document.createElement('div');
    lbl.className = 'demo-metric-label';
    lbl.textContent = label;
    lbl.style.fontSize = '0.75rem';
    lbl.style.color = 'var(--text-color-secondary)';
    lbl.style.marginBottom = '4px';
    lbl.style.textTransform = 'uppercase';

    const val = document.createElement('div');
    val.className = 'demo-metric-value';
    val.textContent = '--';
    val.style.fontSize = '1.25rem';
    val.style.fontWeight = 'bold';
    val.style.color = 'var(--text-color-primary)';

    card.appendChild(lbl);
    card.appendChild(val);

    return card;
  }

  public updateMetrics(data: MetricsData): void {
    this.ttftValue.textContent = data.ttftMs > 0 ? `${data.ttftMs.toFixed(2)} ms` : '--';
    this.tpsValue.textContent = data.tps > 0 ? `${data.tps.toFixed(2)}` : '--';
    this.totalValue.textContent =
      data.totalLatencyMs > 0 ? `${data.totalLatencyMs.toFixed(2)} ms` : '--';
  }
}
