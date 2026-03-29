/* eslint-disable */
// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { MetricsDashboard } from '../../src/components/MetricsDashboard';

describe('MetricsDashboard', () => {
  it('should render initial empty state', () => {
    const dashboard = new MetricsDashboard();
    const el = (dashboard as object).element as HTMLElement;
    expect(el.className).toBe('demo-metrics-dashboard');

    const ttft = el.querySelector('.demo-metric-ttft .demo-metric-value');
    const tps = el.querySelector('.demo-metric-tps .demo-metric-value');
    const total = el.querySelector('.demo-metric-total .demo-metric-value');

    expect(ttft?.textContent).toBe('--');
    expect(tps?.textContent).toBe('--');
    expect(total?.textContent).toBe('--');
  });

  it('should update metrics properly', () => {
    const dashboard = new MetricsDashboard();
    const el = (dashboard as object).element as HTMLElement;
    dashboard.mount(document.body);

    dashboard.updateMetrics({
      ttftMs: 12.345,
      tps: 60.123,
      totalLatencyMs: 1250.99
    });

    const ttft = el.querySelector('.demo-metric-ttft .demo-metric-value');
    const tps = el.querySelector('.demo-metric-tps .demo-metric-value');
    const total = el.querySelector('.demo-metric-total .demo-metric-value');

    expect(ttft?.textContent).toBe('12.35 ms');
    expect(tps?.textContent).toBe('60.12');
    expect(total?.textContent).toBe('1250.99 ms');

    dashboard.unmount();
  });

  it('should handle zero metrics as empty state', () => {
    const dashboard = new MetricsDashboard();
    const el = (dashboard as object).element as HTMLElement;

    dashboard.updateMetrics({
      ttftMs: 0,
      tps: 0,
      totalLatencyMs: 0
    });

    const ttft = el.querySelector('.demo-metric-ttft .demo-metric-value');
    const tps = el.querySelector('.demo-metric-tps .demo-metric-value');
    const total = el.querySelector('.demo-metric-total .demo-metric-value');

    expect(ttft?.textContent).toBe('--');
    expect(tps?.textContent).toBe('--');
    expect(total?.textContent).toBe('--');
  });
});
