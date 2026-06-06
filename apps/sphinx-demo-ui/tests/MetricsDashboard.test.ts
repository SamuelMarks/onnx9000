// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { MetricsDashboard } from '../src/components/MetricsDashboard.js';

describe('MetricsDashboard', () => {
  it('should render and update', () => {
    const dash = new MetricsDashboard();
    document.body.appendChild(dash.element);

    dash.updateMetrics({ ttftMs: 15.5, tps: 100, totalLatencyMs: 50 });
    expect(dash.element.textContent).toContain('15.50');
    expect(dash.element.textContent).toContain('100');
  });
});
