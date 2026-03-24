import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OnnxVisualizer } from '../../src/components/OnnxVisualizer';
import { globalEventBus } from '../../src/core/EventBus';
import { VizGraph } from '../../src/core/OnnxAdapter';

// Mock cytoscape entirely
const mockCyInstance = {
  on: vi.fn(),
  destroy: vi.fn(),
  elements: vi.fn(() => ({ remove: vi.fn() })),
  add: vi.fn(),
  layout: vi.fn(() => ({ run: vi.fn() })),
  fit: vi.fn(),
  center: vi.fn(),
  resize: vi.fn()
};

vi.mock('cytoscape', () => {
  return {
    default: vi.fn(() => mockCyInstance)
  };
});

describe('OnnxVisualizer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    globalEventBus.clearAll();
  });

  it('should initialize cytoscape on mount', async () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    const mockedCy = vi.mocked((await import('cytoscape')).default);
    expect(mockedCy).toHaveBeenCalled();

    const config: any = mockedCy.mock.calls[0][0];
    expect(config.container).toBe((viz as any).element);
    expect(config.style.length).toBeGreaterThan(0);

    viz.unmount();
    expect(mockCyInstance.destroy).toHaveBeenCalledTimes(1);
  });

  it('should render graph correctly via event bus', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    const graph: VizGraph = {
      inputs: [],
      outputs: [],
      nodes: [{ id: 'n1', name: 'Relu', opType: 'Relu', inputs: [], outputs: [] }]
    };

    globalEventBus.emit('ONNX_GRAPH_GENERATED', graph);

    expect(mockCyInstance.elements).toHaveBeenCalled();
    expect(mockCyInstance.add).toHaveBeenCalled();
    expect(mockCyInstance.layout).toHaveBeenCalledWith({
      name: 'breadthfirst',
      directed: true,
      spacingFactor: 1.5,
      fit: true,
      padding: 50
    });
    expect(mockCyInstance.center).toHaveBeenCalled();
  });

  it('should clear graph on null render', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    viz.renderGraph(null);

    expect(mockCyInstance.elements).toHaveBeenCalled();
    expect((viz as any).tooltip.style.display).toBe('none');
  });

  it('should show tooltip on node tap', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    const onCalls = mockCyInstance.on.mock.calls;
    const nodeTapCb = onCalls.find((c) => c[0] === 'tap' && c[1] === 'node')[2];

    const mockEvt = {
      target: {
        data: () => ({
          type: 'operator',
          label: 'Relu',
          dtype: 'float32',
          attributes: { alpha: 1.0 }
        })
      },
      renderedPosition: () => ({ x: 100, y: 200 })
    };

    nodeTapCb(mockEvt);

    const tooltip = (viz as any).tooltip as HTMLElement;
    expect(tooltip.style.display).toBe('block');
    expect(tooltip.innerHTML).toContain('OPERATOR: Relu');
    expect(tooltip.innerHTML).toContain('Type: float32');
    expect(tooltip.innerHTML).toContain('alpha":1');
    expect(tooltip.style.display).toBe('block');
  });

  it('should hide tooltip on background tap', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    const onCalls = mockCyInstance.on.mock.calls;
    const bgTapCb = onCalls.find((c) => c[0] === 'tap' && typeof c[1] === 'function')[1];

    const mockEvt = { target: mockCyInstance };

    bgTapCb(mockEvt);

    const tooltip = (viz as any).tooltip as HTMLElement;
    expect(tooltip.style.display).toBe('none');
  });

  it('should handle node tap with minimal data', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    const onCalls = mockCyInstance.on.mock.calls;
    const nodeTapCb = onCalls.find((c) => c[0] === 'tap' && c[1] === 'node')[2];

    const mockEvt = {
      target: {
        data: () => ({ type: 'operator', label: 'Relu' })
      },
      renderedPosition: { x: 10, y: 20 }
    };

    nodeTapCb(mockEvt);

    const tooltip = (viz as any).tooltip as HTMLElement;
    expect(tooltip.style.display).toBe('block');
    expect(tooltip.innerHTML).not.toContain('Type:');
    expect(tooltip.innerHTML).not.toContain('Attrs:');
  });

  it('should ignore background tap on non-cy targets', () => {
    const viz = new OnnxVisualizer();
    viz.mount(document.body);

    const onCalls = mockCyInstance.on.mock.calls;
    const bgTapCb = onCalls.find((c) => c[0] === 'tap' && typeof c[1] === 'function')[1];

    const mockEvt = { target: 'some-other-target' };

    const tooltip = (viz as any).tooltip as HTMLElement;
    tooltip.style.display = 'block';

    bgTapCb(mockEvt);

    expect(tooltip.style.display).toBe('block');
  });

  it('should ignore renderGraph if cy is null', () => {
    const viz = new OnnxVisualizer();
    expect(() => viz.renderGraph(null)).not.toThrow();
  });
});
