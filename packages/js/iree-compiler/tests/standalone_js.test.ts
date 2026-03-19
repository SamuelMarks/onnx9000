import { describe, it, expect } from 'vitest';
import { Region, Operation, Block } from '../src/ir/core.js';
import { StandaloneJSExporter } from '../src/passes/standalone_js.js';

describe('Standalone JS Exporter', () => {
  it('should emit index.js and html', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(
      new Operation('web.hal.executable.create', [], [], { shader_code: 'enable f16;' }),
    );
    block.pushOperation(new Operation('web.hal.command_buffer.dispatch', [], [], {}));

    const exporter = new StandaloneJSExporter();
    const { js, html } = exporter.emit(region, 'model.bin');

    // JS assertions
    expect(js).toContain('export class ModelRunner');
    expect(js).toContain('createBuffer');
    expect(js).toContain('caches.open');
    expect(js).toContain('await dispatchKernel(device, pipelines[');
    expect(js).toContain('enable f16;');
    expect(js).toContain('device.queue.submit');
    expect(js).toContain('model.bin');

    // HTML assertions
    expect(html).toContain('<input type="file" id="inputImage" />');
    expect(html).toContain("import { ModelRunner } from './index.js';");
    expect(html).toContain('await runner.run(buffer);');
  });
});
