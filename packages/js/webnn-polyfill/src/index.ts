import {
  MLContextOptions,
  MLOperandDescriptor,
  MLComputeResult,
  MLOperandDataType,
  MLOpSupportLimits,
} from './interfaces.js';
import { PolyfillMLContext } from './context.js';
import { PolyfillMLGraphBuilder } from './builder.js';

export * from './interfaces.js';
export * from './context.js';
export * from './builder.js';
export * from './operand.js';
export * from './graph.js';
export * from './tensor.js';

export class PolyfillML {
  async createContext(options?: MLContextOptions): Promise<PolyfillMLContext> {
    return new PolyfillMLContext(options);
  }
}

if (typeof window !== 'undefined') {
  if (!(window.navigator as ReturnType<typeof JSON.parse>).ml) {
    (window.navigator as ReturnType<typeof JSON.parse>).ml = new PolyfillML();
  }
  (window as ReturnType<typeof JSON.parse>).MLContext = PolyfillMLContext;
  (window as ReturnType<typeof JSON.parse>).MLGraphBuilder = PolyfillMLGraphBuilder;
}
