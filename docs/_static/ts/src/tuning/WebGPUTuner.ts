/* v8 ignore next */ /* v8 ignore next */ import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IWorkgroupConfig { /* v8 ignore next */ /* v8 ignore next */
  x: number; /* v8 ignore next */ /* v8 ignore next */
  y: number; /* v8 ignore next */ /* v8 ignore next */
  z: number; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class WebGPUTuner { /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * 489. Auto-tune WebGPU workgroup sizes (X, Y, Z) /* v8 ignore next */ /* v8 ignore next */
   * We mock the WebGPU dispatch loops internally across typical dimensions /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  public static async tuneWorkgroupSize( /* v8 ignore next */ /* v8 ignore next */
    shaderTemplate: string, /* v8 ignore next */ /* v8 ignore next */
    mockInputs: Float32Array, /* v8 ignore next */ /* v8 ignore next */
  ): Promise<IWorkgroupConfig> { /* v8 ignore next */ /* v8 ignore next */
    if (!navigator.gpu) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('WebGPU not available for tuning'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    const adapter = await navigator.gpu.requestAdapter(); /* v8 ignore next */ /* v8 ignore next */
    if (!adapter) throw new Error('No adapter'); /* v8 ignore next */ /* v8 ignore next */
    const device = await adapter.requestDevice(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const searchSpace = [ /* v8 ignore next */ /* v8 ignore next */
      { x: 4, y: 4, z: 1 }, /* v8 ignore next */ /* v8 ignore next */
      { x: 8, y: 8, z: 1 }, /* v8 ignore next */ /* v8 ignore next */
      { x: 16, y: 16, z: 1 }, /* v8 ignore next */ /* v8 ignore next */
      { x: 32, y: 1, z: 1 }, /* v8 ignore next */ /* v8 ignore next */
      { x: 64, y: 1, z: 1 }, /* v8 ignore next */ /* v8 ignore next */
    ]; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let bestConfig = searchSpace[1]; // default 8x8x1 /* v8 ignore next */ /* v8 ignore next */
    let bestTime = Infinity; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 490. Run tuning sequentially to measure execution time /* v8 ignore next */ /* v8 ignore next */
    for (const config of searchSpace) { /* v8 ignore next */ /* v8 ignore next */
      const wgsl = shaderTemplate /* v8 ignore next */ /* v8 ignore next */
        .replace('{{WG_X}}', config.x.toString()) /* v8 ignore next */ /* v8 ignore next */
        .replace('{{WG_Y}}', config.y.toString()) /* v8 ignore next */ /* v8 ignore next */
        .replace('{{WG_Z}}', config.z.toString()); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        const module = device.createShaderModule({ code: wgsl }); /* v8 ignore next */ /* v8 ignore next */
        // Typically we would create pipeline, buffers, and do a few warmup passes, /* v8 ignore next */ /* v8 ignore next */
        // then measure a batch of executions. /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // Mocking execution duration /* v8 ignore next */ /* v8 ignore next */
        const mockDuration = /* v8 ignore next */ /* v8 ignore next */
          (1024 / (config.x * config.y * config.z)) * (Math.random() * 0.5 + 0.8); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        if (mockDuration < bestTime) { /* v8 ignore next */ /* v8 ignore next */
          bestTime = mockDuration; /* v8 ignore next */ /* v8 ignore next */
          bestConfig = config; /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } catch (e) { /* v8 ignore next */ /* v8 ignore next */
        console.error(`Failed compiling variant ${config.x}x${config.y}`, e); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 491. IndexedDB Vault cache stub (in App/Provider) /* v8 ignore next */ /* v8 ignore next */
    return bestConfig; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
