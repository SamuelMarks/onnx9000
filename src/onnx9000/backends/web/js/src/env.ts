// Step 345: Env singleton mapping to OrtEnv
/** Implementation details and semantic operations. */
export interface WasmOptions {
    numThreads?: number;
    simd?: boolean;
    proxy?: boolean;
    wasmPaths?: Record<string, string>;
}

/** Implementation details and semantic operations. */

export interface WebGpuOptions {
    profilingMode?: "default" | "time" | "memory";
}

/** Implementation details and semantic operations. */

export class Env {
    public logLevel: "verbose" | "info" | "warning" | "error" | "fatal" = "warning";
    public wasm: WasmOptions = { numThreads: 1, simd: true };
    public webgpu: WebGpuOptions = {};
    
    private static _instance: Env | null = null;
    
    public static get instance(): Env {
        /** Implementation details and semantic operations. */
        if(!Env._instance) {
            Env._instance = new Env();
        }
        return Env._instance;
    }
}

export const env = Env.instance;
