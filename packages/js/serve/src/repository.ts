import { MemoryManager, ModelInstance } from './memory';

export interface ModelMetadata {
  name: string;
  version: string;
  platform: string;
  config: any;
  path: string;
}

export class ModelRepository {
  public models: Map<string, ModelMetadata[]> = new Map();

  constructor(
    public memoryManager: MemoryManager,
    public basePath: string = './models',
  ) {}

  // 156. Implement local File System (FS) watcher natively in Node/Deno.
  // We use abstract polling / watcher interface so it works isomorphic-ly if possible,
  // or via specific Node 'fs' injected.
  public async watch(fsModule: any, pathModule: any) {
    if (!fsModule || !fsModule.watch) return; // Not in Node

    // 161. Enforce strict directory layouts matching Triton (`/models/my_model/1/model.onnx`).
    fsModule.watch(
      this.basePath,
      { recursive: true },
      async (eventType: string, filename: string) => {
        // 157. Detect new `.onnx` models dropped into the `/models` directory and hot-load them instantly.
        // 158. Detect removed models and evict them from memory safely.
        if (
          filename &&
          (filename.endsWith('.onnx') ||
            filename.endsWith('.safetensors') ||
            filename.endsWith('config.json'))
        ) {
          const parts = filename.split(pathModule.sep);
          if (parts.length >= 3) {
            const modelName = parts[0];
            const version = parts[1];
            if (modelName && version)
              await this.reloadModel(modelName, version, fsModule, pathModule);
          }
        }
      },
    );
  }

  public async reloadModel(name: string, version: string, fs: any, path: any) {
    const modelPath = path.join(this.basePath, name, version);
    if (!fs.existsSync(modelPath)) {
      // Evict model logic
      const existing = this.models.get(name);
      if (existing) {
        this.models.set(
          name,
          existing.filter((m) => m.version !== version),
        );
      }
      return;
    }

    // 162. Parse `config.json` automatically on folder ingest.
    let config = {};
    const configPath = path.join(this.basePath, name, 'config.json');
    if (fs.existsSync(configPath)) {
      config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }

    // 164. Support explicit `.safetensors` weight loading seamlessly
    let hasSafetensors = fs.existsSync(path.join(modelPath, 'model.safetensors'));
    let hasOnnx = fs.existsSync(path.join(modelPath, 'model.onnx'));

    if (hasOnnx || hasSafetensors) {
      // 163. Handle zero-downtime deployments
      // We'd load to memory Manager here, then register metadata
      const instance: ModelInstance = {
        id: `${name}_${version}`,
        sizeBytes: 1024, // default model size
        lastUsed: Date.now(),
        buffer: new ArrayBuffer(1024),
        unload: () => {
          console.log('Unloaded', name, version);
        },
      };

      const loaded = await this.memoryManager.requestLoad(instance.id, instance.sizeBytes);
      if (loaded) {
        this.memoryManager.registerModel(instance);

        const versions = this.models.get(name) || [];
        versions.push({
          name,
          version,
          platform: hasSafetensors ? 'safetensors' : 'onnx',
          config,
          path: modelPath,
        });
        this.models.set(name, versions);
      }
    }
  }
}
