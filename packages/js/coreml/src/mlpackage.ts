import { emitModel } from './emitter.js';
import { Model } from './schema.js';
import { sanitizeMetadataString, sanitizeFilename } from './utils/sanitize.js';
import { validateZipInputData } from './mil/zip_sandbox.js';
import { assertDeterministicBuild } from './mil/deterministic.js';

export class MLPackageBuilder {
  private files = new Map<string, Uint8Array>();

  constructor(
    private model: Model,
    private weightsChunk: Uint8Array = new Uint8Array(0),
    private options: {
      classLabels?: string[];
      outputMappings?: Record<string, string>;
      stateful?: boolean;
      generateSwiftBoilerplate?: boolean;
      computePrecision?: 'Float16' | 'Float32'; // 279. Explicit precision compute
      imageInputs?: Record<
        string,
        { blueBias?: number; greenBias?: number; redBias?: number; imageScale?: number }
      >; // 206, 207. ImageType and scaling properties
      classifierOutputs?: string[]; // 209. DictionaryType
      visionFrameworkDescription?: string; // 211. Custom Vision Framework descriptions
      sequenceInputs?: string[]; // 213. SequenceType mappings for RNNs
      vocabularyFiles?: Record<string, Uint8Array>; // 214. Embed custom vocab files
    } = {},
  ) {
    assertDeterministicBuild(this.model);
  }

  buildDirectoryStructure(): Map<string, Uint8Array> {
    const encoder = new TextEncoder();

    // 1. Generate Manifest.json
    const manifest: ReturnType<typeof JSON.parse> = {
      itemInfoEntries: {
        'Data/com.apple.CoreML/model.mlmodel': {
          author: this.model.description?.metadata?.author
            ? sanitizeMetadataString(this.model.description.metadata.author)
            : 'onnx9000',
          description: this.options.visionFrameworkDescription
            ? sanitizeMetadataString(this.options.visionFrameworkDescription)
            : this.model.description?.metadata?.shortDescription
              ? sanitizeMetadataString(this.model.description.metadata.shortDescription)
              : 'Converted via onnx9000',
        },
      },
      rootModelIdentifier: 'Data/com.apple.CoreML/model.mlmodel',
      version: '1.0.0',
      computeUnits: 'all', // 215. Configure the generated package to utilize `computeUnits = .all` explicitly by default
    };

    // 292. Add support for specialized Apple Vision Pro (visionOS) deployment targets
    manifest['supportedPlatforms'] = {
      visionOS: '1.0',
      iOS: '17.0',
      macOS: '14.0',
      tvOS: '17.0',
      watchOS: '10.0',
    };

    if (this.options.stateful) {
      // 232. Support exporting models with Stateful=True flags
      manifest['isStateful'] = true;
    }

    if (this.options.computePrecision) {
      // 279. Support explicit definition of the "Compute Precision"
      manifest['computePrecision'] = this.options.computePrecision;
    }

    this.files.set('Manifest.json', encoder.encode(JSON.stringify(manifest, null, 2)));

    // 2. Generate FeatureDescriptions.json
    const featureDescriptions = {
      inputs:
        this.model.description?.input.map((i) => {
          // 206, 207. Parse ImageType explicitly vs MultiArray
          const isImage = this.options.imageInputs && this.options.imageInputs[i.name];
          const isSequence = this.options.sequenceInputs?.includes(i.name); // 213
          return {
            name: sanitizeMetadataString(i.name),
            type: isImage ? 'Image' : isSequence ? 'Sequence' : i.type ? 'MultiArray' : 'Unknown',
            imageProperties: isImage ? this.options.imageInputs![i.name] : undefined,
          };
        }) || [],
      outputs:
        this.model.description?.output.map((o) => {
          // 212. Provide configurable outputs mapping
          // 209. Map generic integer array outputs to DictionaryType for classifications
          const mappedName = this.options.outputMappings?.[o.name] || o.name;
          const isClassifier = this.options.classifierOutputs?.includes(o.name);
          return {
            name: sanitizeMetadataString(mappedName),
            type: isClassifier ? 'Dictionary' : o.type ? 'MultiArray' : 'Unknown',
          };
        }) || [],
      states: this.options.stateful ? [] : undefined, // 228. Core ML Stateful operations mapping layout
    };
    this.files.set(
      'Data/com.apple.CoreML/FeatureDescriptions.json',
      encoder.encode(JSON.stringify(featureDescriptions, null, 2)),
    );

    // 210. Generate the standard Core ML Class Labels file
    if (this.options.classLabels && this.options.classLabels.length > 0) {
      this.files.set(
        'Data/com.apple.CoreML/labels.txt',
        encoder.encode(this.options.classLabels.join('\n')),
      );
    }

    // 214. Embed custom vocabulary files
    if (this.options.vocabularyFiles) {
      for (const [filename, data] of Object.entries(this.options.vocabularyFiles)) {
        this.files.set(`Data/com.apple.CoreML/Metadata/${sanitizeFilename(filename)}`, data);
      }
    }

    // 233. Generate appropriate Swift/Objective-C boilerplate text
    if (this.options.generateSwiftBoilerplate) {
      const swiftCode = `
import CoreML

@available(macOS 14.0, iOS 17.0, *)
class ModelRunner {
    let model: MLModel
    var state: MLState?
    
    init(url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: url, configuration: config)
        ${this.options.stateful ? 'self.state = model.newState()' : ''}
    }
    
    func predict(inputs: [String: Any]) throws -> MLDictionaryFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        ${this.options.stateful ? 'return try model.prediction(from: provider, using: state!)' : 'return try model.prediction(from: provider)'}
    }
}
`;
      this.files.set('Boilerplate.swift', encoder.encode(swiftCode.trim()));
    }

    // 3. Emit model.mlmodel (the protobuf)
    const modelBytes = emitModel(this.model);
    this.files.set('Data/com.apple.CoreML/model.mlmodel', modelBytes);

    // 4. Emit weight.bin if present
    if (this.weightsChunk.length > 0) {
      const weightFilename = sanitizeFilename('weight.bin');
      this.files.set(`Data/com.apple.CoreML/weights/${weightFilename}`, this.weightsChunk);
    }

    return this.files;
  }

  // Expects a zip library object like JSZip to be passed in to keep the core zero-dependency
  async createZipArchive(jszipInstance: ReturnType<typeof JSON.parse>): Promise<Uint8Array> {
    const structure = this.buildDirectoryStructure();
    validateZipInputData(structure); // 281, 282. Sandbox execution check

    const zip = new jszipInstance();
    for (const [path, data] of structure.entries()) {
      zip.file(path, data);
    }
    const content = await zip.generateAsync({ type: 'uint8array' });
    return content;
  }
}
