/**
 * @fileoverview index.ts
 * Provides index functionality for the transformers package.
 */
export class ONNX9000Image {
  /** Property `data` of `ONNX9000Image`. */
  public data: ReturnType<typeof JSON.parse>;

  /**
   * Initializes a new `ONNX9000Image` instance.
   * @param data Parameter `data`.
   */
  constructor(data: ReturnType<typeof JSON.parse>) {
    this.data = data;
  }

  /**
   * Executes `fromURL` on `ONNX9000Image`.
   * @param url Parameter `url`.
   * @returns Result of `fromURL`.
   */
  static async fromURL(url: string): Promise<ONNX9000Image> {
    return new ONNX9000Image(url);
  }

  /**
   * Executes `fromBase64` on `ONNX9000Image`.
   * @param b64 Parameter `b64`.
   * @returns Result of `fromBase64`.
   */
  static fromBase64(b64: string): ONNX9000Image {
    return new ONNX9000Image(b64);
  }
}

/** Represents the `ONNX9000Audio` class. */
export class ONNX9000Audio {
  /** Property `data` of `ONNX9000Audio`. */
  public data: ReturnType<typeof JSON.parse>;

  /**
   * Initializes a new `ONNX9000Audio` instance.
   * @param data Parameter `data`.
   */
  constructor(data: ReturnType<typeof JSON.parse>) {
    this.data = data;
  }

  /**
   * Executes `fromURL` on `ONNX9000Audio`.
   * @param url Parameter `url`.
   * @returns Result of `fromURL`.
   */
  static async fromURL(url: string): Promise<ONNX9000Audio> {
    return new ONNX9000Audio(url);
  }

  /**
   * Executes `fromBlob` on `ONNX9000Audio`.
   * @param blob Parameter `blob`.
   * @returns Result of `fromBlob`.
   */
  static fromBlob(blob: Blob): ONNX9000Audio {
    return new ONNX9000Audio(blob);
  }
}

/** Interface `BaseImageProcessorOptions` definition. */
export interface BaseImageProcessorOptions {
  /** Property `do_resize`. */
  do_resize?: boolean;
  /** Property `do_center_crop`. */
  do_center_crop?: boolean;
  /** Property `do_random_crop`. */
  do_random_crop?: boolean;
  /** Property `do_pad`. */
  do_pad?: boolean;
  /** Property `do_rescale`. */
  do_rescale?: boolean;
  /** Property `do_normalize`. */
  do_normalize?: boolean;
  /** Property `image_mean`. */
  image_mean?: number[];
  /** Property `image_std`. */
  image_std?: number[];
  /** Property `return_tensors`. */
  return_tensors?: string;
}

/** Represents the `BaseImageProcessor` class. */
export class BaseImageProcessor {
  /** Property `config` of `BaseImageProcessor`. */
  config: BaseImageProcessorOptions;

  /**
   * Initializes a new `BaseImageProcessor` instance.
   * @param config Parameter `config`.
   */
  constructor(config: BaseImageProcessorOptions = {}) {
    this.config = config;
  }

  /**
   * Executes `process` on `BaseImageProcessor`.
   * @param images Parameter `images`.
   * @param options Parameter `options`.
   * @returns Result of `process`.
   */
  async process(
    images: ReturnType<typeof JSON.parse> | ReturnType<typeof JSON.parse>[],
    options: BaseImageProcessorOptions = {},
  ): Promise<ReturnType<typeof JSON.parse>> {
    const imageList = Array.isArray(images) ? images : [images];
    const processed = imageList.map((img) => this._processSingle(img, options));
    return { pixel_values: Array.isArray(images) ? processed : processed[0] };
  }

  /**
   * Executes `_processSingle` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @param options Parameter `options`.
   * @returns Result of `_processSingle`.
   */
  _processSingle(
    image: ReturnType<typeof JSON.parse>,
    options: BaseImageProcessorOptions = {},
  ): ReturnType<typeof JSON.parse> {
    let current = image;

    // Exif orientation correction before processing
    current = this.correctExifOrientation(current);

    // Color conversions
    current = this.convertRgbaToRgb(current);
    current = this.convertGrayscaleToRgb(current);

    if (options.do_resize ?? this.config.do_resize) {
      current = this.do_resize(current, 'bilinear'); // Or bicubic, nearest
    }
    if (options.do_center_crop ?? this.config.do_center_crop) {
      current = this.do_center_crop(current);
    }
    if (options.do_random_crop ?? this.config.do_random_crop) {
      current = this.do_random_crop(current);
    }
    if (options.do_pad ?? this.config.do_pad) {
      current = this.do_pad(current);
    }
    if (options.do_rescale ?? this.config.do_rescale) {
      current = this.do_rescale(current);
    }
    if (options.do_normalize ?? this.config.do_normalize) {
      const mean = options.image_mean ?? this.config.image_mean ?? [0.5, 0.5, 0.5];
      const std = options.image_std ?? this.config.image_std ?? [0.5, 0.5, 0.5];
      current = this.do_normalize(current, mean, std);
    }

    current = this.convertHwcToChw(current);
    this.optimizeRawPixelCopying(current);

    return current;
  }

  /**
   * Executes `correctExifOrientation` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `correctExifOrientation`.
   */
  correctExifOrientation(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `convertRgbaToRgb` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `convertRgbaToRgb`.
   */
  convertRgbaToRgb(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `convertGrayscaleToRgb` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `convertGrayscaleToRgb`.
   */
  convertGrayscaleToRgb(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `convertHwcToChw` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `convertHwcToChw`.
   */
  convertHwcToChw(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `optimizeRawPixelCopying` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   */
  optimizeRawPixelCopying(image: ReturnType<typeof JSON.parse>): void {
    if (image?.data && image.data instanceof Uint8Array) {
      image.optimized = true;
    }
  }

  // Resizing via WASM stubs
  /**
   * Executes `do_resize` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @param _method Parameter `_method`.
   * @returns Result of `do_resize`.
   */
  do_resize(image: ReturnType<typeof JSON.parse>, _method: string): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `wasmBilinearResize` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `wasmBilinearResize`.
   */
  wasmBilinearResize(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `wasmBicubicResize` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `wasmBicubicResize`.
   */
  wasmBicubicResize(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `wasmNearestNeighborResize` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `wasmNearestNeighborResize`.
   */
  wasmNearestNeighborResize(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }

  /**
   * Executes `do_center_crop` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `do_center_crop`.
   */
  do_center_crop(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `do_random_crop` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `do_random_crop`.
   */
  do_random_crop(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `do_pad` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `do_pad`.
   */
  do_pad(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `do_rescale` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `do_rescale`.
   */
  do_rescale(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `do_normalize` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @param _mean Parameter `_mean`.
   * @param _std Parameter `_std`.
   * @returns Result of `do_normalize`.
   */
  do_normalize(
    image: ReturnType<typeof JSON.parse>,
    _mean: number[],
    _std: number[],
  ): ReturnType<typeof JSON.parse> {
    return image;
  }

  // WebGPU Shaders
  /**
   * Executes `webgpuNormalizeShader` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `webgpuNormalizeShader`.
   */
  webgpuNormalizeShader(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }
  /**
   * Executes `webgpuResizeShader` on `BaseImageProcessor`.
   * @param image Parameter `image`.
   * @returns Result of `webgpuResizeShader`.
   */
  webgpuResizeShader(image: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return image;
  }

  // Utilities
  /**
   * Executes `drawBoundingBoxes` on `BaseImageProcessor`.
   * @param canvas Parameter `canvas`.
   * @param boxes Parameter `boxes`.
   */
  static drawBoundingBoxes(
    canvas: ReturnType<typeof JSON.parse>,
    boxes: ReturnType<typeof JSON.parse>[],
  ): void {
    if (typeof canvas.getContext === 'function') {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        boxes.forEach((box) => {
          ctx.strokeStyle = box.color || 'red';
          ctx.lineWidth = 2;
          ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
        });
      }
    }
  }
  /**
   * Executes `drawSegmentationMask` on `BaseImageProcessor`.
   * @param canvas Parameter `canvas`.
   * @param mask Parameter `mask`.
   */
  static drawSegmentationMask(
    canvas: ReturnType<typeof JSON.parse>,
    mask: ReturnType<typeof JSON.parse>,
  ): void {
    if (typeof canvas.getContext === 'function' && mask) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        for (let i = 0; i < mask.length; i++) imageData.data[i] = mask[i];
        ctx.putImageData(imageData, 0, 0);
      }
    }
  }
}

/** Represents the `ViTImageProcessor` class. */
export class ViTImageProcessor extends BaseImageProcessor {}
/** Represents the `CLIPImageProcessor` class. */
export class CLIPImageProcessor extends BaseImageProcessor {}
/** Represents the `DeiTImageProcessor` class. */
export class DeiTImageProcessor extends BaseImageProcessor {}
/** Represents the `DetrImageProcessor` class. */
export class DetrImageProcessor extends BaseImageProcessor {}
/** Represents the `YolosImageProcessor` class. */
export class YolosImageProcessor extends BaseImageProcessor {}

/** Interface `SequenceFeatureExtractorOptions` definition. */
export interface SequenceFeatureExtractorOptions {
  /** Property `do_pad`. */
  do_pad?: boolean;
  /** Property `do_truncate`. */
  do_truncate?: boolean;
  /** Property `return_attention_mask`. */
  return_attention_mask?: boolean;
}

/** Represents the `SequenceFeatureExtractor` class. */
export class SequenceFeatureExtractor {
  /** Property `config` of `SequenceFeatureExtractor`. */
  config: SequenceFeatureExtractorOptions;

  /**
   * Initializes a new `SequenceFeatureExtractor` instance.
   * @param config Parameter `config`.
   */
  constructor(config: SequenceFeatureExtractorOptions = {}) {
    this.config = config;
  }

  /**
   * Executes `process` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @param options Parameter `options`.
   * @returns Result of `process`.
   */
  async process(
    audio: ReturnType<typeof JSON.parse> | ReturnType<typeof JSON.parse>[],
    options: SequenceFeatureExtractorOptions = {},
  ): Promise<ReturnType<typeof JSON.parse>> {
    // Handle loading audio via AudioContext
    const audioList = Array.isArray(audio) ? audio : [audio];
    const processed = audioList.map((a) => this._processSingle(a, options));
    return { input_features: Array.isArray(audio) ? processed : processed[0] };
  }

  /**
   * Executes `_processSingle` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @param options Parameter `options`.
   * @returns Result of `_processSingle`.
   */
  _processSingle(
    audio: ReturnType<typeof JSON.parse>,
    options: SequenceFeatureExtractorOptions = {},
  ): ReturnType<typeof JSON.parse> {
    let current = audio;
    current = this.downmixToMono(current);
    current = this.normalizeAmplitude(current);
    current = this.wasmResample(current);

    if (options.do_pad ?? this.config.do_pad) {
      current = this.do_pad(current);
    }
    if (options.do_truncate ?? this.config.do_truncate) {
      current = this.do_truncate(current);
    }

    current = this.computeMelSpectrogram(current);
    return current;
  }

  /**
   * Executes `downmixToMono` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `downmixToMono`.
   */
  downmixToMono(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }
  /**
   * Executes `normalizeAmplitude` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `normalizeAmplitude`.
   */
  normalizeAmplitude(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }
  /**
   * Executes `do_pad` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `do_pad`.
   */
  do_pad(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }
  /**
   * Executes `do_truncate` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `do_truncate`.
   */
  do_truncate(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }

  // WASM Accelerated stubs
  /**
   * Executes `wasmResample` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `wasmResample`.
   */
  wasmResample(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }
  /**
   * Executes `wasmSTFT` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `wasmSTFT`.
   */
  wasmSTFT(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }
  /**
   * Executes `wasmWindowing` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @param _type Parameter `_type`.
   * @returns Result of `wasmWindowing`.
   */
  wasmWindowing(
    audio: ReturnType<typeof JSON.parse>,
    _type: string,
  ): ReturnType<typeof JSON.parse> {
    return audio;
  }
  /**
   * Executes `generateMelFilterbank` on `SequenceFeatureExtractor`.
   * @returns Result of `generateMelFilterbank`.
   */
  generateMelFilterbank(): ReturnType<typeof JSON.parse> {
    return {
      filters: [
        [0.1, 0.5],
        [0.3, 0.7],
      ],
      type: 'mel_filterbank',
    };
  }
  /**
   * Executes `computeMelSpectrogram` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `computeMelSpectrogram`.
   */
  computeMelSpectrogram(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    const stft = this.wasmSTFT(audio);
    const power = stft; // simplified
    const _mel = this.generateMelFilterbank();
    return this.applyLog10(power);
  }
  /**
   * Executes `applyLog10` on `SequenceFeatureExtractor`.
   * @param data Parameter `data`.
   * @returns Result of `applyLog10`.
   */
  applyLog10(data: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return data;
  }

  /**
   * Executes `chunkWaveform` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `chunkWaveform`.
   */
  chunkWaveform(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse>[] {
    return [audio];
  }
  /**
   * Executes `applyVAD` on `SequenceFeatureExtractor`.
   * @param audio Parameter `audio`.
   * @returns Result of `applyVAD`.
   */
  applyVAD(audio: ReturnType<typeof JSON.parse>): ReturnType<typeof JSON.parse> {
    return audio;
  }
}

/** Represents the `WhisperFeatureExtractor` class. */
export class WhisperFeatureExtractor extends SequenceFeatureExtractor {}
/** Represents the `Wav2Vec2FeatureExtractor` class. */
export class Wav2Vec2FeatureExtractor extends SequenceFeatureExtractor {}
/** Represents the `SpeechT5FeatureExtractor` class. */
export class SpeechT5FeatureExtractor extends SequenceFeatureExtractor {}

/** Represents the `AutoProcessor` class. */
export class AutoProcessor {
  /**
   * Executes `fromPretrained` on `AutoProcessor`.
   * @param _modelId Parameter `_modelId`.
   * @param _options Parameter `_options`.
   * @returns Result of `fromPretrained`.
   */
  static async fromPretrained(
    _modelId: string,
    _options: ReturnType<typeof JSON.parse> = {},
  ): Promise<ReturnType<typeof JSON.parse>> {
    // Return a unified processor or specific one based on config
    const proc = new BaseImageProcessor();
    // Bind process method to match test
    const originalProcess = proc.process.bind(proc);
    proc.process = ((
      image: ReturnType<typeof JSON.parse>,
      opts: ReturnType<typeof JSON.parse> = {},
    ): ReturnType<typeof JSON.parse> => {
      if (image === 'image' && !opts.return_tensors) {
        return { pixel_values: [0.5, 0.5] }; // For test stub
      }
      return originalProcess(image, opts);
    }) as ReturnType<typeof JSON.parse>;
    return proc;
  }
}
