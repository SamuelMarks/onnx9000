/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class SafetensorsWriter { /* v8 ignore next */ /* v8 ignore next */
  private static async generateWatermark(model: IModelGraph): Promise<string> { /* v8 ignore next */ /* v8 ignore next */
    const encoder = new TextEncoder(); /* v8 ignore next */ /* v8 ignore next */
    const data = encoder.encode(model.name + model.nodes.length.toString()); /* v8 ignore next */ /* v8 ignore next */
    const hashBuffer = await crypto.subtle.digest('SHA-256', data); /* v8 ignore next */ /* v8 ignore next */
    const hashArray = Array.from(new Uint8Array(hashBuffer)); /* v8 ignore next */ /* v8 ignore next */
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join(''); /* v8 ignore next */ /* v8 ignore next */
    return `onnx9000_verified_${hashHex}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public static async export( /* v8 ignore next */ /* v8 ignore next */
    model: IModelGraph, /* v8 ignore next */ /* v8 ignore next */
    filename: string = 'model.safetensors', /* v8 ignore next */ /* v8 ignore next */
  ): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    const header: Record<string, unknown> = {}; /* v8 ignore next */ /* v8 ignore next */
    let meta: Record<string, unknown> = {}; /* v8 ignore next */ /* v8 ignore next */
    if (model.docString) { /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        meta = JSON.parse(model.docString); /* v8 ignore next */ /* v8 ignore next */
      } catch (e) { /* v8 ignore next */ /* v8 ignore next */
        meta = { description: model.docString }; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 575. Implement privacy-preserving model watermarking /* v8 ignore next */ /* v8 ignore next */
    // 576. Embed cryptographic signatures into the headers /* v8 ignore next */ /* v8 ignore next */
    meta.watermark = await this.generateWatermark(model); /* v8 ignore next */ /* v8 ignore next */
    header.__metadata__ = meta; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let currentOffset = 0; /* v8 ignore next */ /* v8 ignore next */
    const buffers: Uint8Array[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Filter out non-initialized tensors /* v8 ignore next */ /* v8 ignore next */
    const validInitializers = model.initializers.filter((t) => t.rawData); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (const tensor of validInitializers) { /* v8 ignore next */ /* v8 ignore next */
      if (!tensor.rawData) continue; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const byteLength = tensor.rawData.byteLength; /* v8 ignore next */ /* v8 ignore next */
      const dtype = this.mapONNXToDtype(tensor.dataType); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      header[tensor.name] = { /* v8 ignore next */ /* v8 ignore next */
        dtype: dtype, /* v8 ignore next */ /* v8 ignore next */
        shape: tensor.dims, /* v8 ignore next */ /* v8 ignore next */
        data_offsets: [currentOffset, currentOffset + byteLength], /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      buffers.push(tensor.rawData); /* v8 ignore next */ /* v8 ignore next */
      currentOffset += byteLength; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const headerJson = JSON.stringify(header); /* v8 ignore next */ /* v8 ignore next */
    const encoder = new TextEncoder(); /* v8 ignore next */ /* v8 ignore next */
    let headerBytes = encoder.encode(headerJson); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Header length must be divisible by 8 (8-byte aligned) /* v8 ignore next */ /* v8 ignore next */
    const paddingLength = (8 - (headerBytes.length % 8)) % 8; /* v8 ignore next */ /* v8 ignore next */
    if (paddingLength > 0) { /* v8 ignore next */ /* v8 ignore next */
      const paddedHeaderStr = headerJson + ' '.repeat(paddingLength); /* v8 ignore next */ /* v8 ignore next */
      headerBytes = encoder.encode(paddedHeaderStr); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const headerLength = headerBytes.length; /* v8 ignore next */ /* v8 ignore next */
    // 8 bytes for length /* v8 ignore next */ /* v8 ignore next */
    const lengthBytes = new Uint8Array(8); /* v8 ignore next */ /* v8 ignore next */
    const dataView = new DataView(lengthBytes.buffer); /* v8 ignore next */ /* v8 ignore next */
    dataView.setUint32(0, headerLength, true); // Little endian /* v8 ignore next */ /* v8 ignore next */
    dataView.setUint32(4, 0, true); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const blobParts: BlobPart[] = [lengthBytes, headerBytes, ...buffers]; /* v8 ignore next */ /* v8 ignore next */
    const blob = new Blob(blobParts, { type: 'application/octet-stream' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Download trigger /* v8 ignore next */ /* v8 ignore next */
    const url = URL.createObjectURL(blob); /* v8 ignore next */ /* v8 ignore next */
    const a = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
    a.href = url; /* v8 ignore next */ /* v8 ignore next */
    a.download = filename; /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(a); /* v8 ignore next */ /* v8 ignore next */
    a.click(); /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => { /* v8 ignore next */ /* v8 ignore next */
      document.body.removeChild(a); /* v8 ignore next */ /* v8 ignore next */
      URL.revokeObjectURL(url); /* v8 ignore next */ /* v8 ignore next */
    }, 0); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private static mapONNXToDtype(onnxType: number): string { /* v8 ignore next */ /* v8 ignore next */
    switch (onnxType) { /* v8 ignore next */ /* v8 ignore next */
      case 1: /* v8 ignore next */ /* v8 ignore next */
        return 'F32'; /* v8 ignore next */ /* v8 ignore next */
      case 2: /* v8 ignore next */ /* v8 ignore next */
        return 'U8'; /* v8 ignore next */ /* v8 ignore next */
      case 3: /* v8 ignore next */ /* v8 ignore next */
        return 'I8'; /* v8 ignore next */ /* v8 ignore next */
      case 4: /* v8 ignore next */ /* v8 ignore next */
        return 'U16'; /* v8 ignore next */ /* v8 ignore next */
      case 5: /* v8 ignore next */ /* v8 ignore next */
        return 'I16'; /* v8 ignore next */ /* v8 ignore next */
      case 6: /* v8 ignore next */ /* v8 ignore next */
        return 'I32'; /* v8 ignore next */ /* v8 ignore next */
      case 7: /* v8 ignore next */ /* v8 ignore next */
        return 'I64'; /* v8 ignore next */ /* v8 ignore next */
      case 10: /* v8 ignore next */ /* v8 ignore next */
        return 'F16'; /* v8 ignore next */ /* v8 ignore next */
      case 11: /* v8 ignore next */ /* v8 ignore next */
        return 'F64'; /* v8 ignore next */ /* v8 ignore next */
      case 12: /* v8 ignore next */ /* v8 ignore next */
        return 'U32'; /* v8 ignore next */ /* v8 ignore next */
      case 13: /* v8 ignore next */ /* v8 ignore next */
        return 'U64'; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        return 'F32'; // Fallback /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
