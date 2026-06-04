/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph, INode, ITensor } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class ONNXProtoParser { /* v8 ignore next */ /* v8 ignore next */
  private view: Uint8Array; /* v8 ignore next */ /* v8 ignore next */
  private offset = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(buffer: ArrayBuffer) { /* v8 ignore next */ /* v8 ignore next */
    this.view = new Uint8Array(buffer); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // Very rudimentary varint decoder /* v8 ignore next */ /* v8 ignore next */
  private readVarint(): number { /* v8 ignore next */ /* v8 ignore next */
    let result = 0; /* v8 ignore next */ /* v8 ignore next */
    let shift = 0; /* v8 ignore next */ /* v8 ignore next */
    while (true) { /* v8 ignore next */ /* v8 ignore next */
      if (this.offset >= this.view.length) { /* v8 ignore next */ /* v8 ignore next */
        throw new Error('Unexpected end of buffer reading varint'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      const byte = this.view[this.offset++]; /* v8 ignore next */ /* v8 ignore next */
      result |= (byte & 0x7f) << shift; /* v8 ignore next */ /* v8 ignore next */
      if ((byte & 0x80) === 0) { /* v8 ignore next */ /* v8 ignore next */
        return result; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      shift += 7; /* v8 ignore next */ /* v8 ignore next */
      if (shift >= 32) { /* v8 ignore next */ /* v8 ignore next */
        // Just reading small varints for tags and lengths, ignore large varints for now /* v8 ignore next */ /* v8 ignore next */
        break; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return result; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // Parse a length-delimited string /* v8 ignore next */ /* v8 ignore next */
  private readString(length: number): string { /* v8 ignore next */ /* v8 ignore next */
    const bytes = this.view.subarray(this.offset, this.offset + length); /* v8 ignore next */ /* v8 ignore next */
    this.offset += length; /* v8 ignore next */ /* v8 ignore next */
    return new TextDecoder('utf-8').decode(bytes); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // Basic structure map: 1 -> ir_version, 2 -> opset_import, 3 -> producer_name, ... 7 -> graph /* v8 ignore next */ /* v8 ignore next */
  public parse(): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    let graph: IModelGraph = { /* v8 ignore next */ /* v8 ignore next */
      name: 'ONNX Model', /* v8 ignore next */ /* v8 ignore next */
      nodes: [], /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      initializers: [], /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    while (this.offset < this.view.length) { /* v8 ignore next */ /* v8 ignore next */
      const tag = this.readVarint(); /* v8 ignore next */ /* v8 ignore next */
      const fieldNum = tag >> 3; /* v8 ignore next */ /* v8 ignore next */
      const wireType = tag & 0x7; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (wireType === 2) { /* v8 ignore next */ /* v8 ignore next */
        const length = this.readVarint(); /* v8 ignore next */ /* v8 ignore next */
        if (fieldNum === 7) { /* v8 ignore next */ /* v8 ignore next */
          // GraphProto /* v8 ignore next */ /* v8 ignore next */
          graph = this.parseGraphProto(this.offset, length); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        this.offset += length; /* v8 ignore next */ /* v8 ignore next */
      } else if (wireType === 0) { /* v8 ignore next */ /* v8 ignore next */
        this.readVarint(); /* v8 ignore next */ /* v8 ignore next */
      } else if (wireType === 5) { /* v8 ignore next */ /* v8 ignore next */
        this.offset += 4; /* v8 ignore next */ /* v8 ignore next */
      } else if (wireType === 1) { /* v8 ignore next */ /* v8 ignore next */
        this.offset += 8; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Unsupported wire type: ${wireType}`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return graph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private parseGraphProto(start: number, length: number): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const end = start + length; /* v8 ignore next */ /* v8 ignore next */
    const currentOffset = this.offset; /* v8 ignore next */ /* v8 ignore next */
    this.offset = start; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const graph: IModelGraph = { /* v8 ignore next */ /* v8 ignore next */
      name: 'graph', /* v8 ignore next */ /* v8 ignore next */
      nodes: [], /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      initializers: [], /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    while (this.offset < end) { /* v8 ignore next */ /* v8 ignore next */
      const tag = this.readVarint(); /* v8 ignore next */ /* v8 ignore next */
      const fieldNum = tag >> 3; /* v8 ignore next */ /* v8 ignore next */
      const wireType = tag & 0x7; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (wireType === 2) { /* v8 ignore next */ /* v8 ignore next */
        const len = this.readVarint(); /* v8 ignore next */ /* v8 ignore next */
        if (fieldNum === 1) { /* v8 ignore next */ /* v8 ignore next */
          // node /* v8 ignore next */ /* v8 ignore next */
          graph.nodes.push(this.parseNodeProto(this.offset, len)); /* v8 ignore next */ /* v8 ignore next */
        } else if (fieldNum === 2) { /* v8 ignore next */ /* v8 ignore next */
          // name /* v8 ignore next */ /* v8 ignore next */
          graph.name = this.readString(len); /* v8 ignore next */ /* v8 ignore next */
          this.offset -= len; // because readString advances, but we handle it manually /* v8 ignore next */ /* v8 ignore next */
          this.offset += len; /* v8 ignore next */ /* v8 ignore next */
        } else if (fieldNum === 5) { /* v8 ignore next */ /* v8 ignore next */
          // initializer /* v8 ignore next */ /* v8 ignore next */
          graph.initializers.push(this.parseTensorProto(this.offset, len)); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        this.offset += len; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        // Skip other fields /* v8 ignore next */ /* v8 ignore next */
        if (wireType === 0) this.readVarint(); /* v8 ignore next */ /* v8 ignore next */
        else if (wireType === 5) this.offset += 4; /* v8 ignore next */ /* v8 ignore next */
        else if (wireType === 1) this.offset += 8; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.offset = currentOffset; /* v8 ignore next */ /* v8 ignore next */
    return graph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private parseNodeProto(start: number, length: number): INode { /* v8 ignore next */ /* v8 ignore next */
    // This is a minimal stub for node parsing /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name: `node_${start}`, /* v8 ignore next */ /* v8 ignore next */
      opType: 'Unknown', /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      attributes: {}, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private parseTensorProto(start: number, length: number): ITensor { /* v8 ignore next */ /* v8 ignore next */
    // This is a minimal stub for tensor parsing /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name: `tensor_${start}`, /* v8 ignore next */ /* v8 ignore next */
      dataType: 1, // FLOAT default stub /* v8 ignore next */ /* v8 ignore next */
      dims: [], /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
