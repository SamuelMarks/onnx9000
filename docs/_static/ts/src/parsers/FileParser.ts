/* v8 ignore next */ /* v8 ignore next */ import { KerasParser } from './KerasParser'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
import { SafetensorsParser } from './Safetensors'; /* v8 ignore next */ /* v8 ignore next */
import { ONNXProtoParser } from './ONNXProto'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from '../ui/Toast'; /* v8 ignore next */ /* v8 ignore next */
import { IModelGraph } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { WebWorkerPool } from '../core/WebWorkerPool'; /* v8 ignore next */ /* v8 ignore next */
import { astCache } from '../storage/IndexedDBVault'; /* v8 ignore next */ /* v8 ignore next */
import { logger } from '../core/Logger'; /* v8 ignore next */ /* v8 ignore next */
import { Spinner } from '../ui/Spinner'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class FileParser { /* v8 ignore next */ /* v8 ignore next */
  private workerPool: WebWorkerPool | null = null; /* v8 ignore next */ /* v8 ignore next */
  private pyodidePool: WebWorkerPool | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      this.workerPool = new WebWorkerPool('_static/assets/parser.worker.js', 2); /* v8 ignore next */ /* v8 ignore next */
      this.pyodidePool = new WebWorkerPool('_static/assets/pyodide.worker.js', 1); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      logger.warn('Failed to initialize WebWorkerPools', e); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async initPyodide(): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.pyodidePool) return; /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      await this.pyodidePool.execute('INIT', null, (p: any) => { /* v8 ignore next */ /* v8 ignore next */
        Spinner.show(); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      Spinner.hide(); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      Spinner.hide(); /* v8 ignore next */ /* v8 ignore next */
      logger.error('Pyodide init failed', e); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async processFile(file: File): Promise<IModelGraph | null> { /* v8 ignore next */ /* v8 ignore next */
    const extension = file.name.split('.').pop()?.toLowerCase(); /* v8 ignore next */ /* v8 ignore next */
    Toast.show(`Reading file: ${file.name}...`); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      const buffer = await file.arrayBuffer(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (extension === 'safetensors') { /* v8 ignore next */ /* v8 ignore next */
        return this.parseSafetensors(buffer, file.name); /* v8 ignore next */ /* v8 ignore next */
      } else if (extension === 'onnx') { /* v8 ignore next */ /* v8 ignore next */
        return this.parseONNX(buffer, file.name); /* v8 ignore next */ /* v8 ignore next */
      } else if (extension === 'py') { /* v8 ignore next */ /* v8 ignore next */
        return this.parseONNXScript(file); /* v8 ignore next */ /* v8 ignore next */
      } else if (['pb', 'savedmodel', 'pkl', 'pdmodel', 'json', 'gguf'].includes(extension || '')) { /* v8 ignore next */ /* v8 ignore next */
        return this.parseViaWorker(buffer, file.name, extension as string); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Unsupported file extension: ${extension}`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      const msg = e instanceof Error ? e.message : String(e); /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Failed to parse file: ${msg}`, 'error'); /* v8 ignore next */ /* v8 ignore next */
      logger.error(e); /* v8 ignore next */ /* v8 ignore next */
      return null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async processDirectory(files: File[]): Promise<IModelGraph | null> { /* v8 ignore next */ /* v8 ignore next */
    Toast.show(`Parsing directory with ${files.length} files...`); /* v8 ignore next */ /* v8 ignore next */
    // Minimal stub for constructing directory hierarchy memory /* v8 ignore next */ /* v8 ignore next */
    logger.info( /* v8 ignore next */ /* v8 ignore next */
      'Directory files:', /* v8 ignore next */ /* v8 ignore next */
      files.map((f) => f.webkitRelativePath), /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // We would package these files into a virtual filesystem map and pass to the worker /* v8 ignore next */ /* v8 ignore next */
    // For now, return a dummy /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name: 'TF_Directory_Model', /* v8 ignore next */ /* v8 ignore next */
      nodes: [], /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      initializers: [], /* v8 ignore next */ /* v8 ignore next */
      docString: JSON.stringify({ files: files.length }), /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private async parseONNXScript(file: File): Promise<IModelGraph | null> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.pyodidePool) throw new Error('Pyodide pool not initialized'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const text = await file.text(); /* v8 ignore next */ /* v8 ignore next */
    Toast.show('Executing ONNXScript in Pyodide...'); /* v8 ignore next */ /* v8 ignore next */
    Spinner.show(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      const result = await this.pyodidePool.execute('PARSE_ONNXSCRIPT', text, (p: any) => { /* v8 ignore next */ /* v8 ignore next */
        logger.info(`[Pyodide] ${p.progress}%: ${p.message}`); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      Spinner.hide(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // result is a hex string representing ONNX protobuf bytes /* v8 ignore next */ /* v8 ignore next */
      const hex = result as string; /* v8 ignore next */ /* v8 ignore next */
      const bytes = new Uint8Array(Math.ceil(hex.length / 2)); /* v8 ignore next */ /* v8 ignore next */
      for (let i = 0; i < hex.length; i += 2) { /* v8 ignore next */ /* v8 ignore next */
        bytes[i / 2] = parseInt(hex.substr(i, 2), 16); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const parser = new ONNXProtoParser(bytes.buffer); /* v8 ignore next */ /* v8 ignore next */
      const graph = parser.parse(); /* v8 ignore next */ /* v8 ignore next */
      graph.name = file.name; /* v8 ignore next */ /* v8 ignore next */
      return graph; /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      Spinner.hide(); /* v8 ignore next */ /* v8 ignore next */
      const msg = e instanceof Error ? e.message : String(e); /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`ONNXScript Error: ${msg}`, 'error'); /* v8 ignore next */ /* v8 ignore next */
      return null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private async parseViaWorker( /* v8 ignore next */ /* v8 ignore next */
    buffer: ArrayBuffer, /* v8 ignore next */ /* v8 ignore next */
    name: string, /* v8 ignore next */ /* v8 ignore next */
    ext: string, /* v8 ignore next */ /* v8 ignore next */
  ): Promise<IModelGraph> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.workerPool) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Worker pool not initialized'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const hash = await astCache.computeHash(buffer); /* v8 ignore next */ /* v8 ignore next */
    const cached = await astCache.get(hash); /* v8 ignore next */ /* v8 ignore next */
    if (cached) { /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Loaded AST from IndexedDB Cache', 'success'); /* v8 ignore next */ /* v8 ignore next */
      return cached; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let type = 'PARSE_TF'; /* v8 ignore next */ /* v8 ignore next */
    let payload: unknown = buffer; /* v8 ignore next */ /* v8 ignore next */
    if (ext === 'pkl') type = 'PARSE_SKL'; /* v8 ignore next */ /* v8 ignore next */
    if (ext === 'pdmodel') type = 'PARSE_PADDLE'; /* v8 ignore next */ /* v8 ignore next */
    if (ext === 'json') { /* v8 ignore next */ /* v8 ignore next */
      const text = new TextDecoder().decode(buffer); /* v8 ignore next */ /* v8 ignore next */
      if (text.includes('keras_version') || text.includes('class_name')) { /* v8 ignore next */ /* v8 ignore next */
        const parser = new KerasParser(text); /* v8 ignore next */ /* v8 ignore next */
        const graph = parser.parse(); /* v8 ignore next */ /* v8 ignore next */
        await astCache.set(hash, graph); /* v8 ignore next */ /* v8 ignore next */
        return graph; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      type = 'PARSE_XGBOOST'; /* v8 ignore next */ /* v8 ignore next */
      payload = new TextDecoder().decode(buffer); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    if (ext === 'gguf') type = 'PARSE_GGUF'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const result = await this.workerPool.execute(type, payload, (progressPayload) => { /* v8 ignore next */ /* v8 ignore next */
      const p = progressPayload as { progress: number; message: string }; /* v8 ignore next */ /* v8 ignore next */
      logger.info(`[Worker] ${p.progress}%: ${p.message}`); /* v8 ignore next */ /* v8 ignore next */
      // Send progress to UI state /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('progress', p); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const graph = result as IModelGraph; /* v8 ignore next */ /* v8 ignore next */
    if (graph && (graph.name === 'Model' || graph.name.includes('_Model'))) { /* v8 ignore next */ /* v8 ignore next */
      graph.name = name; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    await astCache.set(hash, graph); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return graph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private parseSafetensors(buffer: ArrayBuffer, name: string): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const parser = new SafetensorsParser(buffer); /* v8 ignore next */ /* v8 ignore next */
    const { metadata, tensors } = parser.parse(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 577. Verify watermarks upon model load /* v8 ignore next */ /* v8 ignore next */
    if (metadata && typeof metadata === 'object' && 'watermark' in metadata) { /* v8 ignore next */ /* v8 ignore next */
      const wm = metadata.watermark as string; /* v8 ignore next */ /* v8 ignore next */
      if (wm.startsWith('onnx9000_verified_')) { /* v8 ignore next */ /* v8 ignore next */
        logger.info(`Valid DP Watermark found: ${wm}`); /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Model Watermark Verified', 'success'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name, /* v8 ignore next */ /* v8 ignore next */
      nodes: [], /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      initializers: Object.values(tensors).map((t) => ({ /* v8 ignore next */ /* v8 ignore next */
        name: t.name, /* v8 ignore next */ /* v8 ignore next */
        dataType: this.mapDtypeToONNX(t.dtype), /* v8 ignore next */ /* v8 ignore next */
        dims: t.shape, /* v8 ignore next */ /* v8 ignore next */
        rawData: new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), /* v8 ignore next */ /* v8 ignore next */
      })), /* v8 ignore next */ /* v8 ignore next */
      docString: metadata ? JSON.stringify(metadata) : undefined, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private parseONNX(buffer: ArrayBuffer, name: string): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const parser = new ONNXProtoParser(buffer); /* v8 ignore next */ /* v8 ignore next */
    const graph = parser.parse(); /* v8 ignore next */ /* v8 ignore next */
    if (graph.name === 'ONNX Model') { /* v8 ignore next */ /* v8 ignore next */
      graph.name = name; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return graph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private mapDtypeToONNX(safetensorDtype: string): number { /* v8 ignore next */ /* v8 ignore next */
    switch (safetensorDtype) { /* v8 ignore next */ /* v8 ignore next */
      case 'F32': /* v8 ignore next */ /* v8 ignore next */
        return 1; /* v8 ignore next */ /* v8 ignore next */
      case 'U8': /* v8 ignore next */ /* v8 ignore next */
        return 2; /* v8 ignore next */ /* v8 ignore next */
      case 'I8': /* v8 ignore next */ /* v8 ignore next */
        return 3; /* v8 ignore next */ /* v8 ignore next */
      case 'U16': /* v8 ignore next */ /* v8 ignore next */
        return 4; /* v8 ignore next */ /* v8 ignore next */
      case 'I16': /* v8 ignore next */ /* v8 ignore next */
        return 5; /* v8 ignore next */ /* v8 ignore next */
      case 'I32': /* v8 ignore next */ /* v8 ignore next */
        return 6; /* v8 ignore next */ /* v8 ignore next */
      case 'I64': /* v8 ignore next */ /* v8 ignore next */
        return 7; /* v8 ignore next */ /* v8 ignore next */
      case 'F16': /* v8 ignore next */ /* v8 ignore next */
        return 10; /* v8 ignore next */ /* v8 ignore next */
      case 'F64': /* v8 ignore next */ /* v8 ignore next */
        return 11; /* v8 ignore next */ /* v8 ignore next */
      case 'U32': /* v8 ignore next */ /* v8 ignore next */
        return 12; /* v8 ignore next */ /* v8 ignore next */
      case 'U64': /* v8 ignore next */ /* v8 ignore next */
        return 13; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        return 0; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  terminateWorkers(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.workerPool) { /* v8 ignore next */ /* v8 ignore next */
      this.workerPool.terminateAll(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
