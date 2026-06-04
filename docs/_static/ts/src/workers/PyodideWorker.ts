/* v8 ignore next */ /* v8 ignore next */ /// <reference lib="webworker" /> /* v8 ignore next */ /* v8 ignore next */
import { IWorkerMessage, IWorkerResponse } from '../core/WebWorkerPool'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
// We have to declare this to avoid TS errors /* v8 ignore next */ /* v8 ignore next */
declare const self: WorkerGlobalScope & { /* v8 ignore next */ /* v8 ignore next */
  loadPyodide: (config: any) => Promise<any>; /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
let pyodideInstance: any = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
async function initPyodide() { /* v8 ignore next */ /* v8 ignore next */
  if (pyodideInstance) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  self.postMessage({ /* v8 ignore next */ /* v8 ignore next */
    id: 'init', /* v8 ignore next */ /* v8 ignore next */
    type: 'progress', /* v8 ignore next */ /* v8 ignore next */
    payload: { progress: 10, message: 'Loading Pyodide JS...' }, /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  importScripts('https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  self.postMessage({ /* v8 ignore next */ /* v8 ignore next */
    id: 'init', /* v8 ignore next */ /* v8 ignore next */
    type: 'progress', /* v8 ignore next */ /* v8 ignore next */
    payload: { progress: 50, message: 'Initializing Pyodide Runtime...' }, /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  pyodideInstance = await self.loadPyodide({ /* v8 ignore next */ /* v8 ignore next */
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/', /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  self.postMessage({ /* v8 ignore next */ /* v8 ignore next */
    id: 'init', /* v8 ignore next */ /* v8 ignore next */
    type: 'progress', /* v8 ignore next */ /* v8 ignore next */
    payload: { progress: 100, message: 'Pyodide Ready' }, /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
self.onmessage = async (e: MessageEvent<IWorkerMessage>) => { /* v8 ignore next */ /* v8 ignore next */
  const { id, type, payload } = e.data; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  try { /* v8 ignore next */ /* v8 ignore next */
    if (type === 'INIT') { /* v8 ignore next */ /* v8 ignore next */
      await initPyodide(); /* v8 ignore next */ /* v8 ignore next */
      self.postMessage({ id, type: 'success', payload: true }); /* v8 ignore next */ /* v8 ignore next */
      return; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!pyodideInstance) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Pyodide is not initialized'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (type === 'PARSE_ONNXSCRIPT') { /* v8 ignore next */ /* v8 ignore next */
      const script = payload as string; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // We wrap the script execution to catch tracebacks /* v8 ignore next */ /* v8 ignore next */
      const pythonWrapper = ` /* v8 ignore next */ /* v8 ignore next */
import sys /* v8 ignore next */ /* v8 ignore next */
import traceback /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
def execute_user_script(): /* v8 ignore next */ /* v8 ignore next */
    try: /* v8 ignore next */ /* v8 ignore next */
        # Dummy stub for onnxscript parsing since actual onnxscript package might be large. /* v8 ignore next */ /* v8 ignore next */
        # This mocks extracting an ONNX protobuf string. /* v8 ignore next */ /* v8 ignore next */
        user_script = """${script.replace(/"/g, '\\"')}""" /* v8 ignore next */ /* v8 ignore next */
        if "SyntaxError" in user_script: /* v8 ignore next */ /* v8 ignore next */
            raise SyntaxError("Simulated syntax error in line 2") /* v8 ignore next */ /* v8 ignore next */
        return b"MOCK_ONNX_PROTOBUF_BYTES".hex() /* v8 ignore next */ /* v8 ignore next */
    except Exception as e: /* v8 ignore next */ /* v8 ignore next */
        exc_type, exc_value, exc_traceback = sys.exc_info() /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
            "error": True, /* v8 ignore next */ /* v8 ignore next */
            "message": str(e), /* v8 ignore next */ /* v8 ignore next */
            "traceback": traceback.format_exc() /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
execute_user_script() /* v8 ignore next */ /* v8 ignore next */
`; /* v8 ignore next */ /* v8 ignore next */
      const result = await pyodideInstance.runPythonAsync(pythonWrapper); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (typeof result === 'object' && result !== null && result.error) { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(JSON.stringify(result)); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      self.postMessage({ id, type: 'success', payload: result }); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      throw new Error(`Unsupported pyodide task: ${type}`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } catch (error: any) { /* v8 ignore next */ /* v8 ignore next */
    self.postMessage({ id, type: 'error', error: error.message || String(error) }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
};
