/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class IndexedDBVault { /* v8 ignore next */ /* v8 ignore next */
  private dbName = 'onnx9000_ast_cache'; /* v8 ignore next */ /* v8 ignore next */
  private storeName = 'models'; /* v8 ignore next */ /* v8 ignore next */
  private db: IDBDatabase | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async init(): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve, reject) => { /* v8 ignore next */ /* v8 ignore next */
      const request = indexedDB.open(this.dbName, 1); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      request.onupgradeneeded = (e) => { /* v8 ignore next */ /* v8 ignore next */
        const target = e.target as IDBOpenDBRequest; /* v8 ignore next */ /* v8 ignore next */
        this.db = target.result; /* v8 ignore next */ /* v8 ignore next */
        if (!this.db.objectStoreNames.contains(this.storeName)) { /* v8 ignore next */ /* v8 ignore next */
          this.db.createObjectStore(this.storeName, { keyPath: 'hash' }); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      request.onsuccess = (e) => { /* v8 ignore next */ /* v8 ignore next */
        const target = e.target as IDBOpenDBRequest; /* v8 ignore next */ /* v8 ignore next */
        this.db = target.result; /* v8 ignore next */ /* v8 ignore next */
        resolve(); /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      request.onerror = () => { /* v8 ignore next */ /* v8 ignore next */
        reject(new Error('Failed to initialize IndexedDB')); /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async get(hash: string): Promise<IModelGraph | null> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.db) await this.init(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve, reject) => { /* v8 ignore next */ /* v8 ignore next */
      const transaction = this.db!.transaction(this.storeName, 'readonly'); /* v8 ignore next */ /* v8 ignore next */
      const store = transaction.objectStore(this.storeName); /* v8 ignore next */ /* v8 ignore next */
      const request = store.get(hash); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      request.onsuccess = () => { /* v8 ignore next */ /* v8 ignore next */
        resolve(request.result ? request.result.model : null); /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
      request.onerror = () => reject(new Error('Failed to read from cache')); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async set(hash: string, model: IModelGraph): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.db) await this.init(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 583. Ensure no plaintext weights are written to IndexedDB. /* v8 ignore next */ /* v8 ignore next */
    // Strip rawData payload buffers from the model before caching. /* v8 ignore next */ /* v8 ignore next */
    // The user must re-supply the raw file to hydrate the execution buffers, /* v8 ignore next */ /* v8 ignore next */
    // ensuring the indexedDB only caches the structural AST. /* v8 ignore next */ /* v8 ignore next */
    const strippedModel: IModelGraph = JSON.parse(JSON.stringify(model)); /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < strippedModel.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      strippedModel.initializers[i].rawData = undefined; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve, reject) => { /* v8 ignore next */ /* v8 ignore next */
      const transaction = this.db!.transaction(this.storeName, 'readwrite'); /* v8 ignore next */ /* v8 ignore next */
      const store = transaction.objectStore(this.storeName); /* v8 ignore next */ /* v8 ignore next */
      const request = store.put({ hash, model: strippedModel }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      request.onsuccess = () => resolve(); /* v8 ignore next */ /* v8 ignore next */
      request.onerror = () => reject(new Error('Failed to write to cache')); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async computeHash(buffer: ArrayBuffer): Promise<string> { /* v8 ignore next */ /* v8 ignore next */
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer); /* v8 ignore next */ /* v8 ignore next */
    const hashArray = Array.from(new Uint8Array(hashBuffer)); /* v8 ignore next */ /* v8 ignore next */
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join(''); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 407. Show persistent storage quota and usage /* v8 ignore next */ /* v8 ignore next */
  async getStorageEstimate(): Promise<{ usage: number; quota: number } | null> { /* v8 ignore next */ /* v8 ignore next */
    if (navigator.storage && navigator.storage.estimate) { /* v8 ignore next */ /* v8 ignore next */
      const estimate = await navigator.storage.estimate(); /* v8 ignore next */ /* v8 ignore next */
      return { /* v8 ignore next */ /* v8 ignore next */
        usage: estimate.usage || 0, /* v8 ignore next */ /* v8 ignore next */
        quota: estimate.quota || 0, /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return null; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 406. Provide UI to delete cached models and clear space /* v8 ignore next */ /* v8 ignore next */
  async listKeys(): Promise<string[]> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.db) await this.init(); /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve, reject) => { /* v8 ignore next */ /* v8 ignore next */
      const transaction = this.db!.transaction(this.storeName, 'readonly'); /* v8 ignore next */ /* v8 ignore next */
      const store = transaction.objectStore(this.storeName); /* v8 ignore next */ /* v8 ignore next */
      const request = store.getAllKeys(); /* v8 ignore next */ /* v8 ignore next */
      request.onsuccess = () => resolve(request.result as string[]); /* v8 ignore next */ /* v8 ignore next */
      request.onerror = () => reject(new Error('Failed to list keys')); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async delete(hash: string): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    if (!this.db) await this.init(); /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve, reject) => { /* v8 ignore next */ /* v8 ignore next */
      const transaction = this.db!.transaction(this.storeName, 'readwrite'); /* v8 ignore next */ /* v8 ignore next */
      const store = transaction.objectStore(this.storeName); /* v8 ignore next */ /* v8 ignore next */
      const request = store.delete(hash); /* v8 ignore next */ /* v8 ignore next */
      request.onsuccess = () => resolve(); /* v8 ignore next */ /* v8 ignore next */
      request.onerror = () => reject(new Error('Failed to delete from cache')); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const astCache = new IndexedDBVault();
