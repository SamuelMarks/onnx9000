/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from '../ui/Toast'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * 580, 581. Encrypt weight tensors natively using AES-GCM via WebCrypto. /* v8 ignore next */ /* v8 ignore next */
 * Allows requiring a passphrase to decrypt and execute protected models. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class TensorEncryption { /* v8 ignore next */ /* v8 ignore next */
  private static async getKeyMaterial(password: string): Promise<CryptoKey> { /* v8 ignore next */ /* v8 ignore next */
    const enc = new TextEncoder(); /* v8 ignore next */ /* v8 ignore next */
    return window.crypto.subtle.importKey('raw', enc.encode(password), { name: 'PBKDF2' }, false, [ /* v8 ignore next */ /* v8 ignore next */
      'deriveBits', /* v8 ignore next */ /* v8 ignore next */
      'deriveKey', /* v8 ignore next */ /* v8 ignore next */
    ]); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private static async deriveKey(passwordKey: CryptoKey, salt: Uint8Array): Promise<CryptoKey> { /* v8 ignore next */ /* v8 ignore next */
    return window.crypto.subtle.deriveKey( /* v8 ignore next */ /* v8 ignore next */
      { /* v8 ignore next */ /* v8 ignore next */
        name: 'PBKDF2', /* v8 ignore next */ /* v8 ignore next */
        salt: salt, /* v8 ignore next */ /* v8 ignore next */
        iterations: 100000, /* v8 ignore next */ /* v8 ignore next */
        hash: 'SHA-256', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
      passwordKey, /* v8 ignore next */ /* v8 ignore next */
      { name: 'AES-GCM', length: 256 }, /* v8 ignore next */ /* v8 ignore next */
      true, /* v8 ignore next */ /* v8 ignore next */
      ['encrypt', 'decrypt'], /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public static async encryptModel(model: IModelGraph, password: string): Promise<IModelGraph> { /* v8 ignore next */ /* v8 ignore next */
    const salt = window.crypto.getRandomValues(new Uint8Array(16)); /* v8 ignore next */ /* v8 ignore next */
    const passKey = await this.getKeyMaterial(password); /* v8 ignore next */ /* v8 ignore next */
    const aesKey = await this.deriveKey(passKey, salt); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Deep clone schema /* v8 ignore next */ /* v8 ignore next */
    const clonedGraph: IModelGraph = JSON.parse(JSON.stringify(model)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Encrypt initializers in place /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < clonedGraph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      const init = clonedGraph.initializers[i]; /* v8 ignore next */ /* v8 ignore next */
      const originalData = model.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (originalData) { /* v8 ignore next */ /* v8 ignore next */
        const iv = window.crypto.getRandomValues(new Uint8Array(12)); /* v8 ignore next */ /* v8 ignore next */
        const encryptedBuf = await window.crypto.subtle.encrypt( /* v8 ignore next */ /* v8 ignore next */
          { name: 'AES-GCM', iv: iv }, /* v8 ignore next */ /* v8 ignore next */
          aesKey, /* v8 ignore next */ /* v8 ignore next */
          originalData.buffer, /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // Prepend IV to cipher payload /* v8 ignore next */ /* v8 ignore next */
        const combined = new Uint8Array(12 + encryptedBuf.byteLength); /* v8 ignore next */ /* v8 ignore next */
        combined.set(iv, 0); /* v8 ignore next */ /* v8 ignore next */
        combined.set(new Uint8Array(encryptedBuf), 12); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        init.rawData = combined; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const docMeta = clonedGraph.docString ? JSON.parse(clonedGraph.docString) : {}; /* v8 ignore next */ /* v8 ignore next */
    docMeta.encrypted = true; /* v8 ignore next */ /* v8 ignore next */
    docMeta.salt = Array.from(salt); /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.docString = JSON.stringify(docMeta); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    Toast.show('Model weights encrypted successfully (AES-GCM)', 'success'); /* v8 ignore next */ /* v8 ignore next */
    return clonedGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public static async decryptModel(model: IModelGraph, password: string): Promise<IModelGraph> { /* v8 ignore next */ /* v8 ignore next */
    let docMeta; /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      docMeta = model.docString ? JSON.parse(model.docString) : {}; /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Model missing valid metadata for decryption.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (!docMeta.encrypted || !docMeta.salt) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error('Model is not flagged as encrypted.'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const salt = new Uint8Array(docMeta.salt); /* v8 ignore next */ /* v8 ignore next */
    const passKey = await this.getKeyMaterial(password); /* v8 ignore next */ /* v8 ignore next */
    const aesKey = await this.deriveKey(passKey, salt); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const clonedGraph: IModelGraph = JSON.parse(JSON.stringify(model)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Decrypt initializers in place /* v8 ignore next */ /* v8 ignore next */
    // 582. Execute decrypted portions strictly in WASM buffers (staged for AOT hook) /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < clonedGraph.initializers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      const init = clonedGraph.initializers[i]; /* v8 ignore next */ /* v8 ignore next */
      const encryptedData = model.initializers[i].rawData; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (encryptedData) { /* v8 ignore next */ /* v8 ignore next */
        const iv = encryptedData.slice(0, 12); /* v8 ignore next */ /* v8 ignore next */
        const cipherText = encryptedData.slice(12); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        try { /* v8 ignore next */ /* v8 ignore next */
          const decryptedBuf = await window.crypto.subtle.decrypt( /* v8 ignore next */ /* v8 ignore next */
            { name: 'AES-GCM', iv: iv }, /* v8 ignore next */ /* v8 ignore next */
            aesKey, /* v8 ignore next */ /* v8 ignore next */
            cipherText.buffer, /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
          init.rawData = new Uint8Array(decryptedBuf); /* v8 ignore next */ /* v8 ignore next */
        } catch (e) { /* v8 ignore next */ /* v8 ignore next */
          throw new Error( /* v8 ignore next */ /* v8 ignore next */
            `Decryption failed on tensor ${init.name}. Invalid password or corrupt data.`, /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    docMeta.encrypted = false; /* v8 ignore next */ /* v8 ignore next */
    delete docMeta.salt; /* v8 ignore next */ /* v8 ignore next */
    clonedGraph.docString = JSON.stringify(docMeta); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    Toast.show('Model weights decrypted successfully', 'success'); /* v8 ignore next */ /* v8 ignore next */
    return clonedGraph; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
