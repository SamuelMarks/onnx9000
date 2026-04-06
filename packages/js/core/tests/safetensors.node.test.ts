import { test, expect, vi } from 'vitest';
import * as node_safetensors from '../src/parser/safetensors.node';
import {
  SafetensorsHeaderTooLargeError,
  SafetensorsInvalidJSONError,
} from '../src/parser/safetensors';

test('safetensors.node readSafetensorsHeaderSync success', () => {
  const mockFs = {
    readSync: vi.fn((fd, buf, off, len, pos) => {
      if (pos === 0) {
        // Return header size (8 bytes)
        const sizeBuf = Buffer.alloc(8);
        sizeBuf.writeBigUInt64LE(BigInt(20), 0);
        sizeBuf.copy(buf);
      } else if (pos === 8) {
        // Return header JSON (20 bytes)
        const jsonBuf = Buffer.from('{"valid": "json_st"}');
        const pad = Buffer.alloc(20 - jsonBuf.length, 32); // spaces
        Buffer.concat([jsonBuf, pad]).copy(buf);
      }
    }),
  };

  const { headerObj, headerSize } = node_safetensors.readSafetensorsHeaderSync(1, mockFs);
  expect(headerSize).toBe(20);
  expect(headerObj.valid).toBe('json_st');
});

test('safetensors.node readSafetensorsHeaderSync too large', () => {
  const mockFs = {
    readSync: vi.fn((fd, buf, off, len, pos) => {
      if (pos === 0) {
        const sizeBuf = Buffer.alloc(8);
        sizeBuf.writeBigUInt64LE(BigInt(200 * 1024 * 1024), 0); // 200MB
        sizeBuf.copy(buf);
      }
    }),
  };

  expect(() => {
    node_safetensors.readSafetensorsHeaderSync(1, mockFs);
  }).toThrowError(SafetensorsHeaderTooLargeError);
});

test('safetensors.node readSafetensorsHeaderSync invalid JSON', () => {
  const mockFs = {
    readSync: vi.fn((fd, buf, off, len, pos) => {
      if (pos === 0) {
        const sizeBuf = Buffer.alloc(8);
        sizeBuf.writeBigUInt64LE(BigInt(10), 0);
        sizeBuf.copy(buf);
      } else if (pos === 8) {
        Buffer.from('not json{').copy(buf);
      }
    }),
  };

  expect(() => {
    node_safetensors.readSafetensorsHeaderSync(1, mockFs);
  }).toThrowError(SafetensorsInvalidJSONError);
});

test('safetensors.node readSafetensorsChunkSync', () => {
  const mockFs = {
    readSync: vi.fn((fd, buf, off, len, pos) => {
      Buffer.from('abcdefgh').copy(buf);
    }),
  };
  const res = node_safetensors.readSafetensorsChunkSync(1, mockFs, 10, 0, 8);
  expect(new TextDecoder().decode(res)).toBe('abcdefgh');
});

test('safetensors.node saveSafetensorsFileSync success', () => {
  const mockFs = {
    writeFileSync: vi.fn(),
  };
  node_safetensors.saveSafetensorsFileSync('test.safetensors', mockFs, {
    a: new Uint8Array([1, 2, 3]),
  });
  expect(mockFs.writeFileSync).toHaveBeenCalled();
});

test('safetensors.node saveSafetensorsFileSync ENOSPC', () => {
  const mockFs = {
    writeFileSync: vi.fn(() => {
      const e = new Error('ENOSPC') as Object;
      e.code = 'ENOSPC';
      throw e;
    }),
  };
  expect(() => {
    node_safetensors.saveSafetensorsFileSync('test.safetensors', mockFs, {
      a: new Uint8Array([1, 2, 3]),
    });
  }).toThrowError(/disk space exhausted/);

  const mockFs2 = {
    writeFileSync: vi.fn(() => {
      const e = new Error('some error with ENOSPC inside') as Object;
      throw e;
    }),
  };
  expect(() => {
    node_safetensors.saveSafetensorsFileSync('test.safetensors', mockFs2, {
      a: new Uint8Array([1, 2, 3]),
    });
  }).toThrowError(/disk space exhausted/);

  const mockFs3 = {
    writeFileSync: vi.fn(() => {
      throw new Error('generic error');
    }),
  };
  expect(() => {
    node_safetensors.saveSafetensorsFileSync('test.safetensors', mockFs3, {
      a: new Uint8Array([1, 2, 3]),
    });
  }).toThrowError(/generic error/);
});
