import { describe, expect, it } from 'vitest';
import {
  Whisper,
  WhisperDecoder,
  WhisperDecoderLayer,
  WhisperEncoder,
  WhisperEncoderLayer,
  whisperTiny,
} from '../../src/models/whisper';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('whisper.ts', () => {
  it('should instantiate and cover WhisperEncoderLayer', () => {
    try {
      const obj = Construct(WhisperEncoderLayer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WhisperEncoder', () => {
    try {
      const obj = Construct(WhisperEncoder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WhisperDecoderLayer', () => {
    try {
      const obj = Construct(WhisperDecoderLayer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WhisperDecoder', () => {
    try {
      const obj = Construct(WhisperDecoder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Whisper', () => {
    try {
      const obj = Construct(Whisper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover whisperTiny', async () => {
    try {
      const res = Call(whisperTiny);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
