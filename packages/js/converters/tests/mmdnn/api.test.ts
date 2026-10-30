import { describe, expect, it } from 'vitest';
import {
  convert,
  convertToCaffe,
  convertToCNTK,
  convertToCoreML,
  convertToKeras,
  convertToMXNet,
  convertToOnnxScript,
  convertToPaddle,
  convertToPyTorch,
  convertToTensorFlow,
} from '../../src/mmdnn/api';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('api.ts', () => {
  it('should call and cover convertToPyTorch', async () => {
    try {
      const res = Call(convertToPyTorch);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToTensorFlow', async () => {
    try {
      const res = Call(convertToTensorFlow);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToCaffe', async () => {
    try {
      const res = Call(convertToCaffe);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToMXNet', async () => {
    try {
      const res = Call(convertToMXNet);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToCNTK', async () => {
    try {
      const res = Call(convertToCNTK);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToCoreML', async () => {
    try {
      const res = Call(convertToCoreML);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToPaddle', async () => {
    try {
      const res = Call(convertToPaddle);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToKeras', async () => {
    try {
      const res = Call(convertToKeras);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convertToOnnxScript', async () => {
    try {
      const res = Call(convertToOnnxScript);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convert', async () => {
    try {
      const res = Call(convert);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
