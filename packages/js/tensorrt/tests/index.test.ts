import { DataType, ElementWiseOperation } from '../src/enums';
import { trtFfi } from '../src/ffi';

describe('TensorRT FFI', () => {
  it('should have correct enums', () => {
    expect(DataType.kFLOAT).toBe(0);
    expect(ElementWiseOperation.kSUM).toBe(0);
  });

  it('should instantiate ffi correctly', () => {
    expect(trtFfi.getVersion()).toBeDefined();
  });
});
