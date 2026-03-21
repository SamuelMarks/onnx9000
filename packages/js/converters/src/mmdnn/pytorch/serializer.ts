import { Tensor } from '@onnx9000/core';
import { zipSync } from 'fflate';

class PickleBuilder {
  private chunks: Uint8Array[] = [];

  write(b: Uint8Array | number[]) {
    if (b instanceof Uint8Array) this.chunks.push(b);
    else this.chunks.push(new Uint8Array(b));
  }

  writeString(s: string) {
    const enc = new TextEncoder().encode(s);
    if (enc.length <= 255) {
      this.write([0x58, enc.length, 0, 0, 0]); // BINUNICODE short
      this.write(enc);
    } else {
      const sizeBuf = new Uint8Array(4);
      new DataView(sizeBuf.buffer).setUint32(0, enc.length, true);
      this.write([0x58]); // BINUNICODE
      this.write(sizeBuf);
      this.write(enc);
    }
  }

  writeAscii(s: string) {
    const enc = new TextEncoder().encode(s);
    this.write(enc);
  }

  writeInt(val: number) {
    if (val >= 0 && val <= 255) {
      this.write([0x4b, val]);
    } else if (val >= 0 && val <= 65535) {
      const b = new Uint8Array(2);
      new DataView(b.buffer).setUint16(0, val, true);
      this.write([0x4d]);
      this.write(b);
    } else if (val <= 2147483647 && val >= -2147483648) {
      const b = new Uint8Array(4);
      new DataView(b.buffer).setInt32(0, val, true);
      this.write([0x4a]);
      this.write(b);
    } else {
      // LONG1
      this.write([0x8a, 8]);
      const b = new Uint8Array(8);
      new DataView(b.buffer).setBigInt64(0, BigInt(val), true);
      this.write(b);
    }
  }

  writeTuple(arr: number[]) {
    this.write([0x28]); // MARK
    for (const val of arr) {
      this.writeInt(val);
    }
    this.write([0x74]); // TUPLE
  }

  getBuffer(): Uint8Array {
    let total = 0;
    for (const c of this.chunks) total += c.length;
    const res = new Uint8Array(total);
    let offset = 0;
    for (const c of this.chunks) {
      res.set(c, offset);
      offset += c.length;
    }
    return res;
  }
}

export class PyTorchSerializer {
  static getStorageClass(dtype: string): string {
    switch (dtype) {
      case 'float32':
        return 'FloatStorage';
      case 'float64':
        return 'DoubleStorage';
      case 'float16':
        return 'HalfStorage';
      case 'bfloat16':
        return 'BFloat16Storage';
      case 'int32':
        return 'IntStorage';
      case 'int64':
        return 'LongStorage';
      case 'int16':
        return 'ShortStorage';
      case 'int8':
        return 'CharStorage';
      case 'uint8':
        return 'ByteStorage';
      case 'bool':
        return 'BoolStorage';
      default:
        return 'FloatStorage';
    }
  }

  static serialize(tensors: Tensor[]): Uint8Array {
    const files: Record<string, Uint8Array> = {};
    files['archive/byteorder'] = new TextEncoder().encode('little');
    files['archive/version'] = new TextEncoder().encode('3');

    const pkl = new PickleBuilder();
    pkl.write([0x80, 0x02]); // PROTO 2
    pkl.write([0x7d, 0x28]); // EMPTY_DICT, MARK

    for (let i = 0; i < tensors.length; i++) {
      const t = tensors[i];
      if (!t) continue;
      const storageType = PyTorchSerializer.getStorageClass(t.dtype);
      const dataStr = i.toString();
      const numElements = t.size;

      let shape = t.shape.map((s) => (typeof s === 'number' ? s : 1));
      if (shape.length === 0) shape = [1];

      const stride = new Array(shape.length).fill(1);
      for (let j = shape.length - 2; j >= 0; j--) {
        stride[j] = stride[j + 1] * (shape[j + 1] || 1);
      }

      pkl.writeString(t.name);
      pkl.writeAscii('ctorch._utils\n_rebuild_tensor_v2\n');
      pkl.write([0x28, 0x28]); // MARK, MARK
      pkl.writeString('storage');
      pkl.writeAscii(`ctorch\n${storageType}\n`);
      pkl.writeString(dataStr);
      pkl.writeString('cpu');
      pkl.writeInt(numElements);
      pkl.writeAscii('tQ'); // TUPLE, BINPERSID
      pkl.writeInt(0); // offset

      pkl.writeTuple(shape);
      pkl.writeTuple(stride);

      pkl.write([0x89]); // requires_grad=False
      pkl.writeAscii('ccollections\nOrderedDict\n)RtR'); // collections.OrderedDict() -> rebuild_tensor

      if (t.data) {
        files[`archive/data/${dataStr}`] = new Uint8Array(
          t.data.buffer,
          t.data.byteOffset,
          t.data.byteLength,
        );
      } else {
        let bpe = 4;
        if (['float64', 'int64'].includes(t.dtype)) bpe = 8;
        if (['float16', 'bfloat16', 'int16'].includes(t.dtype)) bpe = 2;
        if (['int8', 'uint8', 'bool'].includes(t.dtype)) bpe = 1;
        files[`archive/data/${dataStr}`] = new Uint8Array(numElements * bpe);
      }
    }

    pkl.writeAscii('u.'); // SETITEMS, STOP

    files['archive/data.pkl'] = pkl.getBuffer();

    return zipSync(files, { level: 0 }); // level: 0 means NO COMPRESSION, required by torch.load
  }
}
