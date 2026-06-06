import { describe, expect, it } from 'vitest';
import { SafetensorsParser } from '../src/parsers/Safetensors';

describe('Safetensors Parser', () => {
  it('should parse an empty safetensors buffer', () => {
    const encoder = new TextEncoder();
    const headerStr = JSON.stringify({ __metadata__: { format: 'pt' } });

    // Add padding to make header 8-byte aligned if needed
    let headerBytes = encoder.encode(headerStr);
    const paddingLength = (8 - (headerBytes.length % 8)) % 8;
    if (paddingLength > 0) {
      headerBytes = encoder.encode(headerStr + ' '.repeat(paddingLength));
    }

    const lenBuf = new Uint8Array(8);
    const dv = new DataView(lenBuf.buffer);
    dv.setUint32(0, headerBytes.length, true);

    const finalBuffer = new Uint8Array(8 + headerBytes.length);
    finalBuffer.set(lenBuf, 0);
    finalBuffer.set(headerBytes, 8);

    const parser = new SafetensorsParser(finalBuffer.buffer);
    const result = parser.parse();

    expect(result.metadata.format).to.equal('pt');
    expect(Object.keys(result.tensors).length).to.equal(0);
  });
});
