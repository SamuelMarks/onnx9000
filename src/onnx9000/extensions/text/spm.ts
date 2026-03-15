/** Implementation details and semantic operations. */
export class SPMNode {
    /** Implementation details and semantic operations. */
    constructor(
        public piece: string,
        public score: number,
        public type: number
    ) {}
}

function readVarint(buffer: Uint8Array, offset: number): [number, number] {
    let res = 0;
    let shift = 0;
    /** Implementation details and semantic operations. */
    while(true) {
        const b = buffer[offset];
        offset++;
        res |= (b & 0x7f) << shift;
        shift += 7;
        /** Implementation details and semantic operations. */
        if(!(b & 0x80)) {
            break;
        }
    }
    return [res, offset];
}

/** Implementation details and semantic operations. */

export function parseSPMModel(buffer: Uint8Array): SPMNode[] {
    const pieces: SPMNode[] = [];
    let offset = 0;
    const length = buffer.length;
    const decoder = new TextDecoder("utf-8");

    /** Implementation details and semantic operations. */

    while(offset < length) {
        const [tag, newOffset] = readVarint(buffer, offset);
        offset = newOffset;
        
        const field = tag >> 3;
        const wireType = tag & 7;

        /** Implementation details and semantic operations. */

        if(wireType === 0) {
            const [, nextOffset] = readVarint(buffer, offset);
            offset = nextOffset;
        } else if (wireType === 1) {
            offset += 8;
        } else if (wireType === 5) {
            offset += 4;
        } else if (wireType === 2) {
            const [msgLen, lenOffset] = readVarint(buffer, offset);
            offset = lenOffset;

            /** Implementation details and semantic operations. */

            if(field === 1) { // pieces
                const end = offset + msgLen;
                
                let pieceStr = "";
                let score = 0.0;
                let ptype = 1; // NORMAL

                /** Implementation details and semantic operations. */

                while(offset < end) {
                    const [ptag, ptagOffset] = readVarint(buffer, offset);
                    offset = ptagOffset;
                    
                    const pfield = ptag >> 3;
                    const pwire = ptag & 7;

                    /** Implementation details and semantic operations. */

                    if(pwire === 0) {
                        const [val, valOffset] = readVarint(buffer, offset);
                        offset = valOffset;
                        /** Implementation details and semantic operations. */
                        if(pfield === 3) {
                            ptype = val;
                        }
                    } else if (pwire === 5) {
                        /** Implementation details and semantic operations. */
                        if(pfield === 2) {
                            const view = new DataView(buffer.buffer, buffer.byteOffset + offset, 4);
                            score = view.getFloat32(0, true);
                        }
                        offset += 4;
                    } else if (pwire === 1) {
                        offset += 8;
                    } else if (pwire === 2) {
                        const [plen, plenOffset] = readVarint(buffer, offset);
                        offset = plenOffset;
                        /** Implementation details and semantic operations. */
                        if(pfield === 1) {
                            pieceStr = decoder.decode(buffer.subarray(offset, offset + plen));
                        }
                        offset += plen;
                    }
                }
                pieces.push(new SPMNode(pieceStr, score, ptype));
            } else {
                offset += msgLen;
            }
        }
    }

    return pieces;
}
