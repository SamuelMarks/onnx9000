"""Module providing core logic and structural definitions."""

import struct
from typing import List, Tuple, Dict, Optional, Any


class SPMNode:
    """Provides semantic functionality and verification."""

    def __init__(self, piece: str, score: float, ptype: int) -> None:
        """Provides semantic functionality and verification."""
        self.piece = piece
        self.score = score
        self.type = ptype


def _read_varint(buffer: bytes, offset: int) -> Tuple[int, int]:
    """Provides semantic functionality and verification."""
    res = 0
    shift = 0
    while True:
        b = buffer[offset]
        offset += 1
        res |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return res, offset


def parse_spm_model(buffer: bytes) -> List[SPMNode]:
    """
    Zero-dependency pure Python parser for SentencePiece model protobuf files.
    Extracts the vocabulary (pieces, scores, and types).
    """
    pieces: List[SPMNode] = []
    offset = 0
    length = len(buffer)

    while offset < length:
        tag, offset = _read_varint(buffer, offset)
        field = tag >> 3
        wire_type = tag & 7

        if wire_type == 0:
            _, offset = _read_varint(buffer, offset)
        elif wire_type == 1:
            offset += 8
        elif wire_type == 5:
            offset += 4
        elif wire_type == 2:
            msg_len, offset = _read_varint(buffer, offset)
            if field == 1:  # pieces
                end = offset + msg_len

                piece_str = ""
                score = 0.0
                ptype = 1  # NORMAL default

                while offset < end:
                    ptag, offset = _read_varint(buffer, offset)
                    pfield = ptag >> 3
                    pwire = ptag & 7

                    if pwire == 0:
                        val, offset = _read_varint(buffer, offset)
                        if pfield == 3:  # type
                            ptype = val
                    elif pwire == 5:
                        if pfield == 2:  # score
                            score_bytes = buffer[offset : offset + 4]
                            score = struct.unpack("<f", score_bytes)[0]
                        offset += 4
                    elif pwire == 1:
                        offset += 8
                    elif pwire == 2:
                        plen, offset = _read_varint(buffer, offset)
                        if pfield == 1:  # piece
                            piece_str = buffer[offset : offset + plen].decode("utf-8")
                        offset += plen

                pieces.append(SPMNode(piece_str, score, ptype))
            else:
                offset += msg_len

    return pieces
