"""Flax parser module."""

import struct


def parse_msgpack(data: bytes):
    """Docstring for D103."""

    def _read(offset):
        """read."""
        if offset >= len(data):
            raise ValueError("Unexpected end of data")
        b = data[offset]
        offset += 1

        if b <= 0x7F:
            return b, offset
        elif 0xE0 <= b <= 0xFF:
            return b - 0x100, offset
        elif 0x80 <= b <= 0x8F:
            n = b & 0x0F
            return _read_map(n, offset)
        elif 0x90 <= b <= 0x9F:
            n = b & 0x0F
            return _read_array(n, offset)
        elif 0xA0 <= b <= 0xBF:
            n = b & 0x1F
            return _read_str(n, offset)
        elif b == 0xC0:
            return None, offset
        elif b == 0xC2:
            return False, offset
        elif b == 0xC3:
            return True, offset
        elif b == 0xC4:
            if offset >= len(data):
                raise ValueError("Unexpected end of data")
            n = data[offset]
            offset += 1
            return _read_bin(n, offset)
        elif b == 0xC5:
            (n,) = struct.unpack_from(">H", data, offset)
            offset += 2
            return _read_bin(n, offset)
        elif b == 0xC6:
            (n,) = struct.unpack_from(">I", data, offset)
            offset += 4
            return _read_bin(n, offset)
        elif b == 0xCA:
            (val,) = struct.unpack_from(">f", data, offset)
            offset += 4
            return val, offset
        elif b == 0xCB:
            (val,) = struct.unpack_from(">d", data, offset)
            offset += 8
            return val, offset
        elif b == 0xCC:
            val = data[offset]
            offset += 1
            return val, offset
        elif b == 0xCD:
            (val,) = struct.unpack_from(">H", data, offset)
            offset += 2
            return val, offset
        elif b == 0xCE:
            (val,) = struct.unpack_from(">I", data, offset)
            offset += 4
            return val, offset
        elif b == 0xCF:
            (val,) = struct.unpack_from(">Q", data, offset)
            offset += 8
            return val, offset
        elif b == 0xD0:
            (val,) = struct.unpack_from(">b", data, offset)
            offset += 1
            return val, offset
        elif b == 0xD1:
            (val,) = struct.unpack_from(">h", data, offset)
            offset += 2
            return val, offset
        elif b == 0xD2:
            (val,) = struct.unpack_from(">i", data, offset)
            offset += 4
            return val, offset
        elif b == 0xD3:
            (val,) = struct.unpack_from(">q", data, offset)
            offset += 8
            return val, offset
        elif b == 0xD9:
            if offset >= len(data):
                raise ValueError("Unexpected end of data")
            n = data[offset]
            offset += 1
            return _read_str(n, offset)
        elif b == 0xDA:
            (n,) = struct.unpack_from(">H", data, offset)
            offset += 2
            return _read_str(n, offset)
        elif b == 0xDB:
            (n,) = struct.unpack_from(">I", data, offset)
            offset += 4
            return _read_str(n, offset)
        elif b == 0xDC:
            (n,) = struct.unpack_from(">H", data, offset)
            offset += 2
            return _read_array(n, offset)
        elif b == 0xDD:
            (n,) = struct.unpack_from(">I", data, offset)
            offset += 4
            return _read_array(n, offset)
        elif b == 0xDE:
            (n,) = struct.unpack_from(">H", data, offset)
            offset += 2
            return _read_map(n, offset)
        elif b == 0xDF:
            (n,) = struct.unpack_from(">I", data, offset)
            offset += 4
            return _read_map(n, offset)
        # ext types (Flax might use them?)
        elif b == 0xC7:  # ext 8
            if offset >= len(data):
                raise ValueError("Unexpected end of data")
            n = data[offset]
            offset += 1
            return _read_ext(n, offset)
        elif b == 0xC8:  # ext 16
            (n,) = struct.unpack_from(">H", data, offset)
            offset += 2
            return _read_ext(n, offset)
        elif b == 0xC9:  # ext 32
            (n,) = struct.unpack_from(">I", data, offset)
            offset += 4
            return _read_ext(n, offset)
        elif b == 0xD4:  # fixext 1
            return _read_ext(1, offset)
        elif b == 0xD5:  # fixext 2
            return _read_ext(2, offset)
        elif b == 0xD6:  # fixext 4
            return _read_ext(4, offset)
        elif b == 0xD7:  # fixext 8
            return _read_ext(8, offset)
        elif b == 0xD8:  # fixext 16
            return _read_ext(16, offset)
        else:
            raise ValueError(f"MsgPack type {hex(b)} not implemented")

    def _read_map(n, offset):
        """read map."""
        res = {}
        for _ in range(n):
            k, offset = _read(offset)
            v, offset = _read(offset)
            res[k] = v
        return res, offset

    def _read_array(n, offset):
        """read array."""
        res = []
        for _ in range(n):
            v, offset = _read(offset)
            res.append(v)
        return res, offset

    def _read_str(n, offset):
        """read str."""
        val = data[offset : offset + n].decode("utf-8")
        return val, offset + n

    def _read_bin(n, offset):
        """read bin."""
        val = data[offset : offset + n]
        return val, offset + n

    def _read_ext(n, offset):
        """read ext."""
        ext_type = data[offset]
        offset += 1
        val = data[offset : offset + n]
        return (ext_type, val), offset + n

    res, _ = _read(0)
    return res
