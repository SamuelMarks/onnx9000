import struct
from typing import Optional


class FlatBufferReader:
    """Zero-dependency FlatBuffer Reader."""

    def __init__(self, data: bytes) -> None:
        """Initialize the reader."""
        self.bytes = data

    def get_root(self) -> int:
        """Get the root object offset."""
        return struct.unpack_from("<I", self.bytes, 0)[0]

    def check_magic_bytes(self, magic: str) -> bool:
        """Check if magic bytes are present."""
        if len(self.bytes) < 8:
            return False
        return self.bytes[4:8].decode("ascii") == magic

    def get_vtable(self, obj_offset: int) -> int:
        """Get the vtable offset."""
        vtable_offset = struct.unpack_from("<i", self.bytes, obj_offset)[0]
        return obj_offset - vtable_offset

    def get_field_offset(self, obj_offset: int, vtable_index: int) -> int:
        """Get the field offset relative to obj_offset."""
        vtable = self.get_vtable(obj_offset)
        vtable_size = struct.unpack_from("<H", self.bytes, vtable)[0]
        field_offset = (vtable_index + 2) * 2

        if field_offset >= vtable_size:
            return 0

        offset_in_object = struct.unpack_from("<H", self.bytes, vtable + field_offset)[0]
        if offset_in_object == 0:
            return 0

        return obj_offset + offset_in_object

    def get_int8(self, obj_offset: int, vtable_index: int, def_val: int = 0) -> int:
        """Get int8 field."""
        offset = self.get_field_offset(obj_offset, vtable_index)
        return struct.unpack_from("<b", self.bytes, offset)[0] if offset != 0 else def_val

    def get_int16(self, obj_offset: int, vtable_index: int, def_val: int = 0) -> int:
        """Get int16 field."""
        offset = self.get_field_offset(obj_offset, vtable_index)
        return struct.unpack_from("<h", self.bytes, offset)[0] if offset != 0 else def_val

    def get_int32(self, obj_offset: int, vtable_index: int, def_val: int = 0) -> int:
        """Get int32 field."""
        offset = self.get_field_offset(obj_offset, vtable_index)
        return struct.unpack_from("<i", self.bytes, offset)[0] if offset != 0 else def_val

    def get_float32(self, obj_offset: int, vtable_index: int, def_val: float = 0.0) -> float:
        """Get float32 field."""
        offset = self.get_field_offset(obj_offset, vtable_index)
        return struct.unpack_from("<f", self.bytes, offset)[0] if offset != 0 else def_val

    def get_string(self, obj_offset: int, vtable_index: int) -> Optional[str]:
        """Get string field."""
        offset = self.get_field_offset(obj_offset, vtable_index)
        if offset == 0:
            return None
        offset += struct.unpack_from("<I", self.bytes, offset)[0]
        length = struct.unpack_from("<I", self.bytes, offset)[0]
        return self.bytes[offset + 4 : offset + 4 + length].decode("utf-8")

    def get_vector_length(self, vector_offset: int) -> int:
        """Get vector length."""
        return struct.unpack_from("<I", self.bytes, vector_offset)[0]

    def get_indirect_offset(self, obj_offset: int, vtable_index: int) -> int:
        """Get indirect object offset."""
        offset = self.get_field_offset(obj_offset, vtable_index)
        if offset == 0:
            return 0
        return offset + struct.unpack_from("<I", self.bytes, offset)[0]
