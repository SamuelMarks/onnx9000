import struct
from typing import Optional, Union


class FlatBufferBuilder:
    """Zero-dependency FlatBuffer Builder in Python."""

    def __init__(self, initial_size: int = 1024) -> None:
        """Initialize the FlatBufferBuilder."""
        self.bb = bytearray(initial_size)
        self.space = initial_size
        self.minalign = 1
        self.vtable: Optional[list[int]] = None
        self.object_start = 0
        self.vtables: list[int] = []

    def clear(self) -> None:
        """Clear the builder."""
        self.space = len(self.bb)
        self.minalign = 1
        self.vtable = None
        self.object_start = 0
        self.vtables = []

    def grow_buffer(self) -> None:
        """Double the size of the buffer."""
        old_length = len(self.bb)
        new_length = old_length * 2
        if new_length == 0:
            new_length = 1024
        new_bb = bytearray(new_length)
        new_bb[new_length - old_length :] = self.bb
        self.bb = new_bb
        self.space += new_length - old_length

    def prep(self, size: int, additional_bytes: int) -> None:
        """Prepare to write an element of `size` after `additional_bytes` have been written."""
        if size > self.minalign:
            self.minalign = size
        align_size = (~(len(self.bb) - self.space + additional_bytes) + 1) & (size - 1)
        while self.space < align_size + size + additional_bytes:
            self.grow_buffer()
        self.pad(align_size)

    def pad(self, bytes_to_pad: int) -> None:
        """Pad the buffer."""
        for _ in range(bytes_to_pad):
            self.space -= 1
            self.bb[self.space] = 0

    def place(self, x: int) -> None:
        """Place a 32-bit int."""
        self.space -= 4
        struct.pack_into("<i", self.bb, self.space, x)

    def place_int8(self, x: int) -> None:
        """Place an 8-bit int."""
        self.space -= 1
        x = x if x <= 127 else x - 256
        x = x if x >= -128 else x + 256
        struct.pack_into("<b", self.bb, self.space, x)

    def place_int16(self, x: int) -> None:
        """Place a 16-bit int."""
        self.space -= 2
        struct.pack_into("<h", self.bb, self.space, x)

    def place_int32(self, x: int) -> None:
        """Place a 32-bit int."""
        self.space -= 4
        struct.pack_into("<i", self.bb, self.space, x)

    def place_int64(self, x: int) -> None:
        """Place a 64-bit int."""
        self.space -= 8
        struct.pack_into("<q", self.bb, self.space, x)

    def place_float32(self, x: float) -> None:
        """Place a 32-bit float."""
        self.space -= 4
        struct.pack_into("<f", self.bb, self.space, x)

    def place_float64(self, x: float) -> None:
        """Place a 64-bit float."""
        self.space -= 8
        struct.pack_into("<d", self.bb, self.space, x)

    def add_int8(self, x: int) -> None:
        """Add an 8-bit int."""
        self.prep(1, 0)
        self.place_int8(x)

    def add_int16(self, x: int) -> None:
        """Add a 16-bit int."""
        self.prep(2, 0)
        self.place_int16(x)

    def add_int32(self, x: int) -> None:
        """Add a 32-bit int."""
        self.prep(4, 0)
        self.place_int32(x)

    def add_int64(self, x: int) -> None:
        """Add a 64-bit int."""
        self.prep(8, 0)
        self.place_int64(x)

    def add_float32(self, x: float) -> None:
        """Add a 32-bit float."""
        self.prep(4, 0)
        self.place_float32(x)

    def add_float64(self, x: float) -> None:
        """Add a 64-bit float."""
        self.prep(8, 0)
        self.place_float64(x)

    def add_offset(self, offset: int) -> None:
        """Add an offset."""
        self.prep(4, 0)
        self.place(self.offset() - offset + 4)

    def start_vector(self, elem_size: int, num_elems: int, alignment: int) -> None:
        """Start a vector."""
        self.prep(4, elem_size * num_elems)
        self.prep(alignment, elem_size * num_elems)

    def end_vector(self, num_elems: int) -> int:
        """End a vector."""
        self.place_int32(num_elems)
        return self.offset()

    def create_byte_vector(
        self, data: Union[bytes, bytearray, list[int]], alignment: int = 4
    ) -> int:
        """Create a byte vector with specific alignment."""
        self.start_vector(1, len(data), alignment)
        self.space -= len(data)
        self.bb[self.space : self.space + len(data)] = data
        return self.end_vector(len(data))

    def create_string(self, s: Union[str, bytes, bytearray]) -> int:
        """Create a string or binary buffer."""
        if isinstance(s, str):
            utf8 = s.encode("utf-8")
        else:
            utf8 = bytes(s)
        self.add_int8(0)
        self.start_vector(1, len(utf8), 1)
        self.space -= len(utf8)
        self.bb[self.space : self.space + len(utf8)] = utf8
        return self.end_vector(len(utf8))

    def start_object(self, numfields: int) -> None:
        """Start an object."""
        self.vtable = [0] * numfields
        self.object_start = self.offset()

    def add_field_int8(self, voffset: int, x: int, d: int) -> None:
        """Add an 8-bit int field."""
        if x != d:
            self.add_int8(x)
            self.slot(voffset)

    def add_field_int16(self, voffset: int, x: int, d: int) -> None:
        """Add a 16-bit int field."""
        if x != d:
            self.add_int16(x)
            self.slot(voffset)

    def add_field_int32(self, voffset: int, x: int, d: int) -> None:
        """Add a 32-bit int field."""
        if x != d:
            self.add_int32(x)
            self.slot(voffset)

    def add_field_int64(self, voffset: int, x: int, d: int) -> None:
        """Add a 64-bit int field."""
        if x != d:
            self.add_int64(x)
            self.slot(voffset)

    def add_field_float32(self, voffset: int, x: float, d: float) -> None:
        """Add a 32-bit float field."""
        if x != d:
            self.add_float32(x)
            self.slot(voffset)

    def add_field_float64(self, voffset: int, x: float, d: float) -> None:
        """Add a 64-bit float field."""
        if x != d:
            self.add_float64(x)
            self.slot(voffset)

    def add_field_offset(self, voffset: int, x: int, d: int) -> None:
        """Add an offset field."""
        if x != d:
            self.add_offset(x)
            self.slot(voffset)

    def slot(self, voffset: int) -> None:
        """Slot a field."""
        if self.vtable is not None:
            self.vtable[voffset] = self.offset()

    def end_object(self) -> int:
        """End an object."""
        if self.vtable is None:
            raise ValueError("FlatBufferBuilder: end_object called without start_object")
        self.add_int32(0)
        vtableloc = self.offset()

        # write vtable entries
        i = len(self.vtable) - 1
        while i >= 0 and self.vtable[i] == 0:
            i -= 1
        trimmed_size = i + 1

        for j in range(trimmed_size - 1, -1, -1):
            off = vtableloc - self.vtable[j] if self.vtable[j] else 0
            self.add_int16(off)

        standard_fields = 2
        self.add_int16(vtableloc - self.object_start)
        self.add_int16((trimmed_size + standard_fields) * 2)

        existing_vtable = 0
        vt1 = self.space
        vt1len = struct.unpack_from("<h", self.bb, vt1)[0]

        for vt2_offset in self.vtables:
            vt2 = len(self.bb) - vt2_offset
            vt2len = struct.unpack_from("<h", self.bb, vt2)[0]
            if vt1len == vt2len:
                match = True
                for k in range(2, vt1len, 2):
                    v1 = struct.unpack_from("<h", self.bb, vt1 + k)[0]
                    v2 = struct.unpack_from("<h", self.bb, vt2 + k)[0]
                    if v1 != v2:
                        match = False
                        break
                if match:
                    existing_vtable = vt2_offset
                    break

        if existing_vtable:
            self.space = len(self.bb) - vtableloc
            struct.pack_into("<i", self.bb, self.space, existing_vtable - vtableloc)
        else:
            self.vtables.append(self.offset())
            struct.pack_into("<i", self.bb, len(self.bb) - vtableloc, self.offset() - vtableloc)

        self.vtable = None
        return vtableloc

    def finish(self, root_table: int, identifier: Optional[str] = None) -> None:
        """Finish the buffer."""
        self.prep(self.minalign, 8 if identifier else 4)
        if identifier:
            al = 4
            self.prep(al, 4)
            for i in range(3, -1, -1):
                self.add_int8(ord(identifier[i]))
        self.add_offset(root_table)

    def offset(self) -> int:
        """Get the current offset."""
        return len(self.bb) - self.space

    def as_bytearray(self) -> bytearray:
        """Get the buffer as a bytearray."""
        return self.bb[self.space :]
