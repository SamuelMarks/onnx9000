"""Provide functionality for this module."""

from ..core.ir import Tensor


class SequenceTensorUtils:
    """Use utility class for manipulating Tensors specifically for sequence lengths."""

    @staticmethod
    def expand_sequence_dimension(tensor: Tensor, new_seq_len: int) -> Tensor:
        """Execute the expand_sequence_dimension operation."""
        if len(tensor.shape) < 2:
            raise ValueError("Tensor must have at least 2 dimensions to expand sequence length.")

        new_shape = list(tensor.shape)
        old_seq_len = new_shape[1]
        new_shape[1] = new_seq_len

        # dynamic shape allocation strategy implementation
        def get_vol(shape):
            """Execute the get_vol operation."""
            v = 1
            for s in shape:
                v *= s
            return v

        itemsize = tensor.dtype.itemsize if hasattr(tensor.dtype, "itemsize") else 4
        new_vol = get_vol(new_shape)
        new_data = bytearray(new_vol * itemsize)

        old_data = tensor.data
        if old_data is None:
            old_data = bytearray(get_vol(tensor.shape) * itemsize)

        batch_size = new_shape[0]
        inner_vol = 1
        for s in new_shape[2:]:
            inner_vol *= s

        old_inner_bytes = inner_vol * itemsize

        for b in range(batch_size):
            old_batch_offset = b * old_seq_len * old_inner_bytes
            new_batch_offset = b * new_seq_len * old_inner_bytes

            for s in range(old_seq_len):
                old_offset = old_batch_offset + s * old_inner_bytes
                new_offset = new_batch_offset + s * old_inner_bytes
                # slicing memoryview or bytearray
                new_data[new_offset : new_offset + old_inner_bytes] = old_data[
                    old_offset : old_offset + old_inner_bytes
                ]

        return Tensor(
            name=f"{tensor.name}_expanded",
            dtype=tensor.dtype,
            shape=tuple(new_shape),
            data=new_data,
        )
