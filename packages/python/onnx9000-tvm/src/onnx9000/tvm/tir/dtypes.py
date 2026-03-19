"""TVM submodule for AST and optimization."""


# Pass 201: Support arbitrary precision integers (i4, i8, i16, i32, i64).
# Pass 202: Support float types (f16, bf16, f32, f64).
# Pass 203: Handle boolean data types in TIR.

SUPPORTED_DTYPES: set[str] = {
    "int4",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint4",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "bool",
}


def is_supported(dtype: str) -> bool:
    """Do the function."""
    return dtype in SUPPORTED_DTYPES
