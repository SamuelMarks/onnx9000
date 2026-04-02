# Safetensors Support Coverage

Tracking exhaustive coverage of the `safetensors` python package API.

## Detailed API

| Object Name             | Type     | Signature                                                                                                                           |
| ----------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `SafetensorError`       | Object   | ``                                                                                                                                  |
| `deserialize`           | Object   | ``                                                                                                                                  |
| `flax.load`             | Function | `(data: bytes) -> dict[str, Array]`                                                                                                 |
| `flax.load_file`        | Function | `(filename: Union[str, os.PathLike]) -> dict[str, Array]`                                                                           |
| `flax.numpy`            | Object   | ``                                                                                                                                  |
| `flax.safe_open`        | Object   | ``                                                                                                                                  |
| `flax.save`             | Function | `(tensors: dict[str, Array], metadata: Optional[dict[str, str]]) -> bytes`                                                          |
| `flax.save_file`        | Function | `(tensors: dict[str, Array], filename: Union[str, os.PathLike], metadata: Optional[dict[str, str]]) -> None`                        |
| `mlx.load`              | Function | `(data: bytes) -> dict[str, mx.array]`                                                                                              |
| `mlx.load_file`         | Function | `(filename: Union[str, os.PathLike]) -> dict[str, mx.array]`                                                                        |
| `mlx.numpy`             | Object   | ``                                                                                                                                  |
| `mlx.safe_open`         | Object   | ``                                                                                                                                  |
| `mlx.save`              | Function | `(tensors: dict[str, mx.array], metadata: Optional[dict[str, str]]) -> bytes`                                                       |
| `mlx.save_file`         | Function | `(tensors: dict[str, mx.array], filename: Union[str, os.PathLike], metadata: Optional[dict[str, str]]) -> None`                     |
| `numpy.deserialize`     | Object   | ``                                                                                                                                  |
| `numpy.load`            | Function | `(data: bytes) -> dict[str, np.ndarray]`                                                                                            |
| `numpy.load_file`       | Function | `(filename: Union[str, os.PathLike]) -> dict[str, np.ndarray]`                                                                      |
| `numpy.safe_open`       | Object   | ``                                                                                                                                  |
| `numpy.save`            | Function | `(tensor_dict: dict[str, np.ndarray], metadata: Optional[dict[str, str]]) -> bytes`                                                 |
| `numpy.save_file`       | Function | `(tensor_dict: dict[str, np.ndarray], filename: Union[str, os.PathLike], metadata: Optional[dict[str, str]]) -> None`               |
| `numpy.serialize`       | Object   | ``                                                                                                                                  |
| `numpy.serialize_file`  | Object   | ``                                                                                                                                  |
| `paddle.NPDTYPES`       | Object   | ``                                                                                                                                  |
| `paddle.deserialize`    | Object   | ``                                                                                                                                  |
| `paddle.load`           | Function | `(data: bytes, device: str) -> dict[str, paddle.Tensor]`                                                                            |
| `paddle.load_file`      | Function | `(filename: Union[str, os.PathLike], device) -> dict[str, paddle.Tensor]`                                                           |
| `paddle.numpy`          | Object   | ``                                                                                                                                  |
| `paddle.safe_open`      | Object   | ``                                                                                                                                  |
| `paddle.save`           | Function | `(tensors: dict[str, paddle.Tensor], metadata: Optional[dict[str, str]]) -> bytes`                                                  |
| `paddle.save_file`      | Function | `(tensors: dict[str, paddle.Tensor], filename: Union[str, os.PathLike], metadata: Optional[dict[str, str]]) -> None`                |
| `paddle.serialize`      | Object   | ``                                                                                                                                  |
| `paddle.serialize_file` | Object   | ``                                                                                                                                  |
| `safe_open`             | Object   | ``                                                                                                                                  |
| `serialize`             | Object   | ``                                                                                                                                  |
| `serialize_file`        | Object   | ``                                                                                                                                  |
| `tensorflow.load`       | Function | `(data: bytes) -> dict[str, tf.Tensor]`                                                                                             |
| `tensorflow.load_file`  | Function | `(filename: Union[str, os.PathLike]) -> dict[str, tf.Tensor]`                                                                       |
| `tensorflow.numpy`      | Object   | ``                                                                                                                                  |
| `tensorflow.safe_open`  | Object   | ``                                                                                                                                  |
| `tensorflow.save`       | Function | `(tensors: dict[str, tf.Tensor], metadata: Optional[dict[str, str]]) -> bytes`                                                      |
| `tensorflow.save_file`  | Function | `(tensors: dict[str, tf.Tensor], filename: Union[str, os.PathLike], metadata: Optional[dict[str, str]]) -> None`                    |
| `torch.deserialize`     | Object   | ``                                                                                                                                  |
| `torch.load`            | Function | `(data: bytes) -> dict[str, torch.Tensor]`                                                                                          |
| `torch.load_file`       | Function | `(filename: Union[str, os.PathLike], device: Union[str, int]) -> dict[str, torch.Tensor]`                                           |
| `torch.load_model`      | Function | `(model: torch.nn.Module, filename: Union[str, os.PathLike], strict: bool, device: Union[str, int]) -> Tuple[list[str], list[str]]` |
| `torch.safe_open`       | Object   | ``                                                                                                                                  |
| `torch.save`            | Function | `(tensors: dict[str, torch.Tensor], metadata: Optional[dict[str, str]]) -> bytes`                                                   |
| `torch.save_file`       | Function | `(tensors: dict[str, torch.Tensor], filename: Union[str, os.PathLike], metadata: Optional[dict[str, str]])`                         |
| `torch.save_model`      | Function | `(model: torch.nn.Module, filename: str, metadata: Optional[dict[str, str]], force_contiguous: bool)`                               |
| `torch.serialize`       | Object   | ``                                                                                                                                  |
| `torch.serialize_file`  | Object   | ``                                                                                                                                  |
| `torch.storage_ptr`     | Function | `(tensor: torch.Tensor) -> int`                                                                                                     |
| `torch.storage_size`    | Function | `(tensor: torch.Tensor) -> int`                                                                                                     |
