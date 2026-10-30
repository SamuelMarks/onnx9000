"""Implementation of types (onnx9000 / genai / types.py) providing ModelParams, GeneratorParams."""


class ModelParams:
    """Model configuration parameters."""

    def __init__(
        self,
        max_sequence_length: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        vocab_size: int,
        eos_token_id: int | list[int],
        bos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> None:
        """Initialize ModelParams with configured parameters and state.

        Args:
            max_sequence_length: Parameter specifying max sequence length (int).
            num_hidden_layers: Parameter specifying num hidden layers (int).
            num_attention_heads: Parameter specifying num attention heads (int).
            num_key_value_heads: Parameter specifying num key value heads (int).
            hidden_size: Parameter specifying hidden size (int).
            vocab_size: Parameter specifying vocab size (int).
            eos_token_id: Parameter specifying eos token id (int | list[int]).
            bos_token_id: Parameter specifying bos token id (int | None).
            pad_token_id: Parameter specifying pad token id (int | None).
        """
        self.max_sequence_length = max_sequence_length
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id


class GeneratorParams:
    """Configuration parameters for the generation loop."""

    def __init__(
        self,
        max_length: int,
        max_new_tokens: int | None = None,
        early_stopping: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        seed: int | None = None,
        abort_signal: bool | None = False,
    ) -> None:
        """Initialize GeneratorParams with configured parameters and state.

        Args:
            max_length: Parameter specifying max length (int).
            max_new_tokens: Parameter specifying max new tokens (int | None).
            early_stopping: Parameter specifying early stopping (bool).
            num_beams: Parameter specifying num beams (int).
            temperature: Parameter specifying temperature (float).
            top_k: Parameter specifying top k (int).
            top_p: Parameter specifying top p (float).
            repetition_penalty: Parameter specifying repetition penalty (float).
            num_return_sequences: Parameter specifying num return sequences (int).
            do_sample: Parameter specifying do sample (bool).
            seed: Parameter specifying seed (int | None).
            abort_signal: Parameter specifying abort signal (bool | None).
        """
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_return_sequences = num_return_sequences
        self.do_sample = do_sample
        self.seed = seed
        self.abort_signal = abort_signal
