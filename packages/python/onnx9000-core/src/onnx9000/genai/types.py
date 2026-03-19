from typing import List, Optional, Union


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
        eos_token_id: Union[int, list[int]],
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
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
        max_new_tokens: Optional[int] = None,
        early_stopping: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        seed: Optional[int] = None,
        abort_signal: Optional[bool] = False,
    ):
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
