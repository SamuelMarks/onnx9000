/**
 * Parameters defining the LLM architecture and limits.
 */
export interface ModelParams {
  /** Maximum context window size. */
  maxSequenceLength: number;
  /** Number of transformer blocks. */
  numHiddenLayers: number;
  /** Number of query attention heads. */
  numAttentionHeads: number;
  /** Number of key/value attention heads (for GQA). */
  numKeyValueHeads: number;
  /** Dimension of the hidden state. */
  hiddenSize: number;
  /** Size of the vocabulary. */
  vocabSize: number;
  /** Token ID(s) marking the end of generation. */
  eosTokenId: number | number[];
  /** Optional token ID for sequence start. */
  bosTokenId?: number;
  /** Optional padding token ID. */
  padTokenId?: number;
  /** Whether to enable automatic subgraph partitioning. */
  subgraphPartitioning?: boolean;
}

/**
 * Parameters controlling the generation process.
 */
export interface GeneratorParams {
  /** Absolute maximum sequence length (including prompt). */
  maxLength: number;
  /** Maximum number of new tokens to generate. */
  maxNewTokens?: number;
  /** Whether to stop early in beam search. */
  earlyStopping: boolean;
  /** Beam width for beam search. */
  numBeams: number;
  /** Sampling temperature. */
  temperature: number;
  /** Top-K sampling threshold. */
  topK: number;
  /** Top-P (nucleus) sampling threshold. */
  topP: number;
  /** Penalty for repeated tokens. */
  repetitionPenalty: number;
  /** Number of independent sequences to generate. */
  numReturnSequences: number;
  /** Whether to use probabilistic sampling. */
  doSample: boolean;
  /** Optional random seed. */
  seed?: number;
  /** Optional signal to cancel generation. */
  abortSignal?: AbortSignal;
  /** Penalty for sequence length in beam search. */
  lengthPenalty?: number;
  /** Number of beam groups for diverse search. */
  numBeamGroups?: number;
  /** Typical sampling threshold. */
  typicalP?: number;
  /** Alpha factor for contrastive search. */
  penaltyAlpha?: number;
  /** Whether to include per-token probabilities in output. */
  returnTokenProbabilities?: boolean;
}

/**
 * Callback for monitoring GenAI operations (e.g., downloads).
 */
export interface GenAIProgressCallback {
  /** @param downloaded Bytes received. @param total Total bytes. */
  (downloaded: number, total: number): void;
}

/**
 * Performance metrics for generation.
 */
export interface ProfilingData {
  /** Time To First Token in milliseconds. */
  ttft: number;
  /** Tokens Per Second. */
  tps: number;
}
