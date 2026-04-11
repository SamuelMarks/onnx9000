/* eslint-disable */
/**
 * Supported payload types for worker messages.
 */
export type WorkerMessagePayload =
  | string
  | number
  | boolean
  | number[]
  | { [key: string]: string | number | boolean | number[] | undefined | null }
  | null;

/**
 * Message format for GenAI web workers.
 */
export interface WorkerMessage {
  /** Message type identifier. */
  type: string;
  /** Message payload data. */
  payload: WorkerMessagePayload;
}

/**
 * Processor for batched prefix trees.
 */
export class PrefixTreeBatchedProcessor {}

/**
 * Decoder implementation for lookahead decoding.
 */
export class LookaheadDecoder {}

/**
 * Decoder implementation for Medusa/Eagle architectures.
 */
export class MedusaEagleDecoder {}

/**
 * Matcher for RAG-based prefix optimizations.
 */
export class RAGPrefixMatcher {}
