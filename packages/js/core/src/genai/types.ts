export interface ModelParams {
  maxSequenceLength: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  hiddenSize: number;
  vocabSize: number;
  eosTokenId: number | number[];
  bosTokenId?: number;
  padTokenId?: number;
  subgraphPartitioning?: boolean;
}

export interface GeneratorParams {
  maxLength: number;
  maxNewTokens?: number;
  earlyStopping: boolean;
  numBeams: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  numReturnSequences: number;
  doSample: boolean;
  seed?: number;
  abortSignal?: AbortSignal;
  lengthPenalty?: number;
  numBeamGroups?: number;
  typicalP?: number;
  penaltyAlpha?: number;
  returnTokenProbabilities?: boolean;
}

export interface GenAIProgressCallback {
  (downloaded: number, total: number): void;
}

export interface ProfilingData {
  ttft: number;
  tps: number;
}
