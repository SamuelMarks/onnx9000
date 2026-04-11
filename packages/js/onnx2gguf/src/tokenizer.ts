/* eslint-disable */
export function extractTokenizerMetadata(
  tokenizerJsonStr: string | null = null,
  vocabSize: number = 0,
): Record<string, ReturnType<typeof JSON.parse>> {
  const meta: Record<string, ReturnType<typeof JSON.parse>> = {};

  if (!tokenizerJsonStr) {
    meta['tokenizer.ggml.model'] = 'llama';
    const tokens = [];
    const size = vocabSize > 0 ? vocabSize : 2;
    for (let i = 0; i < size; i++) {
      tokens.push(vocabSize > 0 ? `[TOKEN_${i}]` : i === 0 ? '<s>' : '</s>');
    }
    meta['tokenizer.ggml.tokens'] = tokens;
    meta['tokenizer.ggml.scores'] = Array(size).fill(0.0);
    meta['tokenizer.ggml.token_type'] = Array(size).fill(1);
    meta['tokenizer.ggml.bos_token_id'] = 0;
    meta['tokenizer.ggml.eos_token_id'] = 1;
    meta['tokenizer.ggml.unknown_token_id'] = 0;
    meta['tokenizer.ggml.padding_token_id'] = 0;
    meta['tokenizer.ggml.separator_token_id'] = 0;
    meta['tokenizer.ggml.add_bos_token'] = true;
    meta['tokenizer.ggml.add_eos_token'] = false;
    meta['tokenizer.chat_template'] = '';
    return meta;
  }

  let t: ReturnType<typeof JSON.parse>;
  try {
    t = JSON.parse(tokenizerJsonStr);
  } catch (e) {
    meta['tokenizer.ggml.model'] = 'llama';
    return meta;
  }

  const model = t.model || {};
  const modelType = model.type || 'BPE';

  if (modelType === 'BPE') {
    meta['tokenizer.ggml.model'] = 'gpt2';
  } else if (modelType === 'Unigram') {
    meta['tokenizer.ggml.model'] = 'llama';
  } else {
    meta['tokenizer.ggml.model'] = 'llama';
  }

  const vocab = model.vocab || {};
  if (typeof vocab === 'object' && vocab !== null && !Array.isArray(vocab)) {
    const sortedVocab = Object.entries(vocab).sort(
      (a: ReturnType<typeof JSON.parse>, b: ReturnType<typeof JSON.parse>) => a[1] - b[1],
    );
    let tokens = sortedVocab.map((x) => x[0]);

    if (vocabSize > 0 && tokens.length !== vocabSize) {
      if (tokens.length < vocabSize) {
        const diff = vocabSize - tokens.length;
        const offset = tokens.length;
        for (let i = 0; i < diff; i++) {
          tokens.push(`[DUMMY_${offset + i}]`);
        }
      } else {
        tokens = tokens.slice(0, vocabSize);
      }
    }

    meta['tokenizer.ggml.tokens'] = tokens;
    meta['tokenizer.ggml.scores'] = Array(tokens.length).fill(0.0);
    meta['tokenizer.ggml.token_type'] = Array(tokens.length).fill(1);
  }

  const merges = model.merges || [];
  if (merges.length > 0) {
    meta['tokenizer.ggml.merges'] = merges;
  }

  meta['tokenizer.ggml.bos_token_id'] =
    t.added_tokens && t.added_tokens.length > 0 ? t.added_tokens[0].id || 0 : 0;
  meta['tokenizer.ggml.eos_token_id'] =
    t.added_tokens && t.added_tokens.length > 0
      ? t.added_tokens[t.added_tokens.length - 1].id || 1
      : 1;
  meta['tokenizer.ggml.unknown_token_id'] = 0;
  meta['tokenizer.ggml.padding_token_id'] = 0;
  meta['tokenizer.ggml.separator_token_id'] = 0;
  meta['tokenizer.ggml.add_bos_token'] = true;
  meta['tokenizer.ggml.add_eos_token'] = false;
  meta['tokenizer.chat_template'] = t.chat_template || '';

  return meta;
}
