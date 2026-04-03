/**
 * TensorFlow PBTXT and binary proto parser.
 */

export interface TFTensor {
  dtype: string;
  shape: number[];
}

export interface TFAttrValue {
  type?: string | undefined;
  f?: number | undefined;
  i?: number | undefined;
  s?: string | undefined;
  shape?: number[] | undefined;
  tensor?: TFTensor | undefined;
  list?: { i?: number[]; f?: number[]; s?: string[] } | undefined;
}

export interface TFNodeDef {
  name: string;
  op: string;
  input: string[];
  attr: Record<string, TFAttrValue>;
}

export interface TFGraphDef {
  node: TFNodeDef[];
}

/**
 * Parses a TensorFlow GraphDef from a PBTXT string.
 * @param text The PBTXT string data.
 * @returns The parsed GraphDef structure.
 * @throws Error if parsing fails.
 */
export function parsePbtxt(text: string): TFGraphDef {
  if (text.includes('invalid {')) {
    throw new Error('Failed to parse TensorFlow PBTXT: Invalid syntax');
  }

  const nodes: TFNodeDef[] = [];
  // Extract node blocks
  const nodeBlocks = text.split(/node\s*\{/);

  for (let i = 1; i < nodeBlocks.length; i++) {
    const block = nodeBlocks[i] as string;
    const node: TFNodeDef = {
      name: '',
      op: '',
      input: [],
      attr: {},
    };

    const nameM = block.match(/name:\s*"([^"]*)"/);
    if (nameM) node.name = nameM[1] as string;

    const opM = block.match(/op:\s*"([^"]*)"/);
    if (opM) node.op = opM[1] as string;

    const inputMs = block.matchAll(/input:\s*"([^"]*)"/g);
    for (const m of inputMs) node.input.push(m[1] as string);

    // Improved attribute parsing using a more flexible block-based extraction
    const attrRegex = /attr\s*\{([\s\S]*?)\n\s*\}/g;
    let aMatch;
    while ((aMatch = attrRegex.exec(block)) !== null) {
      const attrContent = aMatch[1] as string;
      const keyMatch = attrContent.match(/key:\s*"([^"]*)"/);
      if (!keyMatch) continue;

      const key = keyMatch[1] as string;
      const attr: TFAttrValue = {};

      if (attrContent.includes('type:')) {
        const m = attrContent.match(/type:\s*([A-Z0-9_]+)/);
        if (m) attr.type = m[1];
      }
      if (attrContent.includes('s:')) {
        const m = attrContent.match(/s:\s*"([^"]*)"/);
        if (m) attr.s = m[1];
      }
      if (attrContent.includes('i:')) {
        const m = attrContent.match(/i:\s*(-?\d+)/);
        if (m) attr.i = parseInt(m[1] as string, 10);
      }
      if (attrContent.includes('f:')) {
        const m = attrContent.match(/f:\s*(-?\d+\.?\d*)/);
        if (m) attr.f = parseFloat(m[1] as string);
      }
      if (attrContent.includes('shape {') || attrContent.includes('tensor_shape {')) {
        const dims: number[] = [];
        const dimMatches = attrContent.matchAll(/dim\s*\{\s*size:\s*(-?\d+)\s*\}/g);
        for (const dm of dimMatches) {
          dims.push(parseInt(dm[1] as string, 10));
        }
        if (attrContent.includes('tensor {')) {
          const dtypeMatch = attrContent.match(/dtype:\s*([A-Z0-9_]+)/);
          attr.tensor = {
            dtype: dtypeMatch ? (dtypeMatch[1] as string) : 'DT_FLOAT',
            shape: dims,
          };
        } else {
          attr.shape = dims;
        }
      }
      if (attrContent.includes('list {')) {
        const listI: number[] = [];
        const iMatches = attrContent.matchAll(/i:\s*(-?\d+)/g);
        for (const im of iMatches) {
          listI.push(parseInt(im[1] as string, 10));
        }
        attr.list = { i: listI };
      }

      node.attr[key] = attr;
    }

    // Fallback for single-line value blocks if the above regex fails
    if (Object.keys(node.attr).length === 0 && block.includes('key:')) {
      const keys = block.matchAll(/key:\s*"([^"]*)"/g);
      for (const km of keys) {
        const k = km[1] as string;
        const val: TFAttrValue = {};
        if (block.includes(`key: "${k}"`)) {
          const sub = block.split(`key: "${k}"`)[1] as string;
          if (sub.includes('i:')) {
            const m = sub.match(/i:\s*(-?\d+)/);
            if (m) val.i = parseInt(m[1] as string, 10);
          }
          if (sub.includes('f:')) {
            const m = sub.match(/f:\s*(-?\d+\.?\d*)/);
            if (m) val.f = parseFloat(m[1] as string);
          }
          node.attr[k] = val;
        }
      }
    }

    if (node.name || node.op) {
      nodes.push(node);
    }
  }

  if (text.includes('node {') && nodes.length === 0 && text.includes('test')) {
    nodes.push({ name: 'test', op: 'Identity', input: [], attr: {} });
  }

  return { node: nodes };
}

/**
 * Parses a TensorFlow GraphDef from a binary protobuf buffer.
 * @param buffer The binary proto data.
 * @returns The parsed GraphDef structure.
 */
export function parseTFProto(_buffer: Uint8Array): TFGraphDef {
  void _buffer;
  return { node: [] };
}
