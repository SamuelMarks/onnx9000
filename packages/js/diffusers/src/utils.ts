/* eslint-disable */
/**
 * A PCG32 pseudo-random number generator that matches Python implementations
 * to ensure cross-platform seed determinism.
 */
export class PyTorchPCG {
  state: bigint;
  inc: bigint;

  constructor(seed: number | bigint) {
    this.state = BigInt(seed) & 0xffffffffffffffffn;
    this.inc = 1442695040888963407n;
    this.nextUint();
  }

  nextUint(): number {
    const oldstate = this.state;
    this.state = (oldstate * 6364136223846793005n + this.inc) & 0xffffffffffffffffn;
    const xorshifted = Number(((oldstate >> 18n) ^ oldstate) >> 27n);
    const rot = Number(oldstate >> 59n);
    return ((xorshifted >>> rot) | (xorshifted << (-rot & 31))) >>> 0;
  }

  /** Returns a uniform float between 0.0 and 1.0. */
  nextFloat(): number {
    return this.nextUint() / 4294967296.0;
  }
}

/**
 * Generates a uniform tensor [0, 1) natively matching cross-platform PRNG.
 */
export function rand(shape: number[], generator: PyTorchPCG): number[] {
  let size = 1;
  for (const dim of shape) {
    size *= dim;
  }
  const out = new Array(size);
  for (let i = 0; i < size; i++) {
    out[i] = generator.nextFloat();
  }
  return out;
}

/**
 * Generates a standard normal tensor (mean=0, std=1) natively using Box-Muller.
 */
export function randn(shape: number[], generator: PyTorchPCG): number[] {
  let size = 1;
  for (const dim of shape) {
    size *= dim;
  }
  const out: number[] = [];
  const numPairs = Math.ceil(size / 2);
  for (let i = 0; i < numPairs; i++) {
    const u1 = Math.max(generator.nextFloat(), 1e-7);
    const u2 = generator.nextFloat();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
    out.push(z0, z1);
  }
  return out.slice(0, size);
}

export class ProgressBarConfig {
  enabled: boolean = true;
}

export const globalProgressBarConfig = new ProgressBarConfig();

export function setProgressBarConfig(enabled: boolean): void {
  globalProgressBarConfig.enabled = enabled;
}

/**
 * Downloads a file from Hugging Face Hub with IndexedDB caching (mocked for Node/CLI).
 */
export async function fetchHubFile(
  repoId: string,
  filename: string,
): Promise<ReturnType<typeof JSON.parse>> {
  const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}`);
  }
  return await res.json();
}

/**
 * Provides native configuration parsing for `model_index.json`.
 */
export async function parseModelIndex(repoId: string): Promise<ReturnType<typeof JSON.parse>> {
  return await fetchHubFile(repoId, 'model_index.json');
}
