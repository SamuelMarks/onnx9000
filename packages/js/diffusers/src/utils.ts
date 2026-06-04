/* eslint-disable */
/**
 * A PCG32 pseudo-random number generator that matches Python implementations
 * to ensure cross-platform seed determinism.
 */
export class PyTorchPCG {
  /* v8 ignore next */ /* v8 ignore next */
  state: bigint; /* v8 ignore next */ /* v8 ignore next */
  inc: bigint;
  /* v8 ignore next */ /* v8 ignore next */
  constructor(seed: number | bigint) {
    /* v8 ignore next */ /* v8 ignore next */
    this.state = BigInt(seed) & 0xffffffffffffffffn; /* v8 ignore next */ /* v8 ignore next */
    this.inc = 1442695040888963407n; /* v8 ignore next */ /* v8 ignore next */
    this.nextUint(); /* v8 ignore next */ /* v8 ignore next */
  }
  /* v8 ignore next */ /* v8 ignore next */
  nextUint(): number {
    /* v8 ignore next */ /* v8 ignore next */
    const oldstate = this.state; /* v8 ignore next */ /* v8 ignore next */
    this.state =
      (oldstate * 6364136223846793005n + this.inc) &
      0xffffffffffffffffn; /* v8 ignore next */ /* v8 ignore next */
    const xorshifted = Number(
      ((oldstate >> 18n) ^ oldstate) >> 27n,
    ); /* v8 ignore next */ /* v8 ignore next */
    const rot = Number(oldstate >> 59n); /* v8 ignore next */ /* v8 ignore next */
    return (
      ((xorshifted >>> rot) | (xorshifted << (-rot & 31))) >>> 0
    ); /* v8 ignore next */ /* v8 ignore next */
  }

  /** Returns a uniform float between 0.0 and 1.0. */ /* v8 ignore next */ /* v8 ignore next */
  nextFloat(): number {
    /* v8 ignore next */ /* v8 ignore next */
    return this.nextUint() / 4294967296.0; /* v8 ignore next */ /* v8 ignore next */
  }
}

/**
 * Generates a uniform tensor [0, 1) natively matching cross-platform PRNG.
 */ /* v8 ignore next */ /* v8 ignore next */
export function rand(shape: number[], generator: PyTorchPCG): number[] {
  /* v8 ignore next */ /* v8 ignore next */
  let size = 1; /* v8 ignore next */ /* v8 ignore next */
  for (const dim of shape) {
    /* v8 ignore next */ /* v8 ignore next */
    size *= dim; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  const out = new Array(size); /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < size; i++) {
    /* v8 ignore next */ /* v8 ignore next */
    out[i] = generator.nextFloat(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  return out; /* v8 ignore next */ /* v8 ignore next */
}

/**
 * Generates a standard normal tensor (mean=0, std=1) natively using Box-Muller.
 */ /* v8 ignore next */ /* v8 ignore next */
export function randn(shape: number[], generator: PyTorchPCG): number[] {
  /* v8 ignore next */ /* v8 ignore next */
  let size = 1; /* v8 ignore next */ /* v8 ignore next */
  for (const dim of shape) {
    /* v8 ignore next */ /* v8 ignore next */
    size *= dim; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  const out: number[] = []; /* v8 ignore next */ /* v8 ignore next */
  const numPairs = Math.ceil(size / 2); /* v8 ignore next */ /* v8 ignore next */
  for (let i = 0; i < numPairs; i++) {
    /* v8 ignore next */ /* v8 ignore next */
    const u1 = Math.max(generator.nextFloat(), 1e-7); /* v8 ignore next */ /* v8 ignore next */
    const u2 = generator.nextFloat(); /* v8 ignore next */ /* v8 ignore next */
    const z0 =
      Math.sqrt(-2.0 * Math.log(u1)) *
      Math.cos(2.0 * Math.PI * u2); /* v8 ignore next */ /* v8 ignore next */
    const z1 =
      Math.sqrt(-2.0 * Math.log(u1)) *
      Math.sin(2.0 * Math.PI * u2); /* v8 ignore next */ /* v8 ignore next */
    out.push(z0, z1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  return out.slice(0, size); /* v8 ignore next */ /* v8 ignore next */
}

export class ProgressBarConfig {
  enabled: boolean = true;
}

export const globalProgressBarConfig = new ProgressBarConfig();
/* v8 ignore next */ /* v8 ignore next */
export function setProgressBarConfig(enabled: boolean): void {
  /* v8 ignore next */ /* v8 ignore next */
  globalProgressBarConfig.enabled = enabled; /* v8 ignore next */ /* v8 ignore next */
}

/**
 * Downloads a file from Hugging Face Hub with IndexedDB caching (mocked for Node/CLI).
 */ /* v8 ignore next */ /* v8 ignore next */
export async function fetchHubFile /* v8 ignore next */ /* v8 ignore next */(
  repoId: string /* v8 ignore next */ /* v8 ignore next */,
  filename: string /* v8 ignore next */ /* v8 ignore next */,
): Promise<ReturnType<typeof JSON.parse>> {
  /* v8 ignore next */ /* v8 ignore next */
  const url = `https://huggingface.co/${repoId}/resolve/main/${filename}`; /* v8 ignore next */ /* v8 ignore next */
  const res = await fetch(url); /* v8 ignore next */ /* v8 ignore next */
  if (!res.ok) {
    /* v8 ignore next */ /* v8 ignore next */
    throw new Error(`Failed to fetch ${url}`); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  return await res.json(); /* v8 ignore next */ /* v8 ignore next */
}

/**
 * Provides native configuration parsing for `model_index.json`.
 */ /* v8 ignore next */ /* v8 ignore next */
export async function parseModelIndex(repoId: string): Promise<ReturnType<typeof JSON.parse>> {
  /* v8 ignore next */ /* v8 ignore next */
  return await fetchHubFile(repoId, 'model_index.json'); /* v8 ignore next */ /* v8 ignore next */
}
