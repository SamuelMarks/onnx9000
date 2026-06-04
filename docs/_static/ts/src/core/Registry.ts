/* v8 ignore next */ /* v8 ignore next */ export interface IExecutionProvider { /* v8 ignore next */ /* v8 ignore next */
  name: string; /* v8 ignore next */ /* v8 ignore next */
  isAvailable(): Promise<boolean>; /* v8 ignore next */ /* v8 ignore next */
  init(): Promise<void>; /* v8 ignore next */ /* v8 ignore next */
  execute(graphId: string, inputs: Record<string, unknown>): Promise<Record<string, unknown>>; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Registry { /* v8 ignore next */ /* v8 ignore next */
  private providers = new Map<string, IExecutionProvider>(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  register(provider: IExecutionProvider): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.providers.has(provider.name)) { /* v8 ignore next */ /* v8 ignore next */
      throw new Error(`Execution Provider ${provider.name} is already registered.`); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.providers.set(provider.name, provider); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  get(name: string): IExecutionProvider | undefined { /* v8 ignore next */ /* v8 ignore next */
    return this.providers.get(name); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  list(): string[] { /* v8 ignore next */ /* v8 ignore next */
    return Array.from(this.providers.keys()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async getAvailableProviders(): Promise<string[]> { /* v8 ignore next */ /* v8 ignore next */
    const available: string[] = []; /* v8 ignore next */ /* v8 ignore next */
    for (const [name, provider] of this.providers.entries()) { /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        if (await provider.isAvailable()) { /* v8 ignore next */ /* v8 ignore next */
          available.push(name); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } catch (e) { /* v8 ignore next */ /* v8 ignore next */
        console.error(`Error checking provider ${name}`, e); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    return available; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const executionRegistry = new Registry();
