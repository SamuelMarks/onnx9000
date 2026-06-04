/* v8 ignore next */ /* v8 ignore next */ export type Listener<T> = (val: T) => void; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class State<T> { /* v8 ignore next */ /* v8 ignore next */
  private value: T; /* v8 ignore next */ /* v8 ignore next */
  private listeners: Listener<T>[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(initialValue: T) { /* v8 ignore next */ /* v8 ignore next */
    this.value = initialValue; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  get(): T { /* v8 ignore next */ /* v8 ignore next */
    return this.value; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  set(newValue: T | ((prev: T) => T)): void { /* v8 ignore next */ /* v8 ignore next */
    if (typeof newValue === 'function') { /* v8 ignore next */ /* v8 ignore next */
      this.value = (newValue as (prev: T) => T)(this.value); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      this.value = newValue; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.notify(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  subscribe(listener: Listener<T>): () => void { /* v8 ignore next */ /* v8 ignore next */
    this.listeners.push(listener); /* v8 ignore next */ /* v8 ignore next */
    return () => { /* v8 ignore next */ /* v8 ignore next */
      this.listeners = this.listeners.filter((l) => l !== listener); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private notify(): void { /* v8 ignore next */ /* v8 ignore next */
    for (const listener of this.listeners) { /* v8 ignore next */ /* v8 ignore next */
      listener(this.value); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class PubSub { /* v8 ignore next */ /* v8 ignore next */
  private events: Map<string, Function[]> = new Map(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  on(event: string, callback: Function): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.events.has(event)) { /* v8 ignore next */ /* v8 ignore next */
      this.events.set(event, []); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.events.get(event)!.push(callback); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  off(event: string, callback: Function): void { /* v8 ignore next */ /* v8 ignore next */
    const callbacks = this.events.get(event); /* v8 ignore next */ /* v8 ignore next */
    if (callbacks) { /* v8 ignore next */ /* v8 ignore next */
      this.events.set( /* v8 ignore next */ /* v8 ignore next */
        event, /* v8 ignore next */ /* v8 ignore next */
        callbacks.filter((cb) => cb !== callback), /* v8 ignore next */ /* v8 ignore next */
      ); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any /* v8 ignore next */ /* v8 ignore next */
  emit(event: string, ...args: any[]): void { /* v8 ignore next */ /* v8 ignore next */
    const callbacks = this.events.get(event); /* v8 ignore next */ /* v8 ignore next */
    if (callbacks) { /* v8 ignore next */ /* v8 ignore next */
      for (const callback of callbacks) { /* v8 ignore next */ /* v8 ignore next */
        callback(...args); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const globalEvents = new PubSub(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const isOfflineMode = new State<boolean>(true); /* v8 ignore next */ /* v8 ignore next */
export const isDistributedMode = new State<boolean>(false);
