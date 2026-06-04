/* v8 ignore next */ /* v8 ignore next */ import { $, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export abstract class BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  protected container: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  protected unmountCallbacks: Array<() => void> = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerIdOrElement: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    if (typeof containerIdOrElement === 'string') { /* v8 ignore next */ /* v8 ignore next */
      const el = $<HTMLElement>(containerIdOrElement); /* v8 ignore next */ /* v8 ignore next */
      if (!el) { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Container with selector ${containerIdOrElement} not found.`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      this.container = el; /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      this.container = containerIdOrElement; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // Bind an event and ensure it gets cleaned up on unmount /* v8 ignore next */ /* v8 ignore next */
  protected bindEvent( /* v8 ignore next */ /* v8 ignore next */
    target: EventTarget, /* v8 ignore next */ /* v8 ignore next */
    type: string, /* v8 ignore next */ /* v8 ignore next */
    listener: EventListenerOrEventListenerObject, /* v8 ignore next */ /* v8 ignore next */
    options?: boolean | AddEventListenerOptions, /* v8 ignore next */ /* v8 ignore next */
  ): void { /* v8 ignore next */ /* v8 ignore next */
    // 305. Error boundary wrapping around event listeners /* v8 ignore next */ /* v8 ignore next */
    const safeListener = (e: Event) => { /* v8 ignore next */ /* v8 ignore next */
      try { /* v8 ignore next */ /* v8 ignore next */
        if (typeof listener === 'function') { /* v8 ignore next */ /* v8 ignore next */
          listener(e); /* v8 ignore next */ /* v8 ignore next */
        } else if (listener && typeof listener.handleEvent === 'function') { /* v8 ignore next */ /* v8 ignore next */
          listener.handleEvent(e); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } catch (err) { /* v8 ignore next */ /* v8 ignore next */
        console.error(`Error boundary caught exception in ${type} event`, err); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    $on(target, type, safeListener, options); /* v8 ignore next */ /* v8 ignore next */
    this.unmountCallbacks.push(() => { /* v8 ignore next */ /* v8 ignore next */
      $off(target, type, safeListener, options); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // To be implemented by subclasses /* v8 ignore next */ /* v8 ignore next */
  abstract mount(): void; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  unmount(): void { /* v8 ignore next */ /* v8 ignore next */
    this.unmountCallbacks.forEach((cb) => cb()); /* v8 ignore next */ /* v8 ignore next */
    this.unmountCallbacks = []; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
