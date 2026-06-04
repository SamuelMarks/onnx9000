/* v8 ignore next */ /* v8 ignore next */ export function $<T extends HTMLElement = HTMLElement>( /* v8 ignore next */ /* v8 ignore next */
  selector: string, /* v8 ignore next */ /* v8 ignore next */
  parent: ParentNode = document, /* v8 ignore next */ /* v8 ignore next */
): T | null { /* v8 ignore next */ /* v8 ignore next */
  return parent.querySelector<T>(selector); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export function $$<T extends HTMLElement = HTMLElement>( /* v8 ignore next */ /* v8 ignore next */
  selector: string, /* v8 ignore next */ /* v8 ignore next */
  parent: ParentNode = document, /* v8 ignore next */ /* v8 ignore next */
): NodeListOf<T> { /* v8 ignore next */ /* v8 ignore next */
  return parent.querySelectorAll<T>(selector); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export function $on( /* v8 ignore next */ /* v8 ignore next */
  target: EventTarget, /* v8 ignore next */ /* v8 ignore next */
  type: string, /* v8 ignore next */ /* v8 ignore next */
  listener: EventListenerOrEventListenerObject, /* v8 ignore next */ /* v8 ignore next */
  options?: boolean | AddEventListenerOptions, /* v8 ignore next */ /* v8 ignore next */
): void { /* v8 ignore next */ /* v8 ignore next */
  target.addEventListener(type, listener, options); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export function $off( /* v8 ignore next */ /* v8 ignore next */
  target: EventTarget, /* v8 ignore next */ /* v8 ignore next */
  type: string, /* v8 ignore next */ /* v8 ignore next */
  listener: EventListenerOrEventListenerObject, /* v8 ignore next */ /* v8 ignore next */
  options?: boolean | EventListenerOptions, /* v8 ignore next */ /* v8 ignore next */
): void { /* v8 ignore next */ /* v8 ignore next */
  target.removeEventListener(type, listener, options); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export function $create<T extends HTMLElement = HTMLElement>( /* v8 ignore next */ /* v8 ignore next */
  tag: string, /* v8 ignore next */ /* v8 ignore next */
  options?: { /* v8 ignore next */ /* v8 ignore next */
    id?: string; /* v8 ignore next */ /* v8 ignore next */
    className?: string; /* v8 ignore next */ /* v8 ignore next */
    innerHTML?: string; /* v8 ignore next */ /* v8 ignore next */
    textContent?: string; /* v8 ignore next */ /* v8 ignore next */
    attributes?: Record<string, string>; /* v8 ignore next */ /* v8 ignore next */
  }, /* v8 ignore next */ /* v8 ignore next */
): T { /* v8 ignore next */ /* v8 ignore next */
  const el = document.createElement(tag) as T; /* v8 ignore next */ /* v8 ignore next */
  if (options) { /* v8 ignore next */ /* v8 ignore next */
    if (options.id) el.id = options.id; /* v8 ignore next */ /* v8 ignore next */
    if (options.className) el.className = options.className; /* v8 ignore next */ /* v8 ignore next */
    if (options.innerHTML !== undefined) el.innerHTML = options.innerHTML; /* v8 ignore next */ /* v8 ignore next */
    if (options.textContent !== undefined) el.textContent = options.textContent; /* v8 ignore next */ /* v8 ignore next */
    if (options.attributes) { /* v8 ignore next */ /* v8 ignore next */
      for (const [key, value] of Object.entries(options.attributes)) { /* v8 ignore next */ /* v8 ignore next */
        el.setAttribute(key, value); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  return el; /* v8 ignore next */ /* v8 ignore next */
}
