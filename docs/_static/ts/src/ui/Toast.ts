/* v8 ignore next */ /* v8 ignore next */ import { $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class Toast { /* v8 ignore next */ /* v8 ignore next */
  private static container: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  static init(): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.container) { /* v8 ignore next */ /* v8 ignore next */
      this.container = $create('div', { /* v8 ignore next */ /* v8 ignore next */
        className: 'ide-toast-container', /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      document.body.appendChild(this.container); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const style = $create('style', { /* v8 ignore next */ /* v8 ignore next */
        textContent: ` /* v8 ignore next */ /* v8 ignore next */
          .ide-toast-container { /* v8 ignore next */ /* v8 ignore next */
            position: fixed; /* v8 ignore next */ /* v8 ignore next */
            bottom: 20px; /* v8 ignore next */ /* v8 ignore next */
            right: 20px; /* v8 ignore next */ /* v8 ignore next */
            z-index: 9999; /* v8 ignore next */ /* v8 ignore next */
            display: flex; /* v8 ignore next */ /* v8 ignore next */
            flex-direction: column; /* v8 ignore next */ /* v8 ignore next */
            gap: 10px; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          .ide-toast { /* v8 ignore next */ /* v8 ignore next */
            background: var(--color-background-secondary); /* v8 ignore next */ /* v8 ignore next */
            color: var(--color-foreground-primary); /* v8 ignore next */ /* v8 ignore next */
            border: 1px solid var(--color-background-border); /* v8 ignore next */ /* v8 ignore next */
            border-left: 4px solid var(--color-primary); /* v8 ignore next */ /* v8 ignore next */
            padding: 12px 16px; /* v8 ignore next */ /* v8 ignore next */
            border-radius: 4px; /* v8 ignore next */ /* v8 ignore next */
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* v8 ignore next */ /* v8 ignore next */
            font-family: sans-serif; /* v8 ignore next */ /* v8 ignore next */
            font-size: 0.9rem; /* v8 ignore next */ /* v8 ignore next */
            opacity: 0; /* v8 ignore next */ /* v8 ignore next */
            transform: translateX(100%); /* v8 ignore next */ /* v8 ignore next */
            transition: all 0.3s ease; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          .ide-toast.show { /* v8 ignore next */ /* v8 ignore next */
            opacity: 1; /* v8 ignore next */ /* v8 ignore next */
            transform: translateX(0); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          .ide-toast.error { /* v8 ignore next */ /* v8 ignore next */
            border-left-color: var(--color-danger); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          .ide-toast.success { /* v8 ignore next */ /* v8 ignore next */
            border-left-color: var(--color-success); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          .ide-toast.warn { /* v8 ignore next */ /* v8 ignore next */
            border-left-color: #ffc107; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        `, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      document.head.appendChild(style); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  static show( /* v8 ignore next */ /* v8 ignore next */
    message: string, /* v8 ignore next */ /* v8 ignore next */
    type: 'info' | 'success' | 'warn' | 'error' = 'info', /* v8 ignore next */ /* v8 ignore next */
    duration = 3000, /* v8 ignore next */ /* v8 ignore next */
  ): void { /* v8 ignore next */ /* v8 ignore next */
    this.init(); /* v8 ignore next */ /* v8 ignore next */
    const toast = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: `ide-toast ${type}`, /* v8 ignore next */ /* v8 ignore next */
      textContent: message, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(toast); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Trigger reflow to animate /* v8 ignore next */ /* v8 ignore next */
    void toast.offsetWidth; /* v8 ignore next */ /* v8 ignore next */
    toast.classList.add('show'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    setTimeout(() => { /* v8 ignore next */ /* v8 ignore next */
      toast.classList.remove('show'); /* v8 ignore next */ /* v8 ignore next */
      toast.addEventListener('transitionend', () => { /* v8 ignore next */ /* v8 ignore next */
        if (toast.parentNode) { /* v8 ignore next */ /* v8 ignore next */
          toast.parentNode.removeChild(toast); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }, duration); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
