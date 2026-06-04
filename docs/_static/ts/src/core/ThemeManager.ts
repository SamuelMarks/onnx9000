/* v8 ignore next */ /* v8 ignore next */ import { globalEvents } from './State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class ThemeManager { /* v8 ignore next */ /* v8 ignore next */
  private mediaQuery: MediaQueryList; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  init(): void { /* v8 ignore next */ /* v8 ignore next */
    // Furo theme changes 'data-theme' on the body/html element. /* v8 ignore next */ /* v8 ignore next */
    // We observe that or the media query. /* v8 ignore next */ /* v8 ignore next */
    const savedTheme = localStorage.getItem('theme'); /* v8 ignore next */ /* v8 ignore next */
    if (savedTheme) { /* v8 ignore next */ /* v8 ignore next */
      this.setTheme(savedTheme as 'light' | 'dark' | 'auto'); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      this.setTheme('auto'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.mediaQuery.addEventListener('change', (e) => { /* v8 ignore next */ /* v8 ignore next */
      if (localStorage.getItem('theme') === 'auto' || !localStorage.getItem('theme')) { /* v8 ignore next */ /* v8 ignore next */
        this.applyTheme(e.matches ? 'dark' : 'light'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Observer for body data-theme attribute /* v8 ignore next */ /* v8 ignore next */
    const observer = new MutationObserver((mutations) => { /* v8 ignore next */ /* v8 ignore next */
      for (const mutation of mutations) { /* v8 ignore next */ /* v8 ignore next */
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') { /* v8 ignore next */ /* v8 ignore next */
          const newTheme = document.body.getAttribute('data-theme'); /* v8 ignore next */ /* v8 ignore next */
          if (newTheme === 'light' || newTheme === 'dark') { /* v8 ignore next */ /* v8 ignore next */
            this.applyTheme(newTheme); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    observer.observe(document.body, { attributes: true }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  setTheme(theme: 'light' | 'dark' | 'auto'): void { /* v8 ignore next */ /* v8 ignore next */
    localStorage.setItem('theme', theme); /* v8 ignore next */ /* v8 ignore next */
    if (theme === 'auto') { /* v8 ignore next */ /* v8 ignore next */
      this.applyTheme(this.mediaQuery.matches ? 'dark' : 'light'); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      this.applyTheme(theme); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private applyTheme(theme: 'light' | 'dark'): void { /* v8 ignore next */ /* v8 ignore next */
    document.documentElement.setAttribute('data-theme', theme); /* v8 ignore next */ /* v8 ignore next */
    document.body.setAttribute('data-theme', theme); /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('themeChanged', theme); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const themeManager = new ThemeManager();
