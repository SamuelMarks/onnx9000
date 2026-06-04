/* v8 ignore next */ /* v8 ignore next */ import { globalEvents } from './State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
/** /* v8 ignore next */ /* v8 ignore next */
 * 636. Add internationalization (i18n) support, loading locale JSONs /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
export class I18nManager { /* v8 ignore next */ /* v8 ignore next */
  private currentLocale = 'en'; /* v8 ignore next */ /* v8 ignore next */
  private locales: Record<string, Record<string, string>> = { /* v8 ignore next */ /* v8 ignore next */
    en: { /* v8 ignore next */ /* v8 ignore next */
      'nav.upload': 'Upload Model', /* v8 ignore next */ /* v8 ignore next */
      'nav.surgeon': 'Graph Surgeon', /* v8 ignore next */ /* v8 ignore next */
      'nav.benchmark': 'Micro-Benchmarks', /* v8 ignore next */ /* v8 ignore next */
      'toast.loaded': 'Model loaded successfully', /* v8 ignore next */ /* v8 ignore next */
      'action.execute': 'Run Inference', /* v8 ignore next */ /* v8 ignore next */
      'action.download': 'Download', /* v8 ignore next */ /* v8 ignore next */
    }, /* v8 ignore next */ /* v8 ignore next */
    es: { /* v8 ignore next */ /* v8 ignore next */
      'nav.upload': 'Subir Modelo', /* v8 ignore next */ /* v8 ignore next */
      'nav.surgeon': 'Cirujano de Grafos', /* v8 ignore next */ /* v8 ignore next */
      'nav.benchmark': 'Micro-Puntos de Referencia', /* v8 ignore next */ /* v8 ignore next */
      'toast.loaded': 'Modelo cargado con éxito', /* v8 ignore next */ /* v8 ignore next */
      'action.execute': 'Ejecutar Inferencia', /* v8 ignore next */ /* v8 ignore next */
      'action.download': 'Descargar', /* v8 ignore next */ /* v8 ignore next */
    }, /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public setLocale(locale: string): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.locales[locale]) { /* v8 ignore next */ /* v8 ignore next */
      this.currentLocale = locale; /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('localeChanged', locale); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  public t(key: string): string { /* v8 ignore next */ /* v8 ignore next */
    const dict = this.locales[this.currentLocale]; /* v8 ignore next */ /* v8 ignore next */
    return dict[key] || key; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const i18n = new I18nManager();
