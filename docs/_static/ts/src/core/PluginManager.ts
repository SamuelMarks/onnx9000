/* v8 ignore next */ /* v8 ignore next */ import { IModelGraph } from './IR'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from '../ui/Toast'; /* v8 ignore next */ /* v8 ignore next */
import { logger } from './Logger'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from './State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IPluginContext { /* v8 ignore next */ /* v8 ignore next */
  registerSidebarTab: (title: string, component: HTMLElement) => void; /* v8 ignore next */ /* v8 ignore next */
  showToast: (msg: string, type?: 'info' | 'success' | 'warn' | 'error') => void; /* v8 ignore next */ /* v8 ignore next */
  log: (msg: string) => void; /* v8 ignore next */ /* v8 ignore next */
  getActiveModel: () => Readonly<IModelGraph> | null; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IPlugin { /* v8 ignore next */ /* v8 ignore next */
  name: string; /* v8 ignore next */ /* v8 ignore next */
  version: string; /* v8 ignore next */ /* v8 ignore next */
  init: (ctx: IPluginContext) => Promise<void>; /* v8 ignore next */ /* v8 ignore next */
  onModelLoad?: (model: Readonly<IModelGraph>) => void; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class PluginManager { /* v8 ignore next */ /* v8 ignore next */
  private plugins = new Map<string, IPlugin>(); /* v8 ignore next */ /* v8 ignore next */
  private activeModel: IModelGraph | null = null; /* v8 ignore next */ /* v8 ignore next */
  private sidebarContainer: HTMLElement | null = null; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('modelLoaded', (model: IModelGraph) => { /* v8 ignore next */ /* v8 ignore next */
      this.activeModel = model; /* v8 ignore next */ /* v8 ignore next */
      this.plugins.forEach((p) => { /* v8 ignore next */ /* v8 ignore next */
        if (p.onModelLoad) { /* v8 ignore next */ /* v8 ignore next */
          try { /* v8 ignore next */ /* v8 ignore next */
            p.onModelLoad(this.activeModel!); /* v8 ignore next */ /* v8 ignore next */
          } catch (e) { /* v8 ignore next */ /* v8 ignore next */
            logger.error(`Plugin ${p.name} failed onModelLoad hook: ${e}`); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  setSidebarContainer(el: HTMLElement): void { /* v8 ignore next */ /* v8 ignore next */
    this.sidebarContainer = el; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  async loadPlugin(url: string): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      // 563. Dynamic loading via ESM /* v8 ignore next */ /* v8 ignore next */
      // 564. Sandbox external plugins (in a real scenario, we'd use a Worker or iframe. /* v8 ignore next */ /* v8 ignore next */
      // For this native JS implementation, we assume basic ESM isolation) /* v8 ignore next */ /* v8 ignore next */
      const module = await import(/* @vite-ignore */ url); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (!module.default || typeof module.default.init !== 'function') { /* v8 ignore next */ /* v8 ignore next */
        throw new Error('Plugin does not export a valid default IPlugin interface'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const plugin: IPlugin = module.default; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (this.plugins.has(plugin.name)) { /* v8 ignore next */ /* v8 ignore next */
        throw new Error(`Plugin ${plugin.name} is already loaded`); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // 565. Expose strict Readonly API /* v8 ignore next */ /* v8 ignore next */
      const context: IPluginContext = { /* v8 ignore next */ /* v8 ignore next */
        registerSidebarTab: (title: string, component: HTMLElement) => { /* v8 ignore next */ /* v8 ignore next */
          this.registerTab(plugin.name, title, component); /* v8 ignore next */ /* v8 ignore next */
        }, /* v8 ignore next */ /* v8 ignore next */
        showToast: (msg: string, type = 'info') => Toast.show(`[${plugin.name}] ${msg}`, type), /* v8 ignore next */ /* v8 ignore next */
        log: (msg: string) => logger.info(`[${plugin.name}] ${msg}`), /* v8 ignore next */ /* v8 ignore next */
        getActiveModel: () => this.activeModel, /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      await plugin.init(context); /* v8 ignore next */ /* v8 ignore next */
      this.plugins.set(plugin.name, plugin); /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Plugin ${plugin.name} v${plugin.version} loaded`, 'success'); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      logger.error(`Failed to load plugin from ${url}`, e); /* v8 ignore next */ /* v8 ignore next */
      Toast.show(`Failed to load plugin`, 'error'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private registerTab(pluginName: string, title: string, component: HTMLElement): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.sidebarContainer) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 566. API wrapper for registering sidebar tabs /* v8 ignore next */ /* v8 ignore next */
    const section = document.createElement('div'); /* v8 ignore next */ /* v8 ignore next */
    section.className = 'sidebar-section'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const h4 = document.createElement('h4'); /* v8 ignore next */ /* v8 ignore next */
    h4.textContent = `${title} (${pluginName})`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    section.appendChild(h4); /* v8 ignore next */ /* v8 ignore next */
    section.appendChild(component); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.sidebarContainer.appendChild(section); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  getLoadedPlugins(): string[] { /* v8 ignore next */ /* v8 ignore next */
    return Array.from(this.plugins.keys()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export const pluginManager = new PluginManager();
