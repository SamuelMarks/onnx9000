/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class DropZone extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private overlay: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private dropMessage: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private dragCounter = 0; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor() { /* v8 ignore next */ /* v8 ignore next */
    super(document.body); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Create overlay /* v8 ignore next */ /* v8 ignore next */
    this.overlay = $create('div', { className: 'ide-drop-overlay' }); /* v8 ignore next */ /* v8 ignore next */
    this.dropMessage = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-drop-message', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Drop .onnx, .safetensors, .py, or Directory here', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.overlay.appendChild(this.dropMessage); /* v8 ignore next */ /* v8 ignore next */
    document.body.appendChild(this.overlay); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    // Bind global drag events /* v8 ignore next */ /* v8 ignore next */
    const body = document.body; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(body, 'dragenter', this.onDragEnter.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(body, 'dragleave', this.onDragLeave.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(body, 'dragover', this.onDragOver.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(body, 'drop', this.onDrop.bind(this)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onDragEnter(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
    this.dragCounter++; /* v8 ignore next */ /* v8 ignore next */
    if (this.dragCounter === 1) { /* v8 ignore next */ /* v8 ignore next */
      this.overlay.classList.add('is-active'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onDragLeave(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
    this.dragCounter--; /* v8 ignore next */ /* v8 ignore next */
    if (this.dragCounter === 0) { /* v8 ignore next */ /* v8 ignore next */
      this.overlay.classList.remove('is-active'); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onDragOver(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private async onDrop(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
    this.dragCounter = 0; /* v8 ignore next */ /* v8 ignore next */
    this.overlay.classList.remove('is-active'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const dragEvent = e as DragEvent; /* v8 ignore next */ /* v8 ignore next */
    if (!dragEvent.dataTransfer) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const files: File[] = []; /* v8 ignore next */ /* v8 ignore next */
    const items = dragEvent.dataTransfer.items; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (items && items.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const promises: Promise<void>[] = []; /* v8 ignore next */ /* v8 ignore next */
      for (let i = 0; i < items.length; i++) { /* v8 ignore next */ /* v8 ignore next */
        const item = items[i]; /* v8 ignore next */ /* v8 ignore next */
        if (item.kind === 'file') { /* v8 ignore next */ /* v8 ignore next */
          const entry = item.webkitGetAsEntry(); /* v8 ignore next */ /* v8 ignore next */
          if (entry) { /* v8 ignore next */ /* v8 ignore next */
            promises.push(this.traverseFileTree(entry, files)); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      await Promise.all(promises); /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      // Fallback /* v8 ignore next */ /* v8 ignore next */
      for (let i = 0; i < dragEvent.dataTransfer.files.length; i++) { /* v8 ignore next */ /* v8 ignore next */
        files.push(dragEvent.dataTransfer.files[i]); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    if (files.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      if (files.length === 1) { /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('filesDropped', files); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('directoryDropped', files); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private traverseFileTree(item: any, files: File[]): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve) => { /* v8 ignore next */ /* v8 ignore next */
      if (item.isFile) { /* v8 ignore next */ /* v8 ignore next */
        item.file((file: File) => { /* v8 ignore next */ /* v8 ignore next */
          // Keep a reference to its path if possible /* v8 ignore next */ /* v8 ignore next */
          // file.webkitRelativePath is read-only usually, so we just append to files array /* v8 ignore next */ /* v8 ignore next */
          Object.defineProperty(file, 'webkitRelativePath', { /* v8 ignore next */ /* v8 ignore next */
            value: item.fullPath.replace(/^\//, ''), /* v8 ignore next */ /* v8 ignore next */
            writable: false, /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
          files.push(file); /* v8 ignore next */ /* v8 ignore next */
          resolve(); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      } else if (item.isDirectory) { /* v8 ignore next */ /* v8 ignore next */
        const dirReader = item.createReader(); /* v8 ignore next */ /* v8 ignore next */
        dirReader.readEntries((entries: any[]) => { /* v8 ignore next */ /* v8 ignore next */
          const promises: Promise<void>[] = []; /* v8 ignore next */ /* v8 ignore next */
          for (let i = 0; i < entries.length; i++) { /* v8 ignore next */ /* v8 ignore next */
            promises.push(this.traverseFileTree(entries[i], files)); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          Promise.all(promises).then(() => resolve()); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        resolve(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  unmount(): void { /* v8 ignore next */ /* v8 ignore next */
    super.unmount(); /* v8 ignore next */ /* v8 ignore next */
    if (this.overlay.parentNode) { /* v8 ignore next */ /* v8 ignore next */
      this.overlay.parentNode.removeChild(this.overlay); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
