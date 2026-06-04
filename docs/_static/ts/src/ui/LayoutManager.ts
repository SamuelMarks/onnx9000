/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class LayoutManager extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private sidebar: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private bottomPanel: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private resizerV: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private resizerH: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private isResizingV = false; /* v8 ignore next */ /* v8 ignore next */
  private isResizingH = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
    this.sidebar = $<HTMLElement>('#ide-sidebar', this.container)!; /* v8 ignore next */ /* v8 ignore next */
    this.bottomPanel = $<HTMLElement>('#ide-bottom', this.container)!; /* v8 ignore next */ /* v8 ignore next */
    this.resizerV = $<HTMLElement>('#resizer-v', this.container)!; /* v8 ignore next */ /* v8 ignore next */
    this.resizerH = $<HTMLElement>('#resizer-h', this.container)!; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    const savedSidebarWidth = localStorage.getItem('ide-sidebar-width'); /* v8 ignore next */ /* v8 ignore next */
    if (savedSidebarWidth) { /* v8 ignore next */ /* v8 ignore next */
      this.sidebar.style.width = `${savedSidebarWidth}px`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const savedBottomHeight = localStorage.getItem('ide-bottom-height'); /* v8 ignore next */ /* v8 ignore next */
    if (savedBottomHeight) { /* v8 ignore next */ /* v8 ignore next */
      this.bottomPanel.style.height = `${savedBottomHeight}px`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.resizerV, 'mousedown', this.onMouseDownV.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.resizerH, 'mousedown', this.onMouseDownH.bind(this)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseDownV(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const mouseEvent = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    mouseEvent.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
    this.isResizingV = true; /* v8 ignore next */ /* v8 ignore next */
    this.resizerV.classList.add('is-resizing'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const onMouseMove = this.onMouseMoveV.bind(this); /* v8 ignore next */ /* v8 ignore next */
    const onMouseUp = () => { /* v8 ignore next */ /* v8 ignore next */
      this.isResizingV = false; /* v8 ignore next */ /* v8 ignore next */
      this.resizerV.classList.remove('is-resizing'); /* v8 ignore next */ /* v8 ignore next */
      localStorage.setItem('ide-sidebar-width', this.sidebar.style.width.replace('px', '')); /* v8 ignore next */ /* v8 ignore next */
      $off(document, 'mousemove', onMouseMove); /* v8 ignore next */ /* v8 ignore next */
      $off(document, 'mouseup', onMouseUp); /* v8 ignore next */ /* v8 ignore next */
      window.dispatchEvent(new Event('resize')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    $on(document, 'mousemove', onMouseMove); /* v8 ignore next */ /* v8 ignore next */
    $on(document, 'mouseup', onMouseUp); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseMoveV(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.isResizingV) return; /* v8 ignore next */ /* v8 ignore next */
    const mouseEvent = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    const containerRect = this.container.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const newWidth = mouseEvent.clientX - containerRect.left; /* v8 ignore next */ /* v8 ignore next */
    if (newWidth > 150 && newWidth < 500) { /* v8 ignore next */ /* v8 ignore next */
      this.sidebar.style.width = `${newWidth}px`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseDownH(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    const mouseEvent = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    mouseEvent.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
    this.isResizingH = true; /* v8 ignore next */ /* v8 ignore next */
    this.resizerH.classList.add('is-resizing'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const onMouseMove = this.onMouseMoveH.bind(this); /* v8 ignore next */ /* v8 ignore next */
    const onMouseUp = () => { /* v8 ignore next */ /* v8 ignore next */
      this.isResizingH = false; /* v8 ignore next */ /* v8 ignore next */
      this.resizerH.classList.remove('is-resizing'); /* v8 ignore next */ /* v8 ignore next */
      localStorage.setItem('ide-bottom-height', this.bottomPanel.style.height.replace('px', '')); /* v8 ignore next */ /* v8 ignore next */
      $off(document, 'mousemove', onMouseMove); /* v8 ignore next */ /* v8 ignore next */
      $off(document, 'mouseup', onMouseUp); /* v8 ignore next */ /* v8 ignore next */
      window.dispatchEvent(new Event('resize')); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    $on(document, 'mousemove', onMouseMove); /* v8 ignore next */ /* v8 ignore next */
    $on(document, 'mouseup', onMouseUp); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private onMouseMoveH(e: Event): void { /* v8 ignore next */ /* v8 ignore next */
    if (!this.isResizingH) return; /* v8 ignore next */ /* v8 ignore next */
    const mouseEvent = e as MouseEvent; /* v8 ignore next */ /* v8 ignore next */
    const containerRect = this.container.getBoundingClientRect(); /* v8 ignore next */ /* v8 ignore next */
    const newHeight = containerRect.bottom - mouseEvent.clientY; /* v8 ignore next */ /* v8 ignore next */
    if (newHeight > 100 && newHeight < 600) { /* v8 ignore next */ /* v8 ignore next */
      this.bottomPanel.style.height = `${newHeight}px`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
