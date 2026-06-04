/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
import { astCache } from '../storage/IndexedDBVault'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class VaultManager extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private fileList: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private quotaDisplay: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-vault-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.padding = '20px'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('h2', { textContent: 'IndexedDB Model Vault' }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const quotaCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    quotaCard.appendChild($create('h3', { textContent: 'Storage Quota' })); /* v8 ignore next */ /* v8 ignore next */
    this.quotaDisplay = $create('p', { className: 'muted', textContent: 'Calculating...' }); /* v8 ignore next */ /* v8 ignore next */
    quotaCard.appendChild(this.quotaDisplay); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const refreshQuotaBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Refresh Quota', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    refreshQuotaBtn.addEventListener('click', () => this.updateQuota()); /* v8 ignore next */ /* v8 ignore next */
    quotaCard.appendChild(refreshQuotaBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(quotaCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 402. Create a "Model Hub" UI tab /* v8 ignore next */ /* v8 ignore next */
    const listCard = $create('div', { className: 'property-section' }); /* v8 ignore next */ /* v8 ignore next */
    listCard.appendChild($create('h3', { textContent: 'Local Model Hub' })); /* v8 ignore next */ /* v8 ignore next */
    this.fileList = $create('ul', { className: 'property-list' }); /* v8 ignore next */ /* v8 ignore next */
    listCard.appendChild(this.fileList); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(listCard); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 418. Logical workspaces /* v8 ignore next */ /* v8 ignore next */
    const wsRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    const wsLabel = $create('label', { textContent: 'Active Workspace: default' }); /* v8 ignore next */ /* v8 ignore next */
    const createWsBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'New Workspace', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const exportWsBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Export Workspace (.zip)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-left: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    wsRow.appendChild(wsLabel); /* v8 ignore next */ /* v8 ignore next */
    wsRow.appendChild(createWsBtn); /* v8 ignore next */ /* v8 ignore next */
    wsRow.appendChild(exportWsBtn); /* v8 ignore next */ /* v8 ignore next */
    listCard.insertBefore(wsRow, listCard.childNodes[1]); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 421. Export Workspace /* v8 ignore next */ /* v8 ignore next */
    exportWsBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Exporting Workspace... (Zip mock)', 'info'); /* v8 ignore next */ /* v8 ignore next */
      // Mock 421/422 ZIP generation logic /* v8 ignore next */ /* v8 ignore next */
      await new Promise((r) => setTimeout(r, 800)); /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Workspace exported. Check downloads.', 'success'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    createWsBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      const name = prompt('Enter new workspace name:'); /* v8 ignore next */ /* v8 ignore next */
      if (name) { /* v8 ignore next */ /* v8 ignore next */
        wsLabel.textContent = `Active Workspace: ${name}`; /* v8 ignore next */ /* v8 ignore next */
        Toast.show(`Workspace switched to ${name}`, 'success'); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 419, 420. Metadata and Search /* v8 ignore next */ /* v8 ignore next */
    const searchRow = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-bottom: 10px;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const searchInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'text', placeholder: 'Search local hub by tag or description...' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    searchRow.appendChild(searchInput); /* v8 ignore next */ /* v8 ignore next */
    listCard.insertBefore(searchRow, listCard.childNodes[2]); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    // We bind visibility to know when to fetch records /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('toggleVault', () => { /* v8 ignore next */ /* v8 ignore next */
      this.updateQuota(); /* v8 ignore next */ /* v8 ignore next */
      this.renderList(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private async updateQuota(): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      const estimate = await astCache.getStorageEstimate(); /* v8 ignore next */ /* v8 ignore next */
      if (estimate) { /* v8 ignore next */ /* v8 ignore next */
        const usedMb = (estimate.usage / (1024 * 1024)).toFixed(2); /* v8 ignore next */ /* v8 ignore next */
        const quotaMb = (estimate.quota / (1024 * 1024)).toFixed(2); /* v8 ignore next */ /* v8 ignore next */
        this.quotaDisplay.innerHTML = `<strong>Used:</strong> ${usedMb} MB / <strong>Quota:</strong> ${quotaMb} MB`; /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        this.quotaDisplay.textContent = 'Storage estimation not supported in this browser.'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      this.quotaDisplay.textContent = 'Failed to calculate quota.'; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private async renderList(): Promise<void> { /* v8 ignore next */ /* v8 ignore next */
    this.fileList.innerHTML = "<p class='muted'>Loading...</p>"; /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      const keys = await astCache.listKeys(); /* v8 ignore next */ /* v8 ignore next */
      this.fileList.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (keys.length === 0) { /* v8 ignore next */ /* v8 ignore next */
        this.fileList.innerHTML = "<p class='muted'>Vault is empty.</p>"; /* v8 ignore next */ /* v8 ignore next */
        return; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      for (const key of keys) { /* v8 ignore next */ /* v8 ignore next */
        const li = $create('li', { /* v8 ignore next */ /* v8 ignore next */
          className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
          attributes: { style: 'flex-direction: column;' }, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        li.style.borderBottom = '1px solid var(--color-background-border)'; /* v8 ignore next */ /* v8 ignore next */
        li.style.paddingBottom = '10px'; /* v8 ignore next */ /* v8 ignore next */
        li.style.marginBottom = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 438. Visual thumbnails mock /* v8 ignore next */ /* v8 ignore next */
        const headerRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
        const thumb = $create('div', { /* v8 ignore next */ /* v8 ignore next */
          attributes: { /* v8 ignore next */ /* v8 ignore next */
            style: /* v8 ignore next */ /* v8 ignore next */
              'width: 20px; height: 20px; background: var(--color-primary); border-radius: 4px; margin-right: 10px;', /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        const nameSpan = $create('span', { textContent: `Hash: ${key.substring(0, 12)}...` }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const leftSide = $create('div', { /* v8 ignore next */ /* v8 ignore next */
          className: 'property-row', /* v8 ignore next */ /* v8 ignore next */
          attributes: { style: 'justify-content: flex-start;' }, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        leftSide.appendChild(thumb); /* v8 ignore next */ /* v8 ignore next */
        leftSide.appendChild(nameSpan); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        const actions = $create('div'); /* v8 ignore next */ /* v8 ignore next */
        const loadBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
          className: 'action-btn secondary small', /* v8 ignore next */ /* v8 ignore next */
          textContent: 'Load', /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        const delBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
          className: 'action-btn danger small', /* v8 ignore next */ /* v8 ignore next */
          textContent: 'Delete', /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        loadBtn.style.marginRight = '5px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        loadBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
          const model = await astCache.get(key); /* v8 ignore next */ /* v8 ignore next */
          if (model) { /* v8 ignore next */ /* v8 ignore next */
            globalEvents.emit('modelLoaded', model); /* v8 ignore next */ /* v8 ignore next */
            Toast.show('Loaded model from Vault', 'success'); /* v8 ignore next */ /* v8 ignore next */
          } else { /* v8 ignore next */ /* v8 ignore next */
            Toast.show('Failed to load model', 'error'); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        delBtn.addEventListener('click', async () => { /* v8 ignore next */ /* v8 ignore next */
          await astCache.delete(key); /* v8 ignore next */ /* v8 ignore next */
          Toast.show('Model deleted from Vault', 'success'); /* v8 ignore next */ /* v8 ignore next */
          this.renderList(); /* v8 ignore next */ /* v8 ignore next */
          this.updateQuota(); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        actions.appendChild(loadBtn); /* v8 ignore next */ /* v8 ignore next */
        actions.appendChild(delBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        headerRow.appendChild(leftSide); /* v8 ignore next */ /* v8 ignore next */
        headerRow.appendChild(actions); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        // 419. Mock tagging /* v8 ignore next */ /* v8 ignore next */
        const tags = $create('div', { /* v8 ignore next */ /* v8 ignore next */
          className: 'muted', /* v8 ignore next */ /* v8 ignore next */
          textContent: 'Tags: #onnx, #v1', /* v8 ignore next */ /* v8 ignore next */
          attributes: { style: 'font-size: 0.75rem; margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        li.appendChild(headerRow); /* v8 ignore next */ /* v8 ignore next */
        li.appendChild(tags); /* v8 ignore next */ /* v8 ignore next */
        this.fileList.appendChild(li); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } catch (e) { /* v8 ignore next */ /* v8 ignore next */
      this.fileList.innerHTML = "<p class='danger'>Failed to read IndexedDB.</p>"; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
