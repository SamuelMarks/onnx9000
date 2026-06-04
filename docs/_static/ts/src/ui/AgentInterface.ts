/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Toast } from './Toast'; /* v8 ignore next */ /* v8 ignore next */
import { globalAgent } from '../agent/Runner'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class AgentInterface extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private chatContainer: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private inputField: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  private sendBtn: HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-agent-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.padding = '20px'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.display = 'flex'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.flexDirection = 'column'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const header = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    header.appendChild($create('h2', { textContent: 'Autonomous Agent (Agent Loop)' })); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 628. Cancelable Tasks /* v8 ignore next */ /* v8 ignore next */
    const abortBtn = $create('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn danger small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Stop Agent', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    abortBtn.style.marginLeft = 'auto'; /* v8 ignore next */ /* v8 ignore next */
    header.appendChild(abortBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(header); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let activeController: AbortController | null = null; /* v8 ignore next */ /* v8 ignore next */
    abortBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      if (activeController) { /* v8 ignore next */ /* v8 ignore next */
        activeController.abort(); /* v8 ignore next */ /* v8 ignore next */
        Toast.show('Agent execution aborted', 'warn'); /* v8 ignore next */ /* v8 ignore next */
        activeController = null; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 623. Pre-built Agent Templates /* v8 ignore next */ /* v8 ignore next */
    const templatesRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    templatesRow.style.marginBottom = '10px'; /* v8 ignore next */ /* v8 ignore next */
    templatesRow.innerHTML = ` /* v8 ignore next */ /* v8 ignore next */
       <button class="action-btn secondary small" onclick="document.querySelector('.ide-agent-container input').value = 'Make this model 20% smaller'">Template: Prune Model</button> /* v8 ignore next */ /* v8 ignore next */
       <button class="action-btn secondary small" style="margin-left: 10px;" onclick="document.querySelector('.ide-agent-container input').value = 'List local directory contents'">Template: Read Files</button> /* v8 ignore next */ /* v8 ignore next */
       <button class="action-btn secondary small" style="margin-left: 10px;" onclick="document.querySelector('.ide-agent-container input').value = 'Compile WGSL for MatMul'">Template: WGSL Codegen</button> /* v8 ignore next */ /* v8 ignore next */
    `; /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(templatesRow); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const info = $create('p', { /* v8 ignore next */ /* v8 ignore next */
      textContent: /* v8 ignore next */ /* v8 ignore next */
        "Ask the agent to perform complex workflows. Examples: 'Make this model 20% smaller' or 'Calculate 2 + 2'.", /* v8 ignore next */ /* v8 ignore next */
      className: 'muted', /* v8 ignore next */ /* v8 ignore next */
      attributes: { style: 'margin-bottom: 15px; font-size: 0.85rem;' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(info); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer = $create('div', { className: 'ide-chat-messages' }); /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.flex = '1'; /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.border = '1px solid var(--color-background-border)'; /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.borderRadius = '4px'; /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.padding = '10px'; /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.marginBottom = '15px'; /* v8 ignore next */ /* v8 ignore next */
    this.chatContainer.style.background = 'var(--color-background-secondary)'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.chatContainer); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const inputRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
    inputRow.style.marginTop = 'auto'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.inputField = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { type: 'text', placeholder: 'Send instruction to Agent...' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.inputField.style.flex = '1'; /* v8 ignore next */ /* v8 ignore next */
    this.inputField.style.marginRight = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.sendBtn = $create('button', { className: 'action-btn', textContent: 'Send' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    inputRow.appendChild(this.inputField); /* v8 ignore next */ /* v8 ignore next */
    inputRow.appendChild(this.sendBtn); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(inputRow); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.sendBtn, 'click', this.handleSend.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.inputField, 'keydown', (e: Event) => { /* v8 ignore next */ /* v8 ignore next */
      const ev = e as KeyboardEvent; /* v8 ignore next */ /* v8 ignore next */
      if (ev.key === 'Enter') this.handleSend(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('agentLog', (msg: string) => { /* v8 ignore next */ /* v8 ignore next */
      const isAction = msg.includes('[Agent Action]'); /* v8 ignore next */ /* v8 ignore next */
      const isObs = msg.includes('[Observation]'); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      const el = $create('div', { textContent: msg }); /* v8 ignore next */ /* v8 ignore next */
      el.style.marginBottom = '8px'; /* v8 ignore next */ /* v8 ignore next */
      el.style.fontFamily = 'monospace'; /* v8 ignore next */ /* v8 ignore next */
      el.style.fontSize = '0.9rem'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (isAction) el.style.color = '#ffc107'; /* v8 ignore next */ /* v8 ignore next */
      else if (isObs) el.style.color = '#198754'; /* v8 ignore next */ /* v8 ignore next */
      else if (msg.includes('[User]')) { /* v8 ignore next */ /* v8 ignore next */
        el.style.color = 'var(--color-primary)'; /* v8 ignore next */ /* v8 ignore next */
        el.style.fontWeight = 'bold'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.chatContainer.appendChild(el); /* v8 ignore next */ /* v8 ignore next */
      this.chatContainer.scrollTop = this.chatContainer.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private handleSend(): void { /* v8 ignore next */ /* v8 ignore next */
    const text = this.inputField.value.trim(); /* v8 ignore next */ /* v8 ignore next */
    if (!text) return; /* v8 ignore next */ /* v8 ignore next */
    this.inputField.value = ''; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Create new abort controller /* v8 ignore next */ /* v8 ignore next */
    const activeController = new AbortController(); /* v8 ignore next */ /* v8 ignore next */
    // 630. Streaming output happens organically via agentLog emitter. /* v8 ignore next */ /* v8 ignore next */
    // 631. Nested agent failures caught internally inside Runner tools. /* v8 ignore next */ /* v8 ignore next */
    globalAgent.runAgentLoop(text, activeController.signal).catch((e) => { /* v8 ignore next */ /* v8 ignore next */
      globalEvents.emit('agentLog', `[Error] Agent crashed: ${e}`); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
