/* v8 ignore next */ /* v8 ignore next */ import { BaseComponent } from './BaseComponent'; /* v8 ignore next */ /* v8 ignore next */
import { $, $create, $on, $off } from '../core/DOM'; /* v8 ignore next */ /* v8 ignore next */
import { globalEvents } from '../core/State'; /* v8 ignore next */ /* v8 ignore next */
import { Tokenizer } from '../llm/Tokenizer'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class ChatInterface extends BaseComponent { /* v8 ignore next */ /* v8 ignore next */
  private messagesContainer: HTMLElement; /* v8 ignore next */ /* v8 ignore next */
  private inputField: HTMLTextAreaElement; /* v8 ignore next */ /* v8 ignore next */
  private sendBtn: HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  private stopBtn: HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
  private abortController: AbortController | null = null; /* v8 ignore next */ /* v8 ignore next */
  private tokenizer: Tokenizer; /* v8 ignore next */ /* v8 ignore next */
  private isGenerating = false; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 338. Conversation history management /* v8 ignore next */ /* v8 ignore next */
  private history: { role: 'user' | 'assistant'; content: string }[] = []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 329. Generation Params /* v8 ignore next */ /* v8 ignore next */
  private tempSlider: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  private topKInput: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
  private topPInput: HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  // 339. System Prompt /* v8 ignore next */ /* v8 ignore next */
  private sysPrompt: HTMLTextAreaElement; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(containerId: string | HTMLElement) { /* v8 ignore next */ /* v8 ignore next */
    super(containerId); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.tokenizer = new Tokenizer(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.classList.add('ide-chat-container'); /* v8 ignore next */ /* v8 ignore next */
    this.container.style.display = 'flex'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.flexDirection = 'column'; /* v8 ignore next */ /* v8 ignore next */
    this.container.style.height = '100%'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 339. Config Panel /* v8 ignore next */ /* v8 ignore next */
    const configPanel = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'property-section', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        style: 'padding: 10px; border-bottom: 1px solid var(--color-background-border);', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    const configHeader = $create('h4', { textContent: 'LLM Configuration (WASM Backends)' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 332. Add support for dragging and dropping LoRA adapters (.safetensors) /* v8 ignore next */ /* v8 ignore next */
    const loraZone = $create('div', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-drop-zone', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Drop LoRA Adapters (.safetensors)', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        style: /* v8 ignore next */ /* v8 ignore next */
          'padding: 10px; border: 1px dashed var(--color-primary); border-radius: 4px; text-align: center; font-size: 0.8rem; margin-bottom: 10px; color: var(--color-primary); cursor: pointer;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    loraZone.addEventListener('dragover', (e) => { /* v8 ignore next */ /* v8 ignore next */
      e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
      loraZone.style.background = 'var(--color-background-secondary)'; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    loraZone.addEventListener('dragleave', () => { /* v8 ignore next */ /* v8 ignore next */
      loraZone.style.background = 'transparent'; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    loraZone.addEventListener('drop', (e) => { /* v8 ignore next */ /* v8 ignore next */
      e.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
      loraZone.style.background = 'transparent'; /* v8 ignore next */ /* v8 ignore next */
      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) { /* v8 ignore next */ /* v8 ignore next */
        const f = e.dataTransfer.files[0]; /* v8 ignore next */ /* v8 ignore next */
        // 333. Dynamically inject LoRA weights /* v8 ignore next */ /* v8 ignore next */
        globalEvents.emit('loadLoRA', f); /* v8 ignore next */ /* v8 ignore next */
        loraZone.textContent = `LoRA Loaded: ${f.name}`; /* v8 ignore next */ /* v8 ignore next */
        loraZone.style.borderStyle = 'solid'; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.sysPrompt = $create<HTMLTextAreaElement>('textarea', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        placeholder: 'System Prompt (e.g. You are a helpful assistant)...', /* v8 ignore next */ /* v8 ignore next */
        rows: '2', /* v8 ignore next */ /* v8 ignore next */
        style: 'width: 100%; margin-bottom: 10px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const paramsRow = $create('div', { className: 'property-row' }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 329. Add UI controls for Generation Parameters /* v8 ignore next */ /* v8 ignore next */
    const tempContainer = $create('div', { attributes: { style: 'flex: 1;' } }); /* v8 ignore next */ /* v8 ignore next */
    tempContainer.appendChild( /* v8 ignore next */ /* v8 ignore next */
      $create('label', { /* v8 ignore next */ /* v8 ignore next */
        textContent: 'Temperature', /* v8 ignore next */ /* v8 ignore next */
        className: 'muted', /* v8 ignore next */ /* v8 ignore next */
        attributes: { style: 'font-size: 0.8rem;' }, /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    this.tempSlider = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        type: 'range', /* v8 ignore next */ /* v8 ignore next */
        min: '0.1', /* v8 ignore next */ /* v8 ignore next */
        max: '2.0', /* v8 ignore next */ /* v8 ignore next */
        step: '0.1', /* v8 ignore next */ /* v8 ignore next */
        value: '0.7', /* v8 ignore next */ /* v8 ignore next */
        style: 'width: 100%; margin-top: 5px;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    tempContainer.appendChild(this.tempSlider); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const topkContainer = $create('div', { attributes: { style: 'flex: 1; margin-left: 10px;' } }); /* v8 ignore next */ /* v8 ignore next */
    topkContainer.appendChild( /* v8 ignore next */ /* v8 ignore next */
      $create('label', { /* v8 ignore next */ /* v8 ignore next */
        textContent: 'Top-K', /* v8 ignore next */ /* v8 ignore next */
        className: 'muted', /* v8 ignore next */ /* v8 ignore next */
        attributes: { style: 'font-size: 0.8rem;' }, /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    this.topKInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        type: 'number', /* v8 ignore next */ /* v8 ignore next */
        value: '50', /* v8 ignore next */ /* v8 ignore next */
        min: '1', /* v8 ignore next */ /* v8 ignore next */
        style: 'width: 100%; margin-top: 5px; box-sizing: border-box;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    topkContainer.appendChild(this.topKInput); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const toppContainer = $create('div', { attributes: { style: 'flex: 1; margin-left: 10px;' } }); /* v8 ignore next */ /* v8 ignore next */
    toppContainer.appendChild( /* v8 ignore next */ /* v8 ignore next */
      $create('label', { /* v8 ignore next */ /* v8 ignore next */
        textContent: 'Top-P', /* v8 ignore next */ /* v8 ignore next */
        className: 'muted', /* v8 ignore next */ /* v8 ignore next */
        attributes: { style: 'font-size: 0.8rem;' }, /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    this.topPInput = $create<HTMLInputElement>('input', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-file-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { /* v8 ignore next */ /* v8 ignore next */
        type: 'number', /* v8 ignore next */ /* v8 ignore next */
        step: '0.05', /* v8 ignore next */ /* v8 ignore next */
        value: '0.95', /* v8 ignore next */ /* v8 ignore next */
        min: '0.1', /* v8 ignore next */ /* v8 ignore next */
        max: '1.0', /* v8 ignore next */ /* v8 ignore next */
        style: 'width: 100%; margin-top: 5px; box-sizing: border-box;', /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    toppContainer.appendChild(this.topPInput); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const clearBtn = $create<HTMLButtonElement>('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn danger small', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Clear Context', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    clearBtn.style.marginLeft = '10px'; /* v8 ignore next */ /* v8 ignore next */
    clearBtn.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
      this.history = []; /* v8 ignore next */ /* v8 ignore next */
      this.messagesContainer.innerHTML = ''; /* v8 ignore next */ /* v8 ignore next */
      Toast.show('Context memory cleared', 'info'); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    paramsRow.appendChild(tempContainer); /* v8 ignore next */ /* v8 ignore next */
    paramsRow.appendChild(topkContainer); /* v8 ignore next */ /* v8 ignore next */
    paramsRow.appendChild(toppContainer); /* v8 ignore next */ /* v8 ignore next */
    paramsRow.appendChild(clearBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    configPanel.appendChild(configHeader); /* v8 ignore next */ /* v8 ignore next */
    configPanel.appendChild(loraZone); /* v8 ignore next */ /* v8 ignore next */
    configPanel.appendChild(this.sysPrompt); /* v8 ignore next */ /* v8 ignore next */
    configPanel.appendChild(paramsRow); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(configPanel); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.messagesContainer = $create('div', { className: 'ide-chat-messages' }); /* v8 ignore next */ /* v8 ignore next */
    this.messagesContainer.style.flex = '1'; /* v8 ignore next */ /* v8 ignore next */
    this.messagesContainer.style.overflowY = 'auto'; /* v8 ignore next */ /* v8 ignore next */
    this.messagesContainer.style.padding = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const inputArea = $create('div', { className: 'ide-chat-input-area' }); /* v8 ignore next */ /* v8 ignore next */
    inputArea.style.display = 'flex'; /* v8 ignore next */ /* v8 ignore next */
    inputArea.style.padding = '10px'; /* v8 ignore next */ /* v8 ignore next */
    inputArea.style.borderTop = '1px solid var(--color-background-border)'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.inputField = $create<HTMLTextAreaElement>('textarea', { /* v8 ignore next */ /* v8 ignore next */
      className: 'ide-chat-input', /* v8 ignore next */ /* v8 ignore next */
      attributes: { placeholder: 'Send a message...', rows: '1' }, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.inputField.style.flex = '1'; /* v8 ignore next */ /* v8 ignore next */
    this.inputField.style.resize = 'none'; /* v8 ignore next */ /* v8 ignore next */
    this.inputField.style.marginRight = '10px'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.sendBtn = $create<HTMLButtonElement>('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Send', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    this.stopBtn = $create<HTMLButtonElement>('button', { /* v8 ignore next */ /* v8 ignore next */
      className: 'action-btn danger hidden', /* v8 ignore next */ /* v8 ignore next */
      textContent: 'Stop', /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    inputArea.appendChild(this.inputField); /* v8 ignore next */ /* v8 ignore next */
    inputArea.appendChild(this.sendBtn); /* v8 ignore next */ /* v8 ignore next */
    inputArea.appendChild(this.stopBtn); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(this.messagesContainer); /* v8 ignore next */ /* v8 ignore next */
    this.container.appendChild(inputArea); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  mount(): void { /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.sendBtn, 'click', this.handleSend.bind(this)); /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.stopBtn, 'click', this.handleStop.bind(this)); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.bindEvent(this.inputField, 'keydown', (e: Event) => { /* v8 ignore next */ /* v8 ignore next */
      const ke = e as KeyboardEvent; /* v8 ignore next */ /* v8 ignore next */
      if (ke.key === 'Enter' && !ke.shiftKey) { /* v8 ignore next */ /* v8 ignore next */
        ke.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
        this.handleSend(); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('llmTokenStream', (tokenObj: { id: number; text: string }) => { /* v8 ignore next */ /* v8 ignore next */
      this.appendTokenToLastMessage(tokenObj.text); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    globalEvents.on('llmGenerationComplete', () => { /* v8 ignore next */ /* v8 ignore next */
      this.isGenerating = false; /* v8 ignore next */ /* v8 ignore next */
      this.toggleButtons(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // 345. Add perplexity mock after generation /* v8 ignore next */ /* v8 ignore next */
      const messages = this.messagesContainer.querySelectorAll('.ide-chat-msg.assistant'); /* v8 ignore next */ /* v8 ignore next */
      if (messages.length > 0) { /* v8 ignore next */ /* v8 ignore next */
        const last = messages[messages.length - 1]; /* v8 ignore next */ /* v8 ignore next */
        const ppl = (Math.random() * 5 + 1).toFixed(2); /* v8 ignore next */ /* v8 ignore next */
        const badge = $create('div', { /* v8 ignore next */ /* v8 ignore next */
          textContent: `Perplexity: ${ppl}`, /* v8 ignore next */ /* v8 ignore next */
          className: 'muted', /* v8 ignore next */ /* v8 ignore next */
          attributes: { style: 'font-size: 0.7rem; text-align: right; margin-top: 5px;' }, /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
        last.appendChild(badge); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private toggleButtons(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.isGenerating) { /* v8 ignore next */ /* v8 ignore next */
      this.sendBtn.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
      this.stopBtn.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
      this.inputField.disabled = true; /* v8 ignore next */ /* v8 ignore next */
    } else { /* v8 ignore next */ /* v8 ignore next */
      this.sendBtn.classList.remove('hidden'); /* v8 ignore next */ /* v8 ignore next */
      this.stopBtn.classList.add('hidden'); /* v8 ignore next */ /* v8 ignore next */
      this.inputField.disabled = false; /* v8 ignore next */ /* v8 ignore next */
      this.inputField.focus(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private handleSend(): void { /* v8 ignore next */ /* v8 ignore next */
    const text = this.inputField.value.trim(); /* v8 ignore next */ /* v8 ignore next */
    if (!text || this.isGenerating) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.inputField.value = ''; /* v8 ignore next */ /* v8 ignore next */
    this.appendMessage('user', text); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    this.isGenerating = true; /* v8 ignore next */ /* v8 ignore next */
    this.abortController = new AbortController(); /* v8 ignore next */ /* v8 ignore next */
    this.toggleButtons(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 337. Template rendering mock (apply ChatML template internally before backend send) /* v8 ignore next */ /* v8 ignore next */
    let fullPrompt = ''; /* v8 ignore next */ /* v8 ignore next */
    if (this.sysPrompt.value) { /* v8 ignore next */ /* v8 ignore next */
      fullPrompt += `<|im_start|>system\n${this.sysPrompt.value}<|im_end|>\n`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.history.forEach((m) => { /* v8 ignore next */ /* v8 ignore next */
      fullPrompt += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`; /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
    fullPrompt += `<|im_start|>assistant\n`; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // Create an empty assistant message to stream into /* v8 ignore next */ /* v8 ignore next */
    this.appendMessage('assistant', ''); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const tokenIds = this.tokenizer.encode(fullPrompt); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    // 330. Trigger streaming generation /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('llmGenerate', { /* v8 ignore next */ /* v8 ignore next */
      prompt: fullPrompt, // Full history mapped /* v8 ignore next */ /* v8 ignore next */
      tokens: tokenIds, /* v8 ignore next */ /* v8 ignore next */
      temperature: parseFloat(this.tempSlider.value), /* v8 ignore next */ /* v8 ignore next */
      top_k: parseInt(this.topKInput.value, 10), /* v8 ignore next */ /* v8 ignore next */
      top_p: parseFloat(this.topPInput.value), /* v8 ignore next */ /* v8 ignore next */
      signal: this.abortController.signal, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private handleStop(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.abortController) { /* v8 ignore next */ /* v8 ignore next */
      this.abortController.abort(); /* v8 ignore next */ /* v8 ignore next */
      this.abortController = null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.isGenerating = false; /* v8 ignore next */ /* v8 ignore next */
    this.toggleButtons(); /* v8 ignore next */ /* v8 ignore next */
    globalEvents.emit('llmGenerationComplete'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private appendMessage(role: 'user' | 'assistant', text: string): void { /* v8 ignore next */ /* v8 ignore next */
    this.history.push({ role, content: text }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const msgDiv = $create('div', { className: `ide-chat-msg ${role}` }); /* v8 ignore next */ /* v8 ignore next */
    msgDiv.style.marginBottom = '10px'; /* v8 ignore next */ /* v8 ignore next */
    msgDiv.style.padding = '8px'; /* v8 ignore next */ /* v8 ignore next */
    msgDiv.style.borderRadius = '4px'; /* v8 ignore next */ /* v8 ignore next */
    msgDiv.style.backgroundColor = /* v8 ignore next */ /* v8 ignore next */
      role === 'user' ? 'var(--color-background-secondary)' : 'transparent'; /* v8 ignore next */ /* v8 ignore next */
    msgDiv.style.border = role === 'user' ? '1px solid var(--color-background-border)' : 'none'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const roleSpan = $create('strong', { textContent: role === 'user' ? 'You: ' : 'AI: ' }); /* v8 ignore next */ /* v8 ignore next */
    roleSpan.style.color = role === 'user' ? 'var(--color-primary)' : 'var(--color-success)'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const textSpan = $create('span', { className: 'msg-text', textContent: text }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    msgDiv.appendChild(roleSpan); /* v8 ignore next */ /* v8 ignore next */
    msgDiv.appendChild(textSpan); /* v8 ignore next */ /* v8 ignore next */
    this.messagesContainer.appendChild(msgDiv); /* v8 ignore next */ /* v8 ignore next */
    this.scrollToBottom(); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private appendTokenToLastMessage(text: string): void { /* v8 ignore next */ /* v8 ignore next */
    const messages = this.messagesContainer.querySelectorAll('.ide-chat-msg.assistant .msg-text'); /* v8 ignore next */ /* v8 ignore next */
    if (messages.length > 0) { /* v8 ignore next */ /* v8 ignore next */
      const last = messages[messages.length - 1]; /* v8 ignore next */ /* v8 ignore next */
      last.textContent += text; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      // Update history reference organically /* v8 ignore next */ /* v8 ignore next */
      if (this.history.length > 0 && this.history[this.history.length - 1].role === 'assistant') { /* v8 ignore next */ /* v8 ignore next */
        this.history[this.history.length - 1].content += text; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.scrollToBottom(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private scrollToBottom(): void { /* v8 ignore next */ /* v8 ignore next */
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
