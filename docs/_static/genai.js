import {
  pipeline,
  env,
  TextStreamer,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.2/dist/transformers.js';
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;
class GenAIDemo {
  generator = null;
  statusEl;
  outputEl;
  inputEl;
  btnEl;
  constructor() {
    this.statusEl = document.getElementById('genai-status');
    this.outputEl = document.getElementById('genai-output');
    this.inputEl = document.getElementById('genai-input');
    this.btnEl = document.getElementById('genai-btn');
    this.inputEl.disabled = false;
    this.btnEl.disabled = false;
    this.statusEl.innerText = 'Ready to start.';
    this.statusEl.className = 'status-badge status-ready';
    this.btnEl.addEventListener('click', () => this.generate());
    this.inputEl.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.generate();
      }
    });
  }
  appendMessage(role, text) {
    const msgEl = document.createElement('div');
    msgEl.className = `chat-message ${role}-message`;
    msgEl.innerText = text;
    this.outputEl.appendChild(msgEl);
    this.outputEl.scrollTop = this.outputEl.scrollHeight;
    return msgEl;
  }
  async loadModel() {
    this.statusEl.innerText = 'Downloading Model (onnx-community/TinyStories-Instruct-1M-ONNX)...';
    this.statusEl.className = 'status-badge status-loading';
    // We use a true 1M parameter model for instantaneous load/generation
    this.generator = await pipeline(
      'text-generation',
      'onnx-community/TinyStories-Instruct-1M-ONNX',
      {
        dtype: 'q4',
        device: 'wasm',
        progress_callback: (info) => {
          if (info.status === 'progress') {
            this.statusEl.innerText = `Loading ${info.file}: ${Math.round(info.progress)}%`;
          } else if (info.status === 'done') {
            this.statusEl.innerText = `Caching ${info.file}...`;
          }
        },
      },
    );
    this.statusEl.innerText = 'WASM Engine Online';
    this.statusEl.className = 'status-badge status-ready';
  }
  async generate() {
    const prompt = this.inputEl.value.trim();
    if (!prompt) return;
    this.inputEl.value = '';
    this.inputEl.disabled = true;
    this.btnEl.disabled = true;
    this.appendMessage('user', prompt);
    try {
      if (!this.generator) {
        await this.loadModel();
      }
      this.statusEl.innerText = 'Generating...';
      this.statusEl.className = 'status-badge status-generating';
      const assistantMsgEl = this.appendMessage('assistant', 'Thinking...');
      let outText = '';
      const streamer = new TextStreamer(this.generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk) => {
          outText += chunk;
          assistantMsgEl.innerText = outText;
          this.outputEl.scrollTop = this.outputEl.scrollHeight;
        },
      });
      // Format as Instruction for TinyStories
      const promptText = `Prompt: ${prompt}\nStory:\n`;
      const result = await this.generator(promptText, {
        max_new_tokens: 128,
        temperature: 0.6,
        do_sample: true,
        streamer: streamer,
      });
      // In Transformers.js, when passing a string, it returns [{generated_text: "Prompt: ... Story: \n..."}]
      let final_text = result[0].generated_text;
      // Strip out the prompt text from the result
      if (final_text.startsWith(promptText)) {
        final_text = final_text.slice(promptText.length).trim();
      }
      assistantMsgEl.innerText = final_text || outText; // Fallback to streamed text if exact slicing fails
    } catch (err) {
      this.appendMessage('system', `Error: ${err.message}`);
    } finally {
      this.statusEl.innerText = 'WASM Engine Online';
      this.statusEl.className = 'status-badge status-ready';
      this.inputEl.disabled = false;
      this.btnEl.disabled = false;
      this.inputEl.focus();
    }
  }
}
document.addEventListener('DOMContentLoaded', () => {
  new GenAIDemo();
});
