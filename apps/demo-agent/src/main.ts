/**
 * @fileoverview ONNX9000 Agent Workflow Demo UI.
 */

/**
 * Initializes the Agent workflow demo UI.
 */
export function initAgentDemo(): void {
  const runBtn = document.getElementById('runBtn') as HTMLButtonElement | null;
  const promptEl = document.getElementById('prompt') as HTMLTextAreaElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!runBtn || !promptEl || !out) return;

  runBtn.addEventListener('click', async () => {
    const prompt = promptEl.value;
    if (!prompt) return;

    runBtn.disabled = true;
    out.innerText = 'Initializing AgentRunner...';

    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `\n[Agent] Thinking...`;

    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `\n[Agent] Planning tool usage: code_interpreter`;

    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `\n[System] Executing tool code_interpreter...`;

    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `\n[Agent] Interpreting results...`;

    await new Promise((r) => setTimeout(r, 50));
    out.innerText += `\n[Agent] Final Answer: 55`;
    runBtn.disabled = false;
  });
}

initAgentDemo();
