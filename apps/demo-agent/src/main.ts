// ONNX9000 Agent Workflow Demo
const runBtn = document.getElementById('runBtn') as HTMLButtonElement;
const promptEl = document.getElementById('prompt') as HTMLTextAreaElement;
const out = document.getElementById('output') as HTMLElement;

runBtn.addEventListener('click', async () => {
  const prompt = promptEl.value;
  if (!prompt) return;

  runBtn.disabled = true;
  out.innerText = 'Initializing AgentRunner...';

  try {
    await new Promise((r) => setTimeout(r, 500));
    out.innerText += '\n[Agent] Thinking...';

    await new Promise((r) => setTimeout(r, 800));
    out.innerText += '\n[Agent] Planning tool usage: code_interpreter';

    await new Promise((r) => setTimeout(r, 800));
    out.innerText += '\n[System] Executing tool code_interpreter...';

    await new Promise((r) => setTimeout(r, 600));
    out.innerText += '\n[Agent] Interpreting results...';

    await new Promise((r) => setTimeout(r, 500));
    out.innerText += '\n[Agent] Final Answer: 55';
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
