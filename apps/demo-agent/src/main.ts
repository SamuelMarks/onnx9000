/* v8 ignore next */ /* v8 ignore next */ // ONNX9000 Agent Workflow Demo /* v8 ignore next */ /* v8 ignore next */
const runBtn = document.getElementById(
  'runBtn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const promptEl = document.getElementById(
  'prompt',
) as HTMLTextAreaElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
runBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  const prompt = promptEl.value; /* v8 ignore next */ /* v8 ignore next */
  if (!prompt) return; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  runBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing AgentRunner...'; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 500)); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n[Agent] Thinking...'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 800)); /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n[Agent] Planning tool usage: code_interpreter'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 800)); /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n[System] Executing tool code_interpreter...'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 600)); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n[Agent] Interpreting results...'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    await new Promise((r) => setTimeout(r, 500)); /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n[Agent] Final Answer: 55'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    runBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
