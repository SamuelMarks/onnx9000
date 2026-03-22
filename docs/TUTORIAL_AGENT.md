# Tutorial: Creating a Local AI Coding Assistant in ONNX9000

This tutorial demonstrates how to use the pure-Vanilla TypeScript Agent APIs inside `onnx9000` to build a zero-dependency, browser-local AI coding assistant.

Because `onnx9000` uses a native `AgentRunner` mapped tightly to an underlying Reason+Act loop, you do not need external API keys, Node.js servers, or complex Docker containers.

## 1. Defining Tools

The `globalAgent` API allows you to bind native JavaScript functions. For a local coding assistant, we might want to expose a secure sandbox to evaluate Python code locally using `Pyodide`.

```typescript
import { globalAgent, IAgentTool } from '../agent/Runner';

globalAgent.registerTool({
  name: 'Execute_Python',
  description: 'Executes raw python syntax safely returning stdout logs.',
  execute: async (codeString) => {
    // Utilizing the underlying Pyodide WebWorker
    const result = await runSandboxedCode(codeString);
    return `Stdout: ${result}`;
  },
});
```

## 2. Setting Up the Agent DAG

If you want a multi-stage approach (e.g. A "Planner" -> "Coder" -> "Critic"), you can define a structured graph.

```typescript
const codingDAG = {
  nodes: [
    { id: 'planner', type: 'llm', prompt: 'Create a 3-step plan to solve the user request.' },
    { id: 'coder', type: 'llm', prompt: 'Write the python code based on the plan.' },
    { id: 'critic', type: 'tool', toolName: 'Execute_Python' },
  ],
  edges: [
    { from: 'planner', to: 'coder' },
    { from: 'coder', to: 'critic' },
  ],
};

// Execute
globalAgent.executeDAG(codingDAG, 'Sort an array using bubble sort.');
```

## 3. Streaming and UI Hooks

The IDE automatically hooks into `globalEvents.on("agentLog", ...)` and `globalEvents.on("agentStep", ...)`. If you build custom UIs, simply subscribe to these events to get real-time rendering of the thought process, exactly as the main `AgentInterface` tab does.

You now have a fully functional local AI agent!
