---
orphan: true
---

# `onnx9000.genai` API Reference

> **Ecosystem Context:** `onnx9000` operates as a zero-dependency, Polyglot Monorepo. Through its integrated Web IDE (`apps/sphinx-demo-ui`), it supports real-time transpilation and offline conversions across C++, PyTorch, MLIR, CoreML, and Caffe targets without native backends.

## AgentRunner API

The Agent API handles orchestration of zero-dependency Local AI agents via the `AgentRunner` paradigm (Tasks 606 - 634).

### `globalAgent.registerTool(tool: IAgentTool)`

Registers a functional closure as an executable agent tool.

- **`name`**: Unique string identifier mapped internally to action requests.
- **`description`**: Context fed directly into the system prompt guiding the LLM selection mechanism.
- **`execute`**: An asynchronous `(args: string) => Promise<string>` callback executing sandbox logic.

### `globalAgent.runAgentLoop(prompt: string, signal?: AbortSignal)`

Initiates an autonomous Reason+Act iterative loop.

- Interacts exclusively with the `globalEvents` PubSub architecture (`agentLog`).
- Stops execution automatically upon encountering an `AbortSignal.abort()`.

### `globalAgent.executeDAG(dag: IAgentDAG, initialInput: string)`

Runs complex topologies involving conditional looping ("critic", "coder") natively via Javascript asynchronous chains, passing state top-down.
