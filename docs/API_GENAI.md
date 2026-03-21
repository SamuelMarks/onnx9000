# `onnx9000.genai` API Reference

## AgentRunner API

The Agent API handles orchestration of zero-dependency Local AI agents via the `AgentRunner` paradigm (Tasks 606 - 634).

### `globalAgent.registerTool(tool: IAgentTool)`

Registers a functional closure as an executable agent tool.

- **`name`**: Unique string identifier mapped internally to action requests.
- **`description`**: Context fed directly into the system prompt guiding the LLM selection mechanism.
- **`execute`**: An asynchronous `(args: string) => Promise<string>` callback executing sandbox logic.

### `globalAgent.runReAct(prompt: string, signal?: AbortSignal)`

Initiates an autonomous Reason+Act iterative loop.

- Interacts exclusively with the `globalEvents` PubSub architecture (`agentLog`).
- Stops execution automatically upon encountering an `AbortSignal.abort()`.

### `globalAgent.executeDAG(dag: IAgentDAG, initialInput: string)`

Runs complex topologies involving conditional looping ("critic", "coder") natively via Javascript asynchronous chains, passing state top-down.
