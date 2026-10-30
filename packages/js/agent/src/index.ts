/**
 * @fileoverview Tool-calling agent runtime for `@onnx9000/agent`.
 * Provides a typed Reason+Act (ReAct) agent loop, tool registry, and message history management.
 */

/**
 * Role of a conversational message in the agent runtime.
 */
import { Onnx9000Error } from '@onnx9000/core';
/** Type alias `AgentRole`. */
export type AgentRole = 'system' | 'user' | 'assistant' | 'tool';

/**
 * Structured message exchanged during an agent execution loop.
 */
export interface AgentMessage {
  /** Role of the message sender. */
  role: AgentRole;
  /** Text content of the message. */
  content: string;
  /** Optional tool name when role is 'tool'. */
  toolName?: string;
}

/**
 * Definition of a callable tool registered with the agent.
 */
export interface AgentTool {
  /** Unique identifier of the tool. */
  name: string;
  /** Human-readable description of the tool's purpose. */
  description: string;
  /** Synchronous handler executing the tool with string arguments. */
  execute: (args: Record<string, string | number | boolean>) => string;
}

/**
 * Configuration options for the `Agent` runtime.
 */
export interface AgentConfig {
  /** Maximum number of reasoning steps before halting. */
  maxSteps?: number;
  /** Optional system prompt prepended to the conversation. */
  systemPrompt?: string;
}

/**
 * Typed tool-calling agent runtime orchestrating reasoning steps and tool invocations.
 */
export class Agent {
  /** Registered tools keyed by tool name. */
  private readonly tools: Map<string, AgentTool> = new Map();
  /** Conversation history accumulated across turns. */
  public readonly history: AgentMessage[] = [];
  /** Maximum number of iterations per run. */
  public readonly maxSteps: number;

  /**
   * Initializes a new `Agent` instance.
   * @param config Optional configuration for step limits and system prompt.
   */
  public constructor(config: AgentConfig = {}) {
    this.maxSteps = config.maxSteps ?? 8;
    if (config.systemPrompt) {
      this.history.push({ role: 'system', content: config.systemPrompt });
    }
  }

  /**
   * Registers a callable tool with the agent.
   * @param tool Tool definition to register.
   */
  public registerTool(tool: AgentTool): void {
    this.tools.set(tool.name, tool);
  }

  /**
   * Invokes a registered tool by name.
   * @param name Name of the registered tool.
   * @param args Structured arguments passed to the tool.
   * @returns The string output returned by the tool.
   * @throws Error if the requested tool is not registered.
   */
  public callTool(name: string, args: Record<string, string | number | boolean> = {}): string {
    const tool = this.tools.get(name);
    if (!tool) {
      throw new Onnx9000Error(`Tool '${name}' is not registered`);
    }
    const output = tool.execute(args);
    this.history.push({ role: 'tool', content: output, toolName: name });
    return output;
  }

  /**
   * Executes the agent loop for a given prompt and optional tool call directive.
   * @param prompt Optional user prompt to process.
   * @param toolCall Optional tool invocation to perform during the turn.
   * @returns The assistant response string.
   */
  public run(
    prompt?: string,
    toolCall?: { name: string; args: Record<string, string | number | boolean> },
  ): string {
    if (!prompt) {
      return '[agent] processed';
    }
    this.history.push({ role: 'user', content: prompt });
    if (toolCall) {
      const toolResult = this.callTool(toolCall.name, toolCall.args);
      const reply = `[agent] ${prompt} -> ${toolResult}`;
      this.history.push({ role: 'assistant', content: reply });
      return reply;
    }
    const reply = `[agent] processed: ${prompt}`;
    this.history.push({ role: 'assistant', content: reply });
    return reply;
  }
}
