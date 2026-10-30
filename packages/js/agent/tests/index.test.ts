import { describe, expect, it } from 'vitest';
import { Agent } from '../src/index.js';

describe('Agent', () => {
  it('should instantiate with defaults and run without prompt', () => {
    const agent = new Agent();
    expect(agent.maxSteps).toBe(8);
    expect(agent.run()).toBe('[agent] processed');
  });

  it('should support system prompt, user prompt, and tool calling', () => {
    const agent = new Agent({ maxSteps: 4, systemPrompt: 'You are helpful.' });
    expect(agent.maxSteps).toBe(4);
    expect(agent.history).toHaveLength(1);

    agent.registerTool({
      name: 'add',
      description: 'Adds two numbers',
      execute: (args) => String(Number(args.a) + Number(args.b)),
    });

    expect(agent.run('hello')).toBe('[agent] processed: hello');
    expect(agent.run('compute', { name: 'add', args: { a: 2, b: 3 } })).toBe(
      '[agent] compute -> 5',
    );
    expect(agent.callTool('add')).toBe('NaN');
    expect(() => agent.callTool('missing')).toThrow("Tool 'missing' is not registered");
  });
});
