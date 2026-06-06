import { describe, it, expect, vi } from 'vitest';
import { handleAgentCommand } from '../src/commands/agent.js';

describe('agent command', () => {
  it('should print help when no args', () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation(() => undefined as never);
    
    handleAgentCommand([]);
    
    expect(logSpy).toHaveBeenCalled();
    expect(exitSpy).toHaveBeenCalledWith(0);
    
    logSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should run task', () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    
    handleAgentCommand(['do', 'something']);
    
    expect(logSpy).toHaveBeenCalledWith('Starting agent workflow with task: "do something"...');
    expect(logSpy).toHaveBeenCalledWith('Final Answer: Task completed successfully.');
    
    logSpy.mockRestore();
  });
});
