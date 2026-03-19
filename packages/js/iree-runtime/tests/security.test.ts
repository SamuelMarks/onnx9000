import { describe, it, expect } from 'vitest';
import { VM_Security_Manager } from '../src/security.js';

describe('Security and Sandbox', () => {
  it('should throw on out of bounds in dev mode', () => {
    expect(() => VM_Security_Manager.checkBounds(100, 10, 50, false)).toThrow('Out of bounds');
  });

  it('should pass on out of bounds in prod mode (WASM native throws)', () => {
    expect(() => VM_Security_Manager.checkBounds(100, 10, 50, true)).not.toThrow();
  });

  it('should trigger watchdog', () => {
    expect(() => VM_Security_Manager.incrementWatchdog(1000001, 1000000)).toThrow(
      'exceeded maximum allowed loop iterations',
    );
  });

  it('should validate huge payloads', () => {
    const dummyHuge = new Uint8Array(51 * 1024 * 1024);
    expect(() => VM_Security_Manager.validateBytecodeIntegrity(dummyHuge)).toThrow(
      'size exceeds 50MB limits',
    );
  });
});
