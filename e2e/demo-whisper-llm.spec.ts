import { test, expect } from '@playwright/test';

test.describe('Whisper LLM Demo E2E', () => {
  test('Page loads and elements are visible', async ({ page }) => {
    await page.goto('/whisper-llm');

    await expect(page.locator('h1')).toHaveText('Local WebGPU Whisper + LLM');
    await expect(page.locator('#chat-container')).toBeVisible();
    await expect(page.locator('#log')).toBeVisible();
    await expect(page.locator('#record-btn')).toBeVisible();
  });

  test('Clicking record button starts recording flow', async ({ page }) => {
    await page.goto('/whisper-llm');

    // Override media devices to avoid permission prompt
    await page.addInitScript(() => {
        // mock getUserMedia
        navigator.mediaDevices.getUserMedia = async () => {
            return new MediaStream();
        };
        // mock MediaRecorder
        (window as any).MediaRecorder = class {
            state = 'inactive';
            stream: any;
            constructor(stream: any) { this.stream = stream; }
            start() { this.state = 'recording'; }
            stop() { 
                this.state = 'inactive'; 
                if (this.onstop) this.onstop();
            }
            onstop: any;
            ondataavailable: any;
        };
    });

    // Let the mock init model text appear
    await page.waitForTimeout(500);

    const recordBtn = page.locator('#record-btn');
    await recordBtn.click();
    
    await expect(recordBtn).toHaveText('Stop Recording');
    const logContent = await page.locator('#log').textContent();
    expect(logContent).toContain('Recording started');

    await recordBtn.click(); // Stop
    await expect(recordBtn).toHaveText('Start Recording');
    
    const logContentAfter = await page.locator('#log').textContent();
    expect(logContentAfter).toContain('Recording stopped');
  });
});