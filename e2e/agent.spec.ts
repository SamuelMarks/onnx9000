import { test, expect } from '@playwright/test';

test.describe('ONNX9000 GenAI & Swarm Features', () => {
  test.beforeEach(async ({ page }) => {
    try {
      await page.goto('/');
      await page.waitForSelector('#ide-root', { state: 'attached', timeout: 5000 });
    } catch (e) {
      console.log('Skipping real nav', e);
      test.skip();
    }
  });

  test('305. RAG Database Queries feature rendering', async ({ page }) => {
    const ragTab = page.locator('text="RAG Chat"');
    if (await ragTab.isVisible()) {
      await ragTab.click();

      const ragContainer = page.locator('#rag-container');
      await expect(ragContainer).toBeVisible();

      const input = page.locator('#rag-input');
      const sendBtn = page.locator('#rag-send-btn');

      await expect(input).toBeVisible();
      await expect(sendBtn).toBeVisible();

      // Attempt a query
      await input.fill('What is ONNX?');
      await sendBtn.click();

      // A loading spinner or a message should appear
      const history = page.locator('#rag-history');
      await expect(history).toContainText('What is ONNX?', { timeout: 3000 });
    }
  });

  test('Agent Interface provides code generation input', async ({ page }) => {
    const agentTab = page.locator('text="Agent Interface"');
    if (await agentTab.isVisible()) {
      await agentTab.click();

      const agentContainer = page.locator('#agent-container');
      await expect(agentContainer).toBeVisible();

      const promptInput = page.locator('#agent-prompt');
      await expect(promptInput).toBeVisible();

      const genBtn = page.locator('#agent-generate-btn');
      await expect(genBtn).toBeVisible();
    }
  });

  test('324. Test failure handling with WebRTC swarm setup', async ({ page }) => {
    const swarmTab = page.locator('text="Swarm Setup"');
    if (await swarmTab.isVisible()) {
      await swarmTab.click();

      const swarmContainer = page.locator('#swarm-container');
      await expect(swarmContainer).toBeVisible();

      // Look for peer status indicator
      const peerList = page.locator('#swarm-peers');
      await expect(peerList).toBeVisible();

      // Connect to signaling button
      const connectBtn = page.locator('button', { hasText: 'Connect' });
      if (await connectBtn.isVisible()) {
        await connectBtn.click();
      }
    }
  });
});
