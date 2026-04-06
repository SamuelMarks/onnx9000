import { test, expect } from '@playwright/test';

test.describe('Llama Web Demo E2E', () => {
  test('Page loads and elements are visible', async ({ page }) => {
    await page.goto('/llama-web');

    await expect(page.locator('h1')).toHaveText('Llama Inference (WebGPU)');
    await expect(page.locator('#chat-container')).toBeVisible();
    await expect(page.locator('#messages')).toBeVisible();
    await expect(page.locator('#prompt-input')).toBeVisible();
    await expect(page.locator('#send-btn')).toBeVisible();
  });

  test('Submitting a message adds it to chat', async ({ page }) => {
    await page.goto('/llama-web');

    const input = page.locator('#prompt-input');
    await input.fill('Hello world!');
    await page.locator('#send-btn').click();

    // Verify user message
    const msgs = page.locator('.message.user');
    await expect(msgs.last()).toHaveText('Hello world!');

    // Verify bot response
    const botMsgs = page.locator('.message.bot');
    await expect(botMsgs.last()).toContainText('I am an AI assistant', { timeout: 3000 });
  });
});