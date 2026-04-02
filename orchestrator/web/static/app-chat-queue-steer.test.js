import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

describe('chat queue steer contract', () => {
  it('contains queue and steer controls and actions', () => {
    const html = readFileSync('orchestrator/web/static/index.html', 'utf-8');
    const events = readFileSync('orchestrator/web/static/app-events.js', 'utf-8');
    const ws = readFileSync('orchestrator/web/static/app-ws.js', 'utf-8');

    expect(html).toContain('id="chatModeQueueBtn"');
    expect(html).toContain('id="chatModeSteerBtn"');
    expect(html).toContain('id="chatStopBtn"');
    expect(events).toContain("data-action=\"chat-queue-steer-now\"");
    expect(events).toContain("type:'chat_steer_now'");
    expect(ws).toContain("case 'chat_steer_ack':");
    expect(ws).toContain("case 'chat_steer_error':");
  });
});
