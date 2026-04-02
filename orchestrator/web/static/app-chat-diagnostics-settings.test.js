import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

describe('chat diagnostics settings contract', () => {
  it('contains diagnostics controls and ws actions', () => {
    const html = readFileSync('orchestrator/web/static/index.html', 'utf-8');
    const core = readFileSync('orchestrator/web/static/app-core.js', 'utf-8');
    const events = readFileSync('orchestrator/web/static/app-events.js', 'utf-8');

    expect(html).toContain('id="chatVerboseBtn"');
    expect(html).toContain('id="chatReasoningBtn"');
    expect(html).toContain('id="chatLifecyclePolicyBtn"');
    expect(html).toContain('id="chatInterimToggle"');
    expect(core).toContain("PREF_CHAT_VERBOSE_LEVEL");
    expect(events).toContain("chat-verbose-cycle");
    expect(events).toContain("chat-reasoning-cycle");
    expect(events).toContain("chat-lifecycle-policy-cycle");
  });
});
