import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

describe('subagent task strip contract', () => {
  it('renders subagent tasks and thinking stream', () => {
    const core = readFileSync('orchestrator/web/static/app-core.js', 'utf-8');
    const events = readFileSync('orchestrator/web/static/app-events.js', 'utf-8');
    const ws = readFileSync('orchestrator/web/static/app-ws.js', 'utf-8');

    expect(core).toContain('subagentTasks');
    expect(core).toContain('subagentTaskPanelOpen');
    expect(events).toContain("type:'subagent_task_thinking_get'");
    expect(ws).toContain("case 'subagent_task_update':");
    expect(ws).toContain("case 'subagent_thinking_append':");
    expect(ws).toContain("case 'subagent_task_terminal':");
  });
});
