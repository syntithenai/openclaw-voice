import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

describe('sandbox task strip contract', () => {
  it('renders and updates sandbox tasks with log panel fetch', () => {
    const core = readFileSync('orchestrator/web/static/app-core.js', 'utf-8');
    const events = readFileSync('orchestrator/web/static/app-events.js', 'utf-8');
    const ws = readFileSync('orchestrator/web/static/app-ws.js', 'utf-8');

    expect(core).toContain('sandboxTasks');
    expect(core).toContain('sandboxTaskPanelOpen');
    expect(events).toContain("type:'sandbox_task_logs_get'");
    expect(ws).toContain("case 'sandbox_exec_update':");
    expect(ws).toContain("case 'sandbox_exec_log_append':");
    expect(ws).toContain("case 'sandbox_task_logs':");
  });
});
