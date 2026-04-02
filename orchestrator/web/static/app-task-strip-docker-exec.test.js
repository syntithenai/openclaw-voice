import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';

describe('docker exec task metadata contract', () => {
  it('tracks docker exec metadata and stream channels', () => {
    const main = readFileSync('orchestrator/main.py', 'utf-8');
    const svc = readFileSync('orchestrator/web/realtime_service.py', 'utf-8');

    expect(main).toContain('container_id');
    expect(main).toContain('container_name');
    expect(main).toContain('exec_id');
    expect(main).toContain('metadata_quality');
    expect(svc).toContain('"type": "sandbox_exec_log_append"');
    expect(svc).toContain('"stream": str(stream or "stdout")');
  });
});
