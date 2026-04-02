from pathlib import Path


def test_docker_exec_payload_fields_contract() -> None:
    main_source = Path("orchestrator/main.py").read_text(encoding="utf-8")
    service_source = Path("orchestrator/web/realtime_service.py").read_text(encoding="utf-8")

    assert '"container_id": container_id' in main_source
    assert '"container_name": container_name' in main_source
    assert '"container_workdir": container_workdir' in main_source
    assert '"exec_id": exec_id' in main_source
    assert '"metadata_quality": metadata_quality' in main_source
    assert '"type": "sandbox_exec_update"' in service_source
    assert '"type": "sandbox_exec_log_append"' in service_source
