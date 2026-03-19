from pathlib import Path


def test_static_ui_template_contains_runtime_tokens() -> None:
    source = Path("orchestrator/web/static/index.html").read_text(encoding="utf-8")

    assert "__WS_PORT__" in source
    assert "__MIC_STARTS_DISABLED__" in source
    assert "__AUDIO_AUTHORITY__" in source
    assert "__SERVER_INSTANCE_ID__" in source


def test_http_server_exposes_workspace_and_media_mount_routes() -> None:
    source = Path("orchestrator/web/http_server.py").read_text(encoding="utf-8")

    assert '"/files/workspace"' in source
    assert '"/files/media"' in source
    assert "workspace_files_allow_listing" in source
    assert "media_files_allow_listing" in source


def test_config_and_main_wire_new_web_ui_file_mount_settings() -> None:
    config_source = Path("orchestrator/config.py").read_text(encoding="utf-8")
    main_source = Path("orchestrator/main.py").read_text(encoding="utf-8")

    assert "web_ui_static_root" in config_source
    assert "web_ui_workspace_files_enabled" in config_source
    assert "web_ui_workspace_files_root" in config_source
    assert "web_ui_workspace_files_allow_listing" in config_source
    assert "web_ui_media_files_enabled" in config_source
    assert "web_ui_media_files_root" in config_source
    assert "web_ui_media_files_allow_listing" in config_source

    assert "static_root=config.web_ui_static_root" in main_source
    assert "workspace_files_enabled=config.web_ui_workspace_files_enabled" in main_source
    assert "workspace_files_root=config.web_ui_workspace_files_root" in main_source
    assert "workspace_files_allow_listing=config.web_ui_workspace_files_allow_listing" in main_source
    assert "media_files_enabled=config.web_ui_media_files_enabled" in main_source
    assert "media_files_root=config.web_ui_media_files_root" in main_source
    assert "media_files_allow_listing=config.web_ui_media_files_allow_listing" in main_source