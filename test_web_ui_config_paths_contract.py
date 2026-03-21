from pathlib import Path

from orchestrator.config import VoiceConfig


def test_relative_web_ui_paths_resolve_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parent
    config = VoiceConfig(
        web_ui_ssl_certfile="certs/doomsday.local-cert.pem",
        web_ui_ssl_keyfile="certs/doomsday.local-key.pem",
        web_ui_static_root="orchestrator/web/static",
    )

    assert config.web_ui_ssl_certfile == str((repo_root / "certs" / "doomsday.local-cert.pem").resolve())
    assert config.web_ui_ssl_keyfile == str((repo_root / "certs" / "doomsday.local-key.pem").resolve())
    assert config.web_ui_static_root == str((repo_root / "orchestrator" / "web" / "static").resolve())
