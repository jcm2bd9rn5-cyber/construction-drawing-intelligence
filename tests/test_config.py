from pathlib import Path

from app.core.config import PROJECT_ENV_FILE, Settings, alembic_database_url


def test_project_env_file_is_absolute_and_independent_of_cwd(monkeypatch, tmp_path):
    assert PROJECT_ENV_FILE.is_absolute()
    assert PROJECT_ENV_FILE == Path(__file__).resolve().parents[1] / ".env"
    env_file = tmp_path / "project.env"
    env_file.write_text(
        "DATABASE_URL=postgresql+psycopg://test:test@db/test\n"
        "GEMINI_MODEL=gemini-2.5-flash\n",
        encoding="utf-8",
    )
    other_directory = tmp_path / "elsewhere"
    other_directory.mkdir()
    monkeypatch.chdir(other_directory)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    settings = Settings(_env_file=env_file)
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert settings.gemini_model == "gemini-2.5-flash"


def test_alembic_url_uses_settings_database_url():
    settings = Settings(database_url="postgresql+psycopg://user:p%ss@db/example")
    assert alembic_database_url(settings) == (
        "postgresql+psycopg://user:p%%ss@db/example"
    )
