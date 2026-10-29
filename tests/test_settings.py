import pytest

from app.config.settings import Settings, SettingsError


def test_environment_overrides_streamlit_secrets():
    settings = Settings.from_sources(
        {
            "OPENAI_MODEL": "env-model",
            "OPENAI_API_KEY": "env-key",
            "CHROMA_PERSIST_DIRECTORY": "env-chroma",
            "ENABLE_GROUNDING_VALIDATION": "true",
        },
        {
            "OPENAI_MODEL": "secret-model",
            "OPENAI_API_KEY": "secret-key",
            "CHROMA_PERSIST_DIRECTORY": "secret-chroma",
        },
    )

    assert settings.openai_model == "env-model"
    assert settings.openai_api_key == "env-key"
    assert str(settings.chroma_persist_directory) == "env-chroma"
    assert settings.grounding_validation_enabled is True


def test_missing_api_key_enables_evidence_only_mode():
    settings = Settings.from_sources({}, {})

    assert settings.openai_api_key is None
    assert settings.generation_enabled is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MAX_QUESTION_CHARS", "1001"),
        ("MAX_EVIDENCE_CHUNKS", "6"),
        ("MAX_EVIDENCE_CHARS", "6001"),
        ("SESSION_REQUESTS", "0"),
        ("GENERATION_CONCURRENCY", "0"),
        ("DAILY_GENERATION_LIMIT", "0"),
    ],
)
def test_settings_reject_unsafe_or_invalid_limits(name, value):
    with pytest.raises(SettingsError) as exc_info:
        Settings.from_sources({name: value})

    assert name in str(exc_info.value)


def test_settings_errors_do_not_include_secret_values():
    secret = "sk-this-must-never-appear"
    with pytest.raises(SettingsError) as exc_info:
        Settings.from_sources({"OPENAI_API_KEY": secret, "SESSION_WINDOW_SECONDS": "bad"})

    assert secret not in str(exc_info.value)


def test_settings_use_production_defaults():
    settings = Settings.from_sources({})

    assert settings.max_question_chars == 1000
    assert settings.max_evidence_chunks == 5
    assert settings.openai_model == "gpt-4.1-mini"
    assert settings.max_evidence_chars == 6000
    assert settings.session_requests == 10
    assert settings.session_window_seconds == 60
    assert settings.generation_concurrency == 2
    assert settings.daily_generation_limit == 500
    assert str(settings.chroma_persist_directory) == "data\\processed\\chroma"
    assert settings.grounding_validation_enabled is True
