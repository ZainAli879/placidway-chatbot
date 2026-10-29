from pathlib import Path


APP_SOURCE = Path("streamlit_app.py")


def test_user_ui_does_not_expose_retrieval_internals():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "Inspect retrieved context" not in source
    assert '"inspection"' not in source
    assert "Artifact:" not in source
    assert "response_mode}" not in source
    assert "Retrieval: hybrid" not in source


def test_user_ui_has_clear_header_and_loading_feedback():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "Your guide to PlacidWay treatment information" in source
    assert "Searching approved pages" in source
    assert "assistant-shell" in source


def test_user_ui_uses_official_brand_identity_and_customer_focused_copy():
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "assets/placidway-logo.png" in source
    assert "Your guide to PlacidWay treatment information" in source
    assert "#F05A55" in source
    assert "Start new conversation" in source


def test_cached_engine_key_includes_evidence_budgets():
    source = APP_SOURCE.read_text(encoding="utf-8")

    signature = source.split("def load_engine(", 1)[1].split(") -> AnswerEngine:", 1)[0]
    assert "max_evidence_chunks" in signature
    assert "max_evidence_chars" in signature
