import io
import json

from app.runtime.telemetry import RuntimeEvent, emit_event


def test_runtime_event_is_json_and_contains_no_content_fields():
    sink = io.StringIO()
    emit_event(
        RuntimeEvent(
            event="answer_completed",
            response_mode="grounded_evidence_response",
            latency_ms=12.5,
            retrieval_ms=1.2,
            generation_ms=0.0,
            evidence_count=2,
            reason="grounded_evidence_response",
            exception_class=None,
        ),
        sink,
    )
    payload = json.loads(sink.getvalue())
    assert payload["latency_ms"] == 12.5
    assert set(payload) == {
        "event", "response_mode", "latency_ms", "retrieval_ms",
        "generation_ms", "evidence_count", "reason", "exception_class",
    }
    serialized = sink.getvalue().casefold()
    assert "question" not in serialized
    assert "api_key" not in serialized
    assert "evidence_text" not in serialized


def test_runtime_event_records_only_exception_class():
    sink = io.StringIO()
    emit_event(RuntimeEvent.failure("provider_error", ValueError("secret patient text")), sink)
    payload = json.loads(sink.getvalue())
    assert payload["exception_class"] == "ValueError"
    assert "secret patient text" not in sink.getvalue()
