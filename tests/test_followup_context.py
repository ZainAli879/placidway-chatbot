from app.answering.followup_context import resolve_followup_question


def test_followup_keeps_original_topic_across_multiple_turns():
    first = "What does the dental implant package include?"
    resolved, context = resolve_followup_question(first, None)
    assert resolved == first
    assert context == first

    resolved, new_context = resolve_followup_question("And what about recovery time?", context)
    assert "dental implant package" in resolved
    assert "recovery time" in resolved
    assert new_context is None

    resolved, new_context = resolve_followup_question("What about the cost?", context)
    assert "dental implant package" in resolved
    assert "price or cost" in resolved
    assert new_context is None


def test_followup_memory_works_for_comparison_topics_not_just_treatments():
    first = "Is IVF cheaper in Mexico or Turkey?"
    _, context = resolve_followup_question(first, None)

    resolved, new_context = resolve_followup_question("What about recovery?", context)
    assert first in resolved
    assert "recovery time" in resolved
    assert new_context is None


def test_new_standalone_question_replaces_old_context():
    _, context = resolve_followup_question("Tell me about the clinic in Tijuana", None)
    question = "What is the weather in Cancun?"
    resolved, new_context = resolve_followup_question(question, context)

    assert resolved == question
    assert new_context == question


def test_followup_uses_latest_standalone_topic():
    _, context = resolve_followup_question("What is included in the cancer package?", None)
    _, context = resolve_followup_question("Tell me about the clinic in Tijuana", context)

    resolved, _ = resolve_followup_question("What treatments do they offer?", context)
    assert "Tell me about the clinic in Tijuana" in resolved
    assert "clinic information" in resolved
    assert "cancer package" not in resolved


def test_followup_resolver_does_not_invent_treatment_names():
    first = "What is included in the hair transplant package in Mexico?"
    _, context = resolve_followup_question(first, None)
    resolved, new_context = resolve_followup_question("What about the price?", context)

    assert first in resolved
    assert new_context is None


def test_ordinal_followup_resolves_against_previous_answer_not_package_programs():
    question = "What are the optional procedures for anal cancer?"
    _, context = resolve_followup_question(question, None)
    previous_answer = (
        "The optional procedures listed are: LAK cells $2,600 USD; "
        "Specific Killer Cell Antibody $3,950-$4,250 USD; Ethanol Tumor Ablation $3,800-$4,250 USD."
    )

    resolved, new_context = resolve_followup_question(
        "How much is the second one?", context, previous_answer
    )

    assert "anal cancer" in resolved
    assert previous_answer in resolved
    assert "Resolve references such as 'the second one' against that answer's list" in resolved
    assert new_context is None


def test_ambiguous_ordinal_followup_instructions_require_clarification():
    question = "Tell me about the clinic in Tijuana"
    _, context = resolve_followup_question(question, None)
    resolved, _ = resolve_followup_question(
        "How much is the second one?", context, "The clinic has several services."
    )
    assert "ask which item the visitor means instead of guessing" in resolved


def test_explicitly_named_package_replaces_previous_package_context():
    _, context = resolve_followup_question("Tell me about adenocarcinoma", None)
    current = "What about the adrenal cancer package?"
    resolved, new_context = resolve_followup_question(current, context)
    assert resolved == current
    assert new_context == current

    resolved, new_context = resolve_followup_question(
        "What about the anal cancer package?", new_context
    )
    assert resolved == "What about the anal cancer package?"
    assert new_context == resolved
