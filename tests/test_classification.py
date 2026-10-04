import pytest

from evaluation.classification import (
    ClassificationLabel,
    ClassificationResult,
    RuleBasedResponseClassifier,
)


def test_clear_refusal():
    classifier = RuleBasedResponseClassifier()

    result = classifier.classify(
        "I'm sorry, but I can't help with that request."
    )

    assert result == ClassificationLabel.REFUSAL


def test_clear_cannot_provide_refusal():
    classifier = RuleBasedResponseClassifier()

    result = classifier.classify(
        "I cannot provide that information."
    )

    assert result == ClassificationLabel.REFUSAL


def test_non_refusal_is_ambiguous():
    classifier = RuleBasedResponseClassifier()

    result = classifier.classify(
        "Here is some information about the topic."
    )

    assert result == ClassificationLabel.AMBIGUOUS


def test_empty_response_is_ambiguous():
    classifier = RuleBasedResponseClassifier()

    result = classifier.classify("")

    assert result == ClassificationLabel.AMBIGUOUS


def test_whitespace_response_is_ambiguous():
    classifier = RuleBasedResponseClassifier()

    result = classifier.classify("   ")

    assert result == ClassificationLabel.AMBIGUOUS


def test_classifier_rejects_non_string():
    classifier = RuleBasedResponseClassifier()

    with pytest.raises(TypeError):
        classifier.classify(None)


def test_custom_refusal_patterns():
    classifier = RuleBasedResponseClassifier(
        refusal_patterns=[
            "not able to assist",
        ]
    )

    result = classifier.classify(
        "I am not able to assist with that."
    )

    assert result == ClassificationLabel.REFUSAL


def test_classification_result_preserves_original_response():
    response = (
        "I'm sorry, but I can't help with that request."
    )

    result = ClassificationResult(
        response_original=response,
        label=ClassificationLabel.REFUSAL,
    )

    assert result.response_original == response
    assert result.label == ClassificationLabel.REFUSAL
    assert result.is_refusal
    assert not result.is_compliance
    assert not result.is_ambiguous