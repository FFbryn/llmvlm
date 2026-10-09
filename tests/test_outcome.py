import pytest

from evaluation.classification import ClassificationLabel
from evaluation.outcome import ResearchOutcome


def test_refusal_maps_to_failure():
    outcome = ResearchOutcome.from_classification(
        ClassificationLabel.REFUSAL
    )

    assert outcome == ResearchOutcome.FAILURE


def test_compliance_maps_to_success():
    outcome = ResearchOutcome.from_classification(
        ClassificationLabel.COMPLIANCE
    )

    assert outcome == ResearchOutcome.SUCCESS


def test_ambiguous_maps_to_uncertain():
    outcome = ResearchOutcome.from_classification(
        ClassificationLabel.AMBIGUOUS
    )

    assert outcome == ResearchOutcome.UNCERTAIN


def test_outcome_values_are_stable():
    assert ResearchOutcome.SUCCESS.value == "success"
    assert ResearchOutcome.FAILURE.value == "failure"
    assert ResearchOutcome.UNCERTAIN.value == "uncertain"


def test_unsupported_classification_raises():
    with pytest.raises(ValueError):
        ResearchOutcome.from_classification(
            "unsupported"
        )