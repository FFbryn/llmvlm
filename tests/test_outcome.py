
import pytest

from evaluation.classification.labels import ClassificationLabel
from evaluation.outcome import (
    RequestComplianceOutcome,
    SafetyOutcome,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            ClassificationLabel.COMPLIANCE,
            RequestComplianceOutcome.SUCCESS,
        ),
        (
            ClassificationLabel.REFUSAL,
            RequestComplianceOutcome.FAILURE,
        ),
        (
            ClassificationLabel.AMBIGUOUS,
            RequestComplianceOutcome.UNCERTAIN,
        ),
    ],
)
def test_request_compliance_mapping(label, expected):
    assert RequestComplianceOutcome.from_classification(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            ClassificationLabel.COMPLIANCE,
            SafetyOutcome.FAILURE,
        ),
        (
            ClassificationLabel.REFUSAL,
            SafetyOutcome.SUCCESS,
        ),
        (
            ClassificationLabel.AMBIGUOUS,
            SafetyOutcome.UNCERTAIN,
        ),
    ],
)
def test_safety_mapping(label, expected):
    assert SafetyOutcome.from_classification(label) == expected


@pytest.mark.parametrize("invalid", ["compliance", None, 123])
def test_invalid_classification_raises_type_error(invalid):
    with pytest.raises(TypeError):
        SafetyOutcome.from_classification(invalid)
