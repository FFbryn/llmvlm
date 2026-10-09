import pytest

from evaluation.classification import ClassificationLabel
from evaluation.compliance import ComplianceEvaluator
from evaluation.conservative_compliance import (
    ConservativeComplianceEvaluator,
)


def test_conservative_evaluator_is_compliance_evaluator():
    evaluator = ConservativeComplianceEvaluator()

    assert isinstance(
        evaluator,
        ComplianceEvaluator,
    )


def test_clear_refusal_returns_refusal():
    evaluator = ConservativeComplianceEvaluator()

    result = evaluator.evaluate(
        behavior="benchmark behavior",
        response=(
            "I'm sorry, but I can't help with that request."
        ),
    )

    assert result == ClassificationLabel.REFUSAL


def test_non_refusal_does_not_become_compliance():
    evaluator = ConservativeComplianceEvaluator()

    result = evaluator.evaluate(
        behavior="benchmark behavior",
        response="Here is some general information.",
    )

    assert result == ClassificationLabel.AMBIGUOUS


def test_empty_response_is_ambiguous():
    evaluator = ConservativeComplianceEvaluator()

    result = evaluator.evaluate(
        behavior="benchmark behavior",
        response="",
    )

    assert result == ClassificationLabel.AMBIGUOUS


def test_behavior_must_be_string():
    evaluator = ConservativeComplianceEvaluator()

    with pytest.raises(TypeError):
        evaluator.evaluate(
            behavior=None,
            response="Some response.",
        )


def test_response_must_be_string():
    evaluator = ConservativeComplianceEvaluator()

    with pytest.raises(TypeError):
        evaluator.evaluate(
            behavior="benchmark behavior",
            response=None,
        )