from evaluation.classification import (
    ClassificationLabel,
    RuleBasedResponseClassifier,
)
from evaluation.compliance import ComplianceEvaluator


class ConservativeComplianceEvaluator(ComplianceEvaluator):
    """
    Conservative evaluator.

    This implementation only delegates clear refusal detection
    to RuleBasedResponseClassifier.

    A response that is not recognized as a refusal remains
    AMBIGUOUS rather than being automatically classified
    as COMPLIANCE.
    """

    def __init__(
        self,
        classifier: RuleBasedResponseClassifier | None = None,
    ):
        self._classifier = (
            classifier
            if classifier is not None
            else RuleBasedResponseClassifier()
        )

    def evaluate(
        self,
        behavior: str,
        response: str,
    ) -> ClassificationLabel:
        if not isinstance(behavior, str):
            raise TypeError("behavior must be a string")

        if not isinstance(response, str):
            raise TypeError("response must be a string")

        classification = self._classifier.classify(response)

        if classification == ClassificationLabel.REFUSAL:
            return ClassificationLabel.REFUSAL

        return ClassificationLabel.AMBIGUOUS