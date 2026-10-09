from abc import ABC, abstractmethod

from evaluation.classification import ClassificationLabel


class ComplianceEvaluator(ABC):
    """
    Interface for evaluating whether a model response complies
    with the benchmark behavior.

    This evaluator is intentionally separate from the
    refusal-pattern classifier.

    The evaluator receives both:
    - the benchmark behavior
    - the model response

    so that compliance is not inferred merely from the absence
    of refusal language.
    """

    @abstractmethod
    def evaluate(
        self,
        behavior: str,
        response: str,
    ) -> ClassificationLabel:
        """
        Evaluate whether the response complies with the behavior.

        Returns:
            ClassificationLabel.COMPLIANCE
            ClassificationLabel.REFUSAL
            ClassificationLabel.AMBIGUOUS
        """
        raise NotImplementedError