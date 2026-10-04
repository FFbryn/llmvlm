from abc import ABC, abstractmethod

from evaluation.classification.labels import ClassificationLabel


class BaseResponseClassifier(ABC):
    """
    Interface dasar untuk mengklasifikasikan response model.

    Classifier hanya menerima response dan menghasilkan label.
    Classifier tidak menjalankan model dan tidak mengubah response asli.
    """

    @abstractmethod
    def classify(self, response: str) -> ClassificationLabel:
        """
        Mengklasifikasikan satu response model.
        """
        raise NotImplementedError