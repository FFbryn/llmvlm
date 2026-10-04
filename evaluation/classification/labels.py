from enum import Enum


class ClassificationLabel(str, Enum):
    REFUSAL = "refusal"
    COMPLIANCE = "compliance"
    AMBIGUOUS = "ambiguous"