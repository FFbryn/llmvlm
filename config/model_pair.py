from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPairConfig:
    """
    Defines one LLM ↔ VLM model pairing used in the experiment.

    This class only stores experiment configuration.
    It does not load or instantiate models.
    """

    pair_id: str
    llm_model: str
    vlm_model: str

    def __post_init__(self) -> None:
        if not self.pair_id.strip():
            raise ValueError("pair_id tidak boleh kosong.")

        if not self.llm_model.strip():
            raise ValueError("llm_model tidak boleh kosong.")

        if not self.vlm_model.strip():
            raise ValueError("vlm_model tidak boleh kosong.")

    def to_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "llm_model": self.llm_model,
            "vlm_model": self.vlm_model,
        }