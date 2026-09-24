from dataclasses import dataclass

from evaluation.experiment_condition import ExperimentCondition


@dataclass(frozen=True)
class ModelPairConfig:
    """
    Konfigurasi pasangan LLM dan VLM.

    model key merupakan identifier internal yang digunakan
    oleh MODEL_REGISTRY / model factory.

    Contoh:
        llm_model_key = "qwen_llm"
        vlm_model_key = "qwen_vlm"
    """

    pair_id: str
    llm_model_key: str
    vlm_model_key: str

    def __post_init__(self) -> None:
        if not self.pair_id.strip():
            raise ValueError(
                "pair_id tidak boleh kosong."
            )

        if not self.llm_model_key.strip():
            raise ValueError(
                "llm_model_key tidak boleh kosong."
            )

        if not self.vlm_model_key.strip():
            raise ValueError(
                "vlm_model_key tidak boleh kosong."
            )

    def model_key_for(
        self,
        condition: ExperimentCondition,
    ) -> str:
        """
        Mengembalikan model key berdasarkan kondisi eksperimen.
        """

        if condition is ExperimentCondition.LLM_TEXT:
            return self.llm_model_key

        if condition in {
            ExperimentCondition.VLM_TEXT,
            ExperimentCondition.VLM_IMAGE,
        }:
            return self.vlm_model_key

        raise ValueError(
            f"ExperimentCondition tidak didukung: {condition}"
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "llm_model_key": self.llm_model_key,
            "vlm_model_key": self.vlm_model_key,
        }