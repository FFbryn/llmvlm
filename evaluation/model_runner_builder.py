from config.model_config import ModelConfig
from models.factory import create_model_from_config
from models.runner import ModelRunner


class ModelRunnerBuilder:
    """
    Membuat ModelRunner dari ModelConfig.

    Builder ini hanya bertanggung jawab terhadap:

        ModelConfig
            ↓
        BaseModel
            ↓
        ModelRunner

    Builder tidak:
        - load model
        - generate response
        - menjalankan benchmark
        - melakukan classification
        - melakukan analysis
    """

    def build(
        self,
        config: ModelConfig,
    ) -> ModelRunner:
        if not isinstance(config, ModelConfig):
            raise TypeError(
                "config harus merupakan instance ModelConfig."
            )

        model = create_model_from_config(config)

        return ModelRunner(model)