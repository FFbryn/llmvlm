from models.llm.qwen import QwenLLM
from models.llm.vicuna import VicunaLLM
from models.vlm.llava import LLaVAVLM
from models.vlm.qwen_v1 import QwenVLM


MODEL_REGISTRY = {
    "qwen_llm": QwenLLM,
    "vicuna_llm": VicunaLLM,
    "qwen_vlm": QwenVLM,
    "llava_vlm": LLaVAVLM,
}