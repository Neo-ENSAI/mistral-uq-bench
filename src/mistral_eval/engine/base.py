from abc import ABC, abstractmethod

from mistral_eval.schemas import InferenceResult


class BaseInferenceEngine(ABC):
    """
    Common interface for all Mistral inference engines
    """

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    async def generate_batch(
        self, prompts: list[dict[str, str]]
    ) -> list[InferenceResult]:
        """
        Process a batch of prompts [{'id': ..., 'prompt': ...}] and retrieve token logprobs
        """
