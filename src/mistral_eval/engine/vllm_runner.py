import time
from typing import List, Dict
from mistral_eval.engine.base import BaseInferenceEngine
from mistral_eval.schemas import InferenceResult
from mistral_eval.stats.metrics import extract_choice_probabilities, compute_predictive_entropy


class VLLMInferenceEngine(BaseInferenceEngine):
    """
    High performance local inference engine using vLLM (PagedAttention)
    to evaluate open-weights Mistral models (e.g., Mistral-7B-Instruct-v0.3)
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.0,
        quantization: str | None = None,
        top_logprobs: int = 20,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed on  this device. Please install it with : uv pip install vllm."
            ) from exc

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=4,
            logprobs=top_logprobs,
        )
        self.llm = LLM(
            model=model_name,
            quantization=quantization,
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
        )

    async def generate_batch(self, prompts: List[Dict[str, str]]) -> List[InferenceResult]:
        raw_prompts = [p["prompt"] for p in prompts]
        t0 = time.perf_counter()
        outputs = self.llm.generate(raw_prompts, self.sampling_params)
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / max(len(prompts), 1)

        results: List[InferenceResult] = []
        for item, out in zip(prompts, outputs):
            comp = out.outputs[0]
            text = comp.text.strip()
            first_step_logprobs: Dict[str, float] = {}
            if comp.logprobs and len(comp.logprobs) > 0:
                for tok_id, lp_obj in comp.logprobs[0].items():
                    decoded = getattr(lp_obj, "decoded_token", str(tok_id))
                    first_step_logprobs[decoded] = float(lp_obj.logprob)

            choice_probs = extract_choice_probabilities(first_step_logprobs)
            entropy = compute_predictive_entropy(choice_probs)
            pred_choice = max(choice_probs, key=choice_probs.get)

            results.append(
                InferenceResult(
                    prompt_id=item["id"],
                    model_name=self.model_name,
                    engine_type="vllm_local",
                    raw_text=text,
                    predicted_choice=pred_choice,
                    choice_probs=choice_probs,
                    predictive_entropy=entropy,
                    tokens=[],
                    latency_ms=per_item_ms,
                    throughput_tok_per_sec=1000.0 / max(per_item_ms, 1e-3),
                )
            )
        return results
