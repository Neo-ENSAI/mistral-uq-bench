import os
import time
import asyncio
import random
from typing import List, Dict

from mistralai.client import Mistral

from mistral_eval.engine.base import BaseInferenceEngine
from mistral_eval.schemas import InferenceResult, TokenLogprob
from mistral_eval.stats.metrics import (
    extract_choice_probabilities,
    compute_predictive_entropy,
)


class MistralAPIEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_name: str = "glm-5-2",
        temperature: float = 0.0,
        max_concurrency: int = 2,
        max_retries: int = 8,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set in the environment variables")

        self.client = Mistral(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self._pause_until = 0.0

    def _extract_retry_after(self, e: Exception) -> float:
        """
        Tente de lire l'en-tête Retry-After de la réponse HTTP du SDK.
        """
        http_res = getattr(e, "http_res", None)
        if http_res is not None:
            headers = getattr(http_res, "headers", None) or {}
            for key in ("retry-after", "Retry-After", "ratelimit-reset"):
                if key in headers:
                    try:
                        return float(headers[key])
                    except (TypeError, ValueError):
                        pass
        return 0.0

    def _is_retryable(self, e: Exception) -> bool:
        """
        If the error is a 429, 5xx or message 'rate limit' -> we can retry
        """
        err = str(e).lower()
        if "429" in err or "rate" in err or "timeout" in err:
            return True
        http_res = getattr(e, "http_res", None)
        if http_res is not None:
            try:
                return http_res.status_code >= 500
            except AttributeError:
                pass
        return False

    async def _wait_for_pause(self):
        """
        Wait until the global pause is over
        """
        while True:
            now = time.monotonic()
            wait = self._pause_until - now
            if wait <= 0:
                return
            await asyncio.sleep(wait)

    async def _call_with_retry(self, item: Dict[str, str]):
        delay = 2.0
        for attempt in range(self.max_retries):
            await self._wait_for_pause()
            try:
                return await self.client.chat.complete_async(
                    model=self.model_name,
                    messages=[{"role": "user", "content": item["prompt"]}],
                    temperature=self.temperature,
                    max_tokens=16,
                    logprobs=True,
                )
            except Exception as e:
                if not self._is_retryable(e) or attempt == self.max_retries - 1:
                    raise
                wait = max(delay, self._extract_retry_after(e))
                wait += random.uniform(0.5, 1.5)
                self._pause_until = max(self._pause_until, time.monotonic() + wait)
                delay *= 2.0

    async def _call_single(self, item: Dict[str, str]) -> InferenceResult:
        async with self.semaphore:
            t0 = time.perf_counter()
            response = await self._call_with_retry(item)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            choice_obj = response.choices[0]
            content = choice_obj.message.content or ""
            usage = getattr(response, "usage", None)
            comp_tokens = max(usage.completion_tokens, 1) if (usage and getattr(usage, "completion_tokens", None)) else 1
            throughput = comp_tokens / max(latency_ms / 1000.0, 0.001)

            tokens_data: List[TokenLogprob] = []
            first_token_top_logprobs: Dict[str, float] = {}

            if choice_obj.logprobs and choice_obj.logprobs.content:
                for idx, logprob_item in enumerate(choice_obj.logprobs.content):
                    candidates = {
                        top.token: top.logprob
                        for top in (logprob_item.top_logprobs or [])
                    }
                    if idx == 0:
                        first_token_top_logprobs = candidates
                    tokens_data.append(
                        TokenLogprob(
                            token=logprob_item.token,
                            logprob=logprob_item.logprob,
                            top_logprobs=candidates,
                        )
                    )

            choice_probs = extract_choice_probabilities(first_token_top_logprobs)
            predictive_entropy = compute_predictive_entropy(choice_probs)
            predicted_choice = max(choice_probs, key=choice_probs.get) if choice_probs else ""

            return InferenceResult(
                prompt_id=item["id"],
                model_name=self.model_name,
                engine_type="mistral_api",
                raw_text=content.strip(),
                predicted_choice=predicted_choice,
                choice_probs=choice_probs,
                predictive_entropy=predictive_entropy,
                tokens=tokens_data,
                latency_ms=latency_ms,
                throughput_tok_per_sec=throughput,
            )

    async def generate_batch(self, prompts: List[Dict[str, str]]) -> List[InferenceResult]:
        tasks = [self._call_single(p) for p in prompts]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            print(f"[warn] {len(failed)}/{len(prompts)} request(s) failed after retry")
        return [r for r in results if not isinstance(r, Exception)]
