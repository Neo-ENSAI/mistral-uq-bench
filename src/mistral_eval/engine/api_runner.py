import os
import time
import asyncio
import random
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

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
        max_concurrency: int = 4,
        max_retries: int = 10,
        top_logprobs: int = 20,
        cache_path: Optional[str] = None,
        request_timeout: int = 60,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        load_dotenv()
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set in the environment variables")

        self.client = Mistral(api_key=api_key, timeout_ms=request_timeout * 1000)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.top_logprobs = top_logprobs
        self.cache_path = cache_path
        self._cache: Dict[str, Dict] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec
        self._pause_until = 0.0

    @staticmethod
    def _cache_key(model: str, prompt: str, temperature: float) -> str:
        import hashlib
        return hashlib.sha1(f"{model}|{temperature}|{prompt}".encode()).hexdigest()

    def _cache_load(self, item: Dict[str, str]) -> Optional[Dict]:
        return self._cache.get(self._cache_key(self.model_name, item["prompt"], self.temperature))

    def _cache_save(self, item: Dict[str, str], record: Dict):
        key = self._cache_key(self.model_name, item["prompt"], self.temperature)
        self._cache[key] = record
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**record, "key": key}) + "\n")

    def _extract_retry_after(self, e: Exception) -> float:
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
        err = str(e).lower()
        if "429" in err or "rate" in err or "timeout" in err or "503" in err:
            return True
        http_res = getattr(e, "http_res", None)
        if http_res is not None:
            try:
                return http_res.status_code >= 500
            except AttributeError:
                pass
        return False

    async def _wait_for_pause(self):
        while True:
            wait = self._pause_until - time.monotonic()
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
                    top_logprobs=self.top_logprobs,
                )
            except Exception as e:
                if not self._is_retryable(e) or attempt == self.max_retries - 1:
                    raise
                wait = max(delay, self._extract_retry_after(e))
                wait += random.uniform(0.5, 1.5)
                self._pause_until = max(self._pause_until, time.monotonic() + wait)
                delay = min(delay * 2.0, 60.0)


    def _build_result(self, item: Dict[str, str], record: Dict) -> InferenceResult:
        """
        Constructs the InferenceResult from a record (cache or API)
        """
        content = record["raw_text"]
        first_token_top_logprobs: Dict[str, float] = record["first_token_top_logprobs"]
        tokens_data = [TokenLogprob(**t) for t in record["tokens"]]

        choice_probs = extract_choice_probabilities(first_token_top_logprobs)
        predictive_entropy = compute_predictive_entropy(choice_probs)
        predicted_choice = max(choice_probs, key=choice_probs.get) if choice_probs else ""

        if not first_token_top_logprobs:
            print(f"[warn] top_logprobs empty (prompt {item['id']})")

        return InferenceResult(
            prompt_id=item["id"],
            model_name=self.model_name,
            engine_type="mistral_api",
            raw_text=content.strip(),
            predicted_choice=predicted_choice,
            choice_probs=choice_probs,
            predictive_entropy=predictive_entropy,
            tokens=tokens_data,
            latency_ms=record["latency_ms"],
            throughput_tok_per_sec=record["throughput_tok_per_sec"],
        )

    async def _call_single(self, item: Dict[str, str]) -> InferenceResult:
        cached = self._cache_load(item)
        if cached is not None:
            return self._build_result(item, cached)

        async with self.semaphore:
            t0 = time.perf_counter()
            response = await self._call_with_retry(item)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            choice_obj = response.choices[0]
            content = choice_obj.message.content or ""
            usage = getattr(response, "usage", None)
            comp_tokens = max(usage.completion_tokens, 1) if (usage and getattr(usage, "completion_tokens", None)) else 1
            throughput = comp_tokens / max(latency_ms / 1000.0, 0.001)

            first_token_top_logprobs: Dict[str, float] = {}
            tokens_data: List[TokenLogprob] = []

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

            record = {
                "raw_text": content,
                "first_token_top_logprobs": first_token_top_logprobs,
                "tokens": [
                    {"token": t.token, "logprob": t.logprob, "top_logprobs": t.top_logprobs}
                    for t in tokens_data
                ],
                "latency_ms": latency_ms,
                "throughput_tok_per_sec": throughput,
            }
            self._cache_save(item, record)
            return self._build_result(item, record)

    async def generate_batch(self, prompts: List[Dict[str, str]]) -> List[InferenceResult]:
        tasks = [self._call_single(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            print(f"[warn] {len(failed)}/{len(prompts)} request(s) failed after retry")
        return [r for r in results if not isinstance(r, Exception)]
