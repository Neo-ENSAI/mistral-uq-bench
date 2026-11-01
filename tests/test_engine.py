from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mistral_eval.engine.api_runner import MistralAPIEngine
from mistral_eval.schemas import InferenceResult


@pytest.mark.asyncio
async def test_mistral_api_engine_batch_contract():

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Paris"))]
    mock_response.usage = MagicMock(completion_tokens=5)

    with (
        patch.dict("os.environ", {"MISTRAL_API_KEY": "fake_test_key"}),
        patch("mistral_eval.engine.api_runner.Mistral") as mock_mistral_cls,
    ):
        mock_client = MagicMock()
        mock_client.chat.complete_async = AsyncMock(return_value=mock_response)
        mock_mistral_cls.return_value = mock_client

        engine = MistralAPIEngine(model_name="mistral-small-latest")

        batch = [
            {"id": "test_1", "prompt": "Prompt 1"},
            {"id": "test_2", "prompt": "Prompt 2"},
            {"id": "test_3", "prompt": "Prompt 3"},
        ]

        results = await engine.generate_batch(batch)

        assert len(results) == 3
        for res in results:
            assert isinstance(res, InferenceResult)
            assert res.latency_ms > 0.0
            assert res.throughput_tok_per_sec > 0.0
            assert res.raw_text == "Paris"
            assert res.engine_type == "mistral_api"
