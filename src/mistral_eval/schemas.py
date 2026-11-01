from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class TokenLogprob(BaseModel):
    """
    Generated token and its log-probability
    """

    token: str
    logprob: float
    top_logprobs: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of {candidate_token: logprob} for the top-k tokens at this position"
    )


class InferenceResult(BaseModel):
    """
    Unified output contract between the Mistral API and local vLLM engine
    """
    
    prompt_id: str
    model_name: str
    engine_type: str  # "mistral_api" or "vllm_local"
    raw_text: str
    predicted_choice: Optional[str] = None
    choice_probs: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized probability distribution on {'A', 'B', 'C', 'D'}"
    )
    predictive_entropy: float = Field(
        default=0.0,
        description="Shannon entropy over the candidate token distribution"
    )
    tokens: List[TokenLogprob] = Field(default_factory=list)
    latency_ms: float
    throughput_tok_per_sec: float
