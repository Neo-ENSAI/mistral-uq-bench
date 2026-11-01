import os

from mistral_eval.schemas import InferenceResult


class LangfuseTracer:
    """
    Logs UQ traces and metrics (entropy, conformal set size) to Langfuse
    """

    def __init__(self):
        pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sec = os.environ.get("LANGFUSE_SECRET_KEY")
        self.enabled = bool(pub and sec)
        self.client = None
        if self.enabled:
            from langfuse import get_client

            self.client = get_client()

    def log_evaluation_step(
        self,
        result: InferenceResult,
        conformal_set: list[str],
        target: str,
    ) -> None:

        if not self.enabled or self.client is None:
            return

        try:
            with self.client.start_as_current_observation(
                name="mistral_conformal_eval",
                as_type="generation",
            ) as obs:
                obs.update(
                    input={"prompt_id": result.prompt_id},
                    output={
                        "predicted": result.predicted_choice,
                        "conformal_set": conformal_set,
                        "target": target,
                    },
                    metadata={
                        "model": result.model_name,
                        "engine": result.engine_type,
                        "latency_ms": result.latency_ms,
                    },
                )

                obs.score(name="predictive_entropy", value=result.predictive_entropy)
                obs.score(name="conformal_set_size", value=float(len(conformal_set)))
                obs.score(
                    name="is_covered", value=1.0 if target in conformal_set else 0.0
                )

        except Exception as e:
            print(f"Langfuse Error: {e}")

    def flush(self) -> None:
        if self.enabled and self.client is not None:
            self.client.flush()
