"""Build the configured answer generator."""

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.rag.bedrock import BedrockGenerator
from app.rag.generator import AnswerGenerator, ExtractiveGenerator, OpenAICompatibleGenerator


class UnknownGeneratorError(ValueError):
    pass


def get_generator(settings: Settings, embedder: EmbeddingProvider | None = None) -> AnswerGenerator:
    if settings.llm_provider == "extractive":
        return ExtractiveGenerator(embedder)
    if settings.llm_provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_model:
            raise UnknownGeneratorError(
                "LLM_PROVIDER=openai_compatible needs LLM_BASE_URL and LLM_MODEL"
            )
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        return OpenAICompatibleGenerator(
            settings.llm_base_url, settings.llm_model, key, settings.llm_timeout_s
        )
    if settings.llm_provider == "bedrock":
        if not settings.bedrock_model_id:
            raise UnknownGeneratorError("LLM_PROVIDER=bedrock needs BEDROCK_MODEL_ID")
        return BedrockGenerator(
            settings.bedrock_model_id,
            region=settings.bedrock_region or None,
            max_tokens=settings.bedrock_max_tokens,
            temperature=settings.bedrock_temperature,
            timeout_s=settings.bedrock_timeout_s,
            max_retries=settings.bedrock_max_retries,
        )
    raise UnknownGeneratorError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r}; "
        "use 'extractive', 'openai_compatible' or 'bedrock'"
    )
