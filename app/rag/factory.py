"""Build the configured answer generator."""

from app.core.config import Settings
from app.rag.generator import AnswerGenerator, ExtractiveGenerator, OpenAICompatibleGenerator


class UnknownGeneratorError(ValueError):
    pass


def get_generator(settings: Settings) -> AnswerGenerator:
    if settings.llm_provider == "extractive":
        return ExtractiveGenerator()
    if settings.llm_provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_model:
            raise UnknownGeneratorError(
                "LLM_PROVIDER=openai_compatible needs LLM_BASE_URL and LLM_MODEL"
            )
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        return OpenAICompatibleGenerator(
            settings.llm_base_url, settings.llm_model, key, settings.llm_timeout_s
        )
    raise UnknownGeneratorError(
        f"unknown LLM_PROVIDER {settings.llm_provider!r}; use 'extractive' or 'openai_compatible'"
    )
