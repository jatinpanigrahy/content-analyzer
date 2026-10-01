"""Unit test suite for LLM inference and prompt management.

Validates prompt compilation, candidate model discovery sorting,
TTL discovery caching, preferred model routing, exception cascading,
and input validation.
All API interactions are mocked to ensure isolated, deterministic execution.
"""

from pathlib import Path
import sys
import time
from unittest.mock import MagicMock, patch
import pytest
from google.genai import errors

# Ensure project root is in sys.path for direct pytest invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.llm as llm
from src.llm import (
    ANALYSIS_PROMPTS,
    AVAILABLE_MODES,
    FALLBACK_MODELS,
    _model_sort_key,
    build_prompt,
    generate_analysis,
    get_candidate_models,
)


@pytest.fixture(autouse=True)
def reset_llm_cache() -> None:
    """Reset module-level discovery cache before each test."""
    llm._CACHED_CANDIDATE_MODELS = []
    llm._CACHE_TIMESTAMP = 0.0


# ---------------------------------------------------------------------------
# Prompt Building Tests
# ---------------------------------------------------------------------------


def test_build_prompt_all_available_modes() -> None:
    """Validate that prompt templates for all 8 modes correctly inject content."""
    sample_content = "Distributed consensus requires leader election and quorum."
    for mode in AVAILABLE_MODES:
        prompt = build_prompt(content=sample_content, mode=mode)
        assert sample_content in prompt
        assert "{content}" not in prompt


def test_build_prompt_curly_brace_safety() -> None:
    """Validate that source content containing literal curly braces does not fail."""
    content_with_braces = "function solve() { return {status: 200, values: [1, 2]}; }"
    prompt = build_prompt(content=content_with_braces, mode="Core Summary")
    assert content_with_braces in prompt


def test_build_prompt_invalid_mode_raises_value_error() -> None:
    """Validate that unrecognized mode strings trigger ValueError."""
    with pytest.raises(ValueError, match="Unknown mode 'UnsupportedMode'"):
        build_prompt(content="Sample text", mode="UnsupportedMode")


# ---------------------------------------------------------------------------
# Model Sort Key & Prioritization Tests
# ---------------------------------------------------------------------------


def test_model_sort_key_prioritization() -> None:
    """Validate sorting hierarchy: Pro > Flash, Stable > Preview, Higher version > Lower."""
    models = [
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3-flash-preview",
        "gemini-3.1-pro-preview",
        "gemini-pro-latest",
    ]
    sorted_models = sorted(models, key=_model_sort_key, reverse=True)

    # Pro models should come before Flash models
    assert sorted_models[0] == "gemini-pro-latest"
    assert sorted_models[1] == "gemini-3.1-pro-preview"

    # Higher version Flash should precede lower version Flash
    assert sorted_models[2] == "gemini-3.8-flash"
    assert sorted_models[3] == "gemini-3.5-flash"

    # Preview Flash should rank below stable versioned Flash
    assert sorted_models[4] == "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Model Discovery & Caching Tests
# ---------------------------------------------------------------------------


def test_get_candidate_models_filtering() -> None:
    """Validate that excluded tags and deprecated models are filtered out."""
    raw_model_names = [
        "models/gemini-2.5-pro",          # Deprecated (404)
        "models/gemini-2.5-flash",        # Deprecated (404)
        "models/gemini-3.8-flash-lite",   # Excluded tag: lite
        "models/gemini-3.8-flash-tts",    # Excluded tag: tts
        "models/gemini-3.8-flash-image",  # Excluded tag: image
        "models/gemini-3.8-flash",        # Valid candidate
        "models/gemini-3.6-flash",        # Valid candidate
        "models/gemini-pro-latest",       # Valid candidate
    ]

    mock_client = MagicMock()
    mock_models = [MagicMock(name=name) for name in raw_model_names]
    for m, name in zip(mock_models, raw_model_names):
        m.name = name
    mock_client.models.list.return_value = mock_models

    candidates = get_candidate_models(mock_client)

    assert "gemini-pro-latest" in candidates
    assert "gemini-3.8-flash" in candidates
    assert "gemini-3.6-flash" in candidates

    assert "gemini-2.5-pro" not in candidates
    assert "gemini-2.5-flash" not in candidates
    assert "gemini-3.8-flash-lite" not in candidates
    assert "gemini-3.8-flash-tts" not in candidates
    assert "gemini-3.8-flash-image" not in candidates


def test_get_candidate_models_ttl_caching() -> None:
    """Validate that candidate discovery results are cached in-memory."""
    mock_client = MagicMock()
    mock_models = [MagicMock(name="models/gemini-3.8-flash")]
    mock_models[0].name = "models/gemini-3.8-flash"
    mock_client.models.list.return_value = mock_models

    # First call queries the API
    first_result = get_candidate_models(mock_client)
    assert mock_client.models.list.call_count == 1

    # Second call uses the cached list without re-querying
    second_result = get_candidate_models(mock_client)
    assert mock_client.models.list.call_count == 1
    assert first_result == second_result

    # Force refresh bypasses cache
    third_result = get_candidate_models(mock_client, force_refresh=True)
    assert mock_client.models.list.call_count == 2
    assert third_result == first_result


def test_get_candidate_models_api_failure_returns_fallback() -> None:
    """Validate that API discovery failures fall back to curated defaults."""
    mock_client = MagicMock()
    mock_client.models.list.side_effect = RuntimeError("API unavailable")

    candidates = get_candidate_models(mock_client)
    assert len(candidates) > 0
    assert "gemini-2.5-pro" not in candidates
    assert candidates == [m for m in FALLBACK_MODELS if m not in llm.DEPRECATED_MODELS]


# ---------------------------------------------------------------------------
# Inference & Cascading Fallback Tests
# ---------------------------------------------------------------------------


def test_generate_analysis_missing_api_key_raises_value_error() -> None:
    """Validate that empty or missing API keys raise ValueError."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY must be provided"):
        generate_analysis(content="Valid content", mode="Core Summary", api_key="")


def test_generate_analysis_empty_content_raises_value_error() -> None:
    """Validate that whitespace-only content raises ValueError."""
    with pytest.raises(ValueError, match="Content cannot be empty"):
        generate_analysis(content="   ", mode="Core Summary", api_key="valid-key")


def test_generate_analysis_explicit_model_override() -> None:
    """Validate that passing an explicit model targets only that model."""
    mock_response = MagicMock()
    mock_response.text = "Analysis output text"

    with patch("src.llm.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = mock_response

        text, model_used = generate_analysis(
            content="Sample text",
            mode="Core Summary",
            api_key="test-key",
            model="custom-gemini-model",
        )

        assert text == "Analysis output text"
        assert model_used == "custom-gemini-model"
        mock_client.models.generate_content.assert_called_once()
        assert mock_client.models.generate_content.call_args.kwargs["model"] == "custom-gemini-model"


def test_generate_analysis_preferred_model_routing() -> None:
    """Validate that preferred_model is prioritized at the front of candidate evaluation."""
    mock_response = MagicMock()
    mock_response.text = "Sticky routed output"

    with patch("src.llm.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.llm.get_candidate_models", return_value=["model-b", "model-c", "model-a"]):
            text, model_used = generate_analysis(
                content="Sample text",
                mode="Core Summary",
                api_key="test-key",
                preferred_model="model-a",
            )

            assert model_used == "model-a"
            assert mock_client.models.generate_content.call_args.kwargs["model"] == "model-a"


def test_generate_analysis_cascading_failover_on_rate_limit() -> None:
    """Validate that 429 quota exhaustion triggers fallback to next candidate and demotes the failed model."""
    llm._CACHED_CANDIDATE_MODELS = ["rate-limited-model", "healthy-model"]
    llm._CACHE_TIMESTAMP = time.time()

    mock_response = MagicMock()
    mock_response.text = "Healthy output after failover"

    with patch("src.llm.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value

        # First call raises 429 APIError, second call succeeds
        rate_limit_err = errors.APIError(429, "Rate limit exceeded")
        mock_client.models.generate_content.side_effect = [rate_limit_err, mock_response]

        with patch("src.llm.get_candidate_models", return_value=["rate-limited-model", "healthy-model"]):
            text, model_used = generate_analysis(
                content="Sample text",
                mode="Core Summary",
                api_key="test-key",
            )

            assert text == "Healthy output after failover"
            assert model_used == "healthy-model"
            assert mock_client.models.generate_content.call_count == 2

            # The 429 model should be demoted to the end of the cached candidate queue
            assert llm._CACHED_CANDIDATE_MODELS[-1] == "rate-limited-model"


def test_generate_analysis_evicts_404_model_from_cache() -> None:
    """Validate that 404 missing endpoint errors evict the model from cache."""
    llm._CACHED_CANDIDATE_MODELS = ["deprecated-404-model", "healthy-model"]
    llm._CACHE_TIMESTAMP = time.time()

    mock_response = MagicMock()
    mock_response.text = "Healthy output"

    with patch("src.llm.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        not_found_err = errors.APIError(404, "Endpoint not found")
        mock_client.models.generate_content.side_effect = [not_found_err, mock_response]

        with patch("src.llm.get_candidate_models", return_value=["deprecated-404-model", "healthy-model"]):
            text, model_used = generate_analysis(
                content="Sample text",
                mode="Core Summary",
                api_key="test-key",
            )

            assert model_used == "healthy-model"
            # 404 model must be completely evicted from cache
            assert "deprecated-404-model" not in llm._CACHED_CANDIDATE_MODELS


def test_generate_analysis_non_transient_error_raises_immediately() -> None:
    """Validate that fatal non-transient API errors (e.g. 400 Bad Request) do not cascade."""
    with patch("src.llm.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        bad_request_err = errors.APIError(400, "Bad Request: Invalid argument")
        mock_client.models.generate_content.side_effect = bad_request_err

        with patch("src.llm.get_candidate_models", return_value=["model-1", "model-2"]):
            with pytest.raises(errors.APIError) as exc_info:
                generate_analysis(
                    content="Sample text",
                    mode="Core Summary",
                    api_key="test-key",
                )

            assert exc_info.value.code == 400
            # Should have terminated immediately on the first attempt without cascading
            assert mock_client.models.generate_content.call_count == 1


def test_generate_analysis_empty_response_raises_runtime_error() -> None:
    """Validate that empty model responses raise RuntimeError."""
    mock_response = MagicMock()
    mock_response.text = ""

    with patch("src.llm.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.llm.get_candidate_models", return_value=["single-model"]):
            with pytest.raises(RuntimeError, match="without generating response text"):
                generate_analysis(
                    content="Sample text",
                    mode="Core Summary",
                    api_key="test-key",
                )
