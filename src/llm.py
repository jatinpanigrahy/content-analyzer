"""LLM inference and prompt management module for Content Analyzer.

Interfaces with the Google GenAI SDK to generate structured insights
across multiple analysis modes, featuring cached model discovery,
adaptive routing, and cascading error fallback.
"""

import re
import time
from google import genai
from google.genai import errors

ANALYSIS_PROMPTS: dict[str, str] = {
    "Core Summary": (
        "You are an objective editorial analyst. Produce a rigorous, structured Core Summary of the provided text.\n\n"
        "Execution Protocol:\n"
        "1. Isolate the central thesis and primary context.\n"
        "2. Synthesize the supporting arguments into a coherent narrative.\n"
        "3. Extract the ultimate conclusion or outcome.\n\n"
        "Output Structure:\n"
        "- ## Overview: A concise 2-3 sentence paragraph establishing the core premise.\n"
        "- ## Key Arguments: 2-3 focused paragraphs synthesizing the body.\n"
        "- ## Core Conclusion: A decisive closing statement on significance or outcome.\n\n"
        "Constraints:\n"
        "- Strictly grounded in the provided text; do not assume external facts.\n"
        "- No conversational preambles or concluding remarks. Output markdown directly.\n\n"
        "Content:\n{content}"
    ),
    "Key Takeaways": (
        "You are a high-signal intelligence analyst. Extract the highest-leverage takeaways from the provided text.\n\n"
        "Execution Protocol:\n"
        "1. Filter out rhetorical noise, background filler, and standard boilerplate.\n"
        "2. Identify the 3 to 5 most impactful, non-obvious insights.\n"
        "3. Frame each takeaway with a bolded concept anchor followed by a concise explanation (max 2 sentences).\n\n"
        "Output Structure:\n"
        "- ## Key Takeaways\n"
        "  - **[Core Concept]**: Direct explanation of the insight and its significance.\n\n"
        "Constraints:\n"
        "- Exactly 3 to 5 bullet points.\n"
        "- Prioritize high-impact substance over surface-level descriptions.\n"
        "- Zero introductory or concluding filler.\n\n"
        "Content:\n{content}"
    ),
    "Action Items": (
        "You are an operational strategist. Translate the provided text into a prioritized, execution-ready action plan.\n\n"
        "Execution Protocol:\n"
        "1. Identify explicit directives, implied tasks, decisions, or procedural recommendations.\n"
        "2. Categorize them by logical sequence or priority (Immediate vs. Secondary).\n"
        "3. Express every item starting with an imperative action verb (e.g., 'Implement', 'Audit', 'Verify').\n\n"
        "Output Structure:\n"
        "- ## Priority Actions: Non-negotiable or immediate steps.\n"
        "- ## Recommended Next Steps: Supporting, long-term, or conditional actions.\n\n"
        "Constraints:\n"
        "- Every bullet must have an unambiguous scope of work.\n"
        "- If the text contains no direct actions, derive logical operational takeaways.\n"
        "- Output pure markdown only.\n\n"
        "Content:\n{content}"
    ),
    "Explain Like I'm 5": (
        "You are a master technical educator. Explain the core ideas in the provided text with total clarity and zero pretension.\n\n"
        "Execution Protocol:\n"
        "1. Identify the central concept and strip away technical jargon and acronyms.\n"
        "2. Anchor the explanation in a concrete, relatable real-world analogy.\n"
        "3. Explain how the pieces work together using simple, intuitive mechanics.\n\n"
        "Output Structure:\n"
        "- ## The Big Idea: The entire concept distilled into 1-2 simple sentences.\n"
        "- ## How It Works (The Analogy): A vivid everyday comparison that demystifies the mechanics.\n"
        "- ## Why It Matters: A grounded, plain-language closing takeaway.\n\n"
        "Constraints:\n"
        "- Simple, accessible vocabulary, but never patronizing or childish.\n"
        "- Do not lose technical accuracy—simplify the delivery, not the truth.\n"
        "- Zero introductory chatter.\n\n"
        "Content:\n{content}"
    ),
    "Blog Post Outline": (
        "You are an editorial architect. Transform the provided text into a publication-ready article outline.\n\n"
        "Execution Protocol:\n"
        "1. Formulate a strong, non-clickbait working title capturing the central insight.\n"
        "2. Build a logical narrative arc (Context -> Core Pillars -> Resolution).\n"
        "3. Provide distinct talking points and key arguments for every section.\n\n"
        "Output Structure:\n"
        "- # [Working Title]\n"
        "- ## 1. Introduction: Hook, core premise, and context.\n"
        "- ## 2. [Thematic Section]: Subpoints and evidence.\n"
        "- ## 3. [Thematic Section]: Deeper exploration and nuances.\n"
        "- ## 4. Conclusion & Takeaway: Key resolution and parting perspective.\n\n"
        "Constraints:\n"
        "- Include 2-3 specific bullet prompts under each major section header.\n"
        "- Maintain a mature, publication-grade editorial tone.\n"
        "- Output pure markdown only.\n\n"
        "Content:\n{content}"
    ),
    "Q&A Generator": (
        "You are a critical reviewer and examiner. Formulate high-value questions and authoritative answers based directly on the provided text.\n\n"
        "Execution Protocol:\n"
        "1. Identify the core premises, non-obvious nuances, and likely points of reader confusion.\n"
        "2. Formulate 3 to 5 targeted, insightful questions that probe the actual substance.\n"
        "3. Provide direct, factual answers grounded strictly in the text.\n\n"
        "Output Structure:\n"
        "- ### Q1: [Specific, substantive question]\n"
        "  **A:** [Direct, concise answer supported by the text.]\n\n"
        "Constraints:\n"
        "- Avoid trivial or superficial questions.\n"
        "- Answers must be fully supported by the source text without extrapolation.\n"
        "- Output pure markdown only.\n\n"
        "Content:\n{content}"
    ),
    "Professional Post": (
        "You are a high-signal professional writer. Convert the provided text into a mature, structured post suitable for a professional audience (LinkedIn / X).\n\n"
        "Execution Protocol:\n"
        "1. Craft an insight-driven opening line (hook) that highlights the core discovery or problem without clickbait.\n"
        "2. Present 2-4 tight, readable paragraphs or scannable bullet points detailing the key takeaways.\n"
        "3. Conclude with a reflective question that invites professional discussion.\n\n"
        "Constraints:\n"
        "- Maintain an authentic, authoritative tone. Avoid buzzwords and performative hype.\n"
        "- Emoji usage strictly limited to 1-2 functional accents (no emoji spam).\n"
        "- Maximum 2-3 targeted hashtags at the very bottom.\n"
        "- Output pure post text only.\n\n"
        "Content:\n{content}"
    ),
    "Casual Post": (
        "You are an articulate peer sharing an interesting finding with a community or group chat (e.g., Discord, WhatsApp, Slack, or an informal social note).\n\n"
        "Execution Protocol:\n"
        "1. Open with an approachable, natural hook that summarizes why this is interesting or worth reading.\n"
        "2. Distill the core points into 2-3 casual, easy-to-read sentences or conversational bullet points.\n"
        "3. Close with a light, natural sign-off or brief takeaway.\n\n"
        "Constraints:\n"
        "- Tone must be conversational, warm, and natural—like messaging a thoughtful colleague.\n"
        "- Maintain substance and maturity; avoid slang, exaggeration, or childish phrasing.\n"
        "- No corporate jargon or stiff formal phrasing.\n"
        "- Emoji Rule: Strictly minimal and meaningful. Use at most 1 to 2 emojis in the entire output, and only where they serve a functional purpose. Never place emojis consecutively, and never use hype emojis (e.g., no rocket, fire, or party icons). If in doubt, use zero emojis.\n"
        "- Output only the message text.\n\n"
        "Content:\n{content}"
    ),
}

AVAILABLE_MODES: list[str] = list(ANALYSIS_PROMPTS.keys())

# Curated fallback models ordered by capability tier if dynamic discovery is unavailable
FALLBACK_MODELS: list[str] = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-pro-latest",
]

# Tags indicating discontinued, specialized, non-text, or lightweight models excluded from inference
EXCLUDED_MODEL_TAGS: list[str] = [
    "lite",
    "tts",
    "image",
    "audio",
    "transcribe",
    "robotics",
    "banana",
    "lyria",
    "veo",
    "live",
    "customtools",
    "embed",
    "imagen",
    "-exp",
    "thinking",
    "computer-use",
    "aqa",
    "2.5",
    "2.0",
]

# Discontinued model endpoints excluded from candidate lists
DEPRECATED_MODELS: set[str] = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
}

# Cache duration for dynamic model discovery (1 hour)
CACHE_TTL_SECONDS: float = 3600.0

# In-memory discovery cache state
_CACHED_CANDIDATE_MODELS: list[str] = []
_CACHE_TIMESTAMP: float = 0.0


def build_prompt(content: str, mode: str) -> str:
    """Build the prompt template for a given mode and content.

    Args:
        content: The sanitized source text.
        mode: The selected analysis mode name.

    Returns:
        Formatted prompt string.

    Raises:
        ValueError: If mode is not recognized.
    """
    template = ANALYSIS_PROMPTS.get(mode)
    if not template:
        valid_modes = ", ".join(ANALYSIS_PROMPTS.keys())
        raise ValueError(f"Unknown mode '{mode}'. Expected one of: {valid_modes}")
    return template.replace("{content}", content)


def _model_sort_key(name: str) -> tuple[int, int, tuple[int, int]]:
    """Generate a comparison key prioritizing frontier tiers and newer versions.

    Prioritization hierarchy:
    1. Tier level: Pro models (tier 2) rank higher than Flash models (tier 1).
    2. Stability: Stable production releases rank higher than preview builds.
    3. Version number: Higher semantic version numbers rank higher.

    Args:
        name: The model identifier string.

    Returns:
        A tuple suitable for sorting models in descending order of capability.
    """
    lower = name.lower()
    tier = 2 if "pro" in lower else 1
    stable_priority = 0 if "preview" in lower else 1

    version_match = re.search(r"(\d+)(?:\.(\d+))?", lower)
    if version_match:
        major = int(version_match.group(1))
        minor = int(version_match.group(2)) if version_match.group(2) else 0
        version = (major, minor)
    elif "latest" in lower:
        version = (99, 0) if tier == 2 else (0, 0)
    else:
        version = (0, 0)

    return (tier, stable_priority, version)


def get_candidate_models(
    client: genai.Client,
    force_refresh: bool = False,
) -> list[str]:
    """Discover and prioritize top frontier models available from the Gemini API.

    Maintains an in-memory TTL cache to eliminate redundant network roundtrips.
    Filters out discontinued, lightweight, specialized, and non-text variants,
    sorting candidates by capability tier and version release.

    Args:
        client: The initialized Google GenAI client instance.
        force_refresh: When True, bypasses the in-memory cache and re-queries the API.

    Returns:
        A prioritized list of model identifier strings.
    """
    global _CACHED_CANDIDATE_MODELS, _CACHE_TIMESTAMP

    now = time.time()
    if (
        not force_refresh
        and _CACHED_CANDIDATE_MODELS
        and (now - _CACHE_TIMESTAMP) < CACHE_TTL_SECONDS
    ):
        return list(_CACHED_CANDIDATE_MODELS)

    try:
        discovered: list[str] = []
        for model in client.models.list():
            raw_name = getattr(model, "name", "")
            clean_name = raw_name.replace("models/", "").strip()
            name_lower = clean_name.lower()

            if not name_lower.startswith("gemini"):
                continue

            if clean_name in DEPRECATED_MODELS:
                continue

            if any(tag in name_lower for tag in EXCLUDED_MODEL_TAGS):
                continue

            if "pro" not in name_lower and "flash" not in name_lower:
                continue

            discovered.append(clean_name)

        if discovered:
            discovered.sort(key=_model_sort_key, reverse=True)
            for fallback in FALLBACK_MODELS:
                if fallback not in discovered and fallback not in DEPRECATED_MODELS:
                    discovered.append(fallback)

            _CACHED_CANDIDATE_MODELS = discovered
            _CACHE_TIMESTAMP = now
            return list(discovered)
    except Exception:
        pass

    fallback_list = [m for m in FALLBACK_MODELS if m not in DEPRECATED_MODELS]
    _CACHED_CANDIDATE_MODELS = fallback_list
    _CACHE_TIMESTAMP = now
    return list(fallback_list)


def generate_analysis(
    content: str,
    mode: str,
    api_key: str,
    model: str | None = None,
    preferred_model: str | None = None,
) -> tuple[str, str]:
    """Execute LLM inference for content analysis with automated failover.

    Evaluates candidate models in order of capability, with support for
    sticky preferred routing to avoid repeated fallback roundtrips.
    Automatically cascades to alternative models if capacity (503),
    quota limits (429), or endpoint errors (404) occur, adaptively
    deprioritizing problematic endpoints for subsequent calls.

    Args:
        content: The plain text content to analyze.
        mode: The desired analysis format.
        api_key: The Google Gemini API key.
        model: Optional explicit model override.
        preferred_model: Optional known-good model to attempt first before fallback.

    Returns:
        A tuple of (markdown_response_text, model_identifier_used).

    Raises:
        ValueError: If content or api_key is missing, or mode is invalid.
        google.genai.errors.APIError: If all candidate models fail due to API errors.
        RuntimeError: If model execution finishes without generating text.
    """
    global _CACHED_CANDIDATE_MODELS

    if not api_key:
        raise ValueError("GEMINI_API_KEY must be provided.")
    if not content or not content.strip():
        raise ValueError("Content cannot be empty.")

    prompt = build_prompt(content=content.strip(), mode=mode)
    client = genai.Client(api_key=api_key)

    if model:
        candidates = [model]
    else:
        candidates = get_candidate_models(client)
        if preferred_model:
            # Promote the preferred model to the front to eliminate failover latency
            candidates = [preferred_model] + [m for m in candidates if m != preferred_model]

    last_error: Exception | None = None

    for candidate in candidates:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
            )
            if response.text and response.text.strip():
                return response.text, candidate
        except errors.APIError as exc:
            last_error = exc
            error_code = getattr(exc, "code", None)
            error_msg = str(exc)

            # Evict discontinued 404 endpoints so they are never retried
            if (error_code == 404 or "404" in error_msg) and candidate in _CACHED_CANDIDATE_MODELS:
                while candidate in _CACHED_CANDIDATE_MODELS:
                    _CACHED_CANDIDATE_MODELS.remove(candidate)

            # Demote rate-limited (429) or high-demand (503) endpoints to the end of cache
            if (
                error_code in (429, 503)
                or "429" in error_msg
                or "503" in error_msg
            ) and candidate in _CACHED_CANDIDATE_MODELS:
                while candidate in _CACHED_CANDIDATE_MODELS:
                    _CACHED_CANDIDATE_MODELS.remove(candidate)
                _CACHED_CANDIDATE_MODELS.append(candidate)

            # Cascade across transient capacity, rate limit, or endpoint errors
            is_transient = (
                error_code in (404, 429, 503)
                or "503" in error_msg
                or "429" in error_msg
                or "404" in error_msg
            )
            if is_transient and candidate != candidates[-1]:
                continue
            raise exc
        except Exception as exc:
            last_error = exc
            if candidate != candidates[-1]:
                continue
            raise exc

    if last_error:
        raise last_error

    raise RuntimeError("Model evaluation completed without generating response text.")
