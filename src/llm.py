"""LLM inference and prompt management module.

Interfaces with the Google Gemini GenAI SDK to generate structured
insights across multiple analysis modes.
"""

from google import genai

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
        "3. Conclude with a grounded, reflective question that invites professional discussion.\n\n"
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
        "- Emoji Rule: Strictly minimal and meaningful. Use at most 1 to 2 emojis in the entire output, and only where they serve a functional purpose. Never place emojis consecutively, and never use hype emojis (e.g., no 🚀, 🔥, or 🎉). If in doubt, use zero emojis.\n"
        "- Output only the message text.\n\n"
        "Content:\n{content}"
    ),
}

AVAILABLE_MODES: list[str] = list(ANALYSIS_PROMPTS.keys())


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
    return template.format(content=content)


def generate_analysis(
    content: str,
    mode: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """Call Google Gemini to analyze content using the selected mode.

    Args:
        content: The plain text content to analyze.
        mode: The desired analysis format.
        api_key: The Google Gemini API key.
        model: Gemini model identifier.

    Returns:
        The markdown-formatted response string from the model.

    Raises:
        ValueError: If content or api_key is missing, or mode is invalid.
        google.genai.errors.APIError: If the Gemini API request fails.
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be provided.")
    if not content or not content.strip():
        raise ValueError("Content cannot be empty.")

    prompt = build_prompt(content=content.strip(), mode=mode)
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    if not response.text:
        raise ValueError("Model returned an empty response.")

    return response.text
