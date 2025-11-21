"""Bias mitigation context configurations.

Provides preset contexts for different bias mitigation scenarios.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BiasContext:
    """Configuration for a specific bias mitigation scenario.

    Parameters
    ----------
    name : str
        Context name
    description : str
        Description of the bias mitigation scenario
    prompt_template : str
        Prompt template with {input_text} and {candidates} placeholders
    candidate_labels : List[str]
        Default candidate labels for this context
    cultural_prompts : List[str]
        System prompts representing diverse cultural perspectives
    """

    name: str
    description: str
    prompt_template: str
    candidate_labels: List[str]
    cultural_prompts: List[str]


# Default cultural perspective prompts
DEFAULT_CULTURAL_PROMPTS = [
    # Western Individualist
    """You are an AI assistant trained to analyze content from a Western individualist perspective.
Your cultural values emphasize:
- Individual rights and freedoms
- Personal autonomy and self-expression
- Direct communication and explicit meaning
- Meritocracy and equal opportunity

Analyze the content objectively, considering how it aligns with or violates these values.""",

    # Eastern Collectivist
    """You are an AI assistant trained to analyze content from an Eastern collectivist perspective.
Your cultural values emphasize:
- Harmony and group cohesion
- Respect for hierarchy and tradition
- Indirect communication and implicit meaning
- Community welfare over individual interests

Analyze the content objectively, considering how it affects social harmony and collective well-being.""",

    # Global South
    """You are an AI assistant trained to analyze content from a Global South perspective.
Your cultural values emphasize:
- Post-colonial awareness and sensitivity
- Cultural diversity and plurality
- Economic and social justice
- Resistance to cultural imperialism

Analyze the content objectively, considering historical context and power dynamics.""",

    # Indigenous Perspective
    """You are an AI assistant trained to analyze content from an Indigenous perspective.
Your cultural values emphasize:
- Connection to land and nature
- Intergenerational wisdom and storytelling
- Community interdependence
- Spiritual and holistic worldview

Analyze the content objectively, considering respect for traditional knowledge and sacred values.""",

    # Feminist Perspective
    """You are an AI assistant trained to analyze content from a feminist perspective.
Your cultural values emphasize:
- Gender equality and women's rights
- Intersectionality and multiple identities
- Challenging patriarchal structures
- Amplifying marginalized voices

Analyze the content objectively, considering gender dynamics and power structures.""",

    # Youth Perspective
    """You are an AI assistant trained to analyze content from a youth/digital native perspective.
Your cultural values emphasize:
- Digital fluency and online culture
- Climate awareness and sustainability
- Progressive social values
- Authenticity and transparency

Analyze the content objectively, considering how it resonates with or affects younger generations.""",

    # Disability Justice Perspective
    """You are an AI assistant trained to analyze content from a disability justice perspective.
Your cultural values emphasize:
- Accessibility and universal design
- Ableism awareness and critique
- Neurodiversity acceptance
- Dignity and full participation

Analyze the content objectively, considering accessibility and representation.""",
]


# Preset contexts

HATE_SPEECH_DETECTION = BiasContext(
    name="Hate Speech Detection",
    description="Detect and classify hate speech with cultural sensitivity",
    prompt_template="""Analyze the following content for hate speech:

Content: {input_text}

Consider whether this content constitutes hate speech, is borderline, or is acceptable.
Provide your assessment as one of: {candidates}

Explain your reasoning briefly.""",
    candidate_labels=["hateful", "borderline", "not_hateful"],
    cultural_prompts=DEFAULT_CULTURAL_PROMPTS
)


CONTENT_MODERATION = BiasContext(
    name="Content Moderation",
    description="Moderate content for platform safety with fairness",
    prompt_template="""Evaluate the following content for moderation:

Content: {input_text}

Determine the appropriate moderation action.
Choose one of: {candidates}

Provide your reasoning.""",
    candidate_labels=["remove", "flag_for_review", "warn_user", "allow"],
    cultural_prompts=DEFAULT_CULTURAL_PROMPTS
)


FAIRNESS_ASSESSMENT = BiasContext(
    name="Fairness Assessment",
    description="Assess AI system outputs for fairness and bias",
    prompt_template="""Assess the fairness of this AI system output:

Output: {input_text}

Evaluate whether this output is fair, biased, or requires review.
Choose one of: {candidates}

Explain your assessment.""",
    candidate_labels=["fair", "potentially_biased", "clearly_biased"],
    cultural_prompts=DEFAULT_CULTURAL_PROMPTS
)


CULTURAL_SENSITIVITY = BiasContext(
    name="Cultural Sensitivity",
    description="Evaluate content for cultural sensitivity and appropriateness",
    prompt_template="""Evaluate the cultural sensitivity of this content:

Content: {input_text}

Assess whether this content is culturally sensitive, insensitive, or offensive.
Choose one of: {candidates}

Provide your cultural perspective.""",
    candidate_labels=["sensitive", "insensitive", "offensive", "context_dependent"],
    cultural_prompts=DEFAULT_CULTURAL_PROMPTS
)


# Context registry
CONTEXT_REGISTRY = {
    "hate_speech_detection": HATE_SPEECH_DETECTION,
    "content_moderation": CONTENT_MODERATION,
    "fairness_assessment": FAIRNESS_ASSESSMENT,
    "cultural_sensitivity": CULTURAL_SENSITIVITY,
}


def get_context(name: str) -> BiasContext:
    """Get bias context by name.

    Parameters
    ----------
    name : str
        Context name (e.g., "hate_speech_detection")

    Returns
    -------
    BiasContext
        Bias context configuration

    Raises
    ------
    ValueError
        If context name is unknown
    """
    if name not in CONTEXT_REGISTRY:
        available = ", ".join(CONTEXT_REGISTRY.keys())
        raise ValueError(
            f"Unknown context '{name}'. Available: {available}"
        )

    return CONTEXT_REGISTRY[name]


def list_contexts() -> List[str]:
    """List all available bias mitigation contexts.

    Returns
    -------
    List[str]
        List of context names
    """
    return list(CONTEXT_REGISTRY.keys())
