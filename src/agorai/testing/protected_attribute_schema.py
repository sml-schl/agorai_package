"""Protected Attribute Schema for Automated Counterfactual Testing.

This module defines the schema for identifying and systematically varying
protected attributes in images for causal robustness evaluation.
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class ProtectedAttributeType(str, Enum):
    """Categories of protected attributes that can be identified and modified."""

    # Demographics
    ETHNICITY = "ethnicity"
    GENDER = "gender"
    AGE = "age"
    RELIGION = "religion"

    # Visual Elements
    SKIN_TONE = "skin_tone"
    CLOTHING_STYLE = "clothing_style"
    RELIGIOUS_SYMBOLS = "religious_symbols"
    CULTURAL_SYMBOLS = "cultural_symbols"

    # Contextual
    SOCIOECONOMIC_INDICATORS = "socioeconomic_indicators"
    ABILITY_STATUS = "ability_status"
    BODY_TYPE = "body_type"

    # Objects/Background
    CULTURAL_OBJECTS = "cultural_objects"
    GEOGRAPHIC_INDICATORS = "geographic_indicators"


class ModificationStrategy(str, Enum):
    """Strategies for modifying protected attributes."""

    SWAP = "swap"  # Swap to specific alternative (e.g., male → female)
    REMOVE = "remove"  # Remove attribute entirely
    NEUTRALIZE = "neutralize"  # Make attribute neutral/ambiguous
    VARY_SPECTRUM = "vary_spectrum"  # Vary along spectrum (e.g., skin tone range)
    CROSS_CULTURAL = "cross_cultural"  # Swap to different cultural context


class ImageGenerationTool(str, Enum):
    """Supported image generation tools for counterfactual creation."""

    DALL_E_3 = "dall-e-3"
    STABLE_DIFFUSION_XL = "stable-diffusion-xl"
    MIDJOURNEY = "midjourney"
    REPLICATE = "replicate"
    HUGGINGFACE_INFERENCE = "huggingface-inference"
    LOCAL_DIFFUSION = "local-diffusion"


@dataclass
class ProtectedAttribute:
    """Represents a detected protected attribute in an image."""

    attribute_type: ProtectedAttributeType
    value: str  # e.g., "female", "elderly", "hijab"
    confidence: float  # 0.0-1.0
    bounding_box: Optional[Dict[str, float]] = None  # {x, y, width, height}
    description: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class CounterfactualVariation:
    """Defines how to create a counterfactual by modifying a protected attribute."""

    original_attribute: ProtectedAttribute
    target_value: str  # What to change it to
    modification_strategy: ModificationStrategy
    prompt_template: str  # Template for image generation prompt
    preserve_elements: List[str] = field(default_factory=list)  # Elements to keep unchanged

    def generate_prompt(self, original_description: str) -> str:
        """Generate the modification prompt for image generation tool."""
        return self.prompt_template.format(
            original=self.original_attribute.value,
            target=self.target_value,
            description=original_description,
            preserve=", ".join(self.preserve_elements)
        )


class ProtectedAttributeDetectionRequest(BaseModel):
    """Request to detect protected attributes in an image."""

    image_url: Optional[str] = Field(None, description="URL to the image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    text_content: Optional[str] = Field(None, description="Associated text (for multimodal)")

    # Detection parameters
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    attribute_types: Optional[List[ProtectedAttributeType]] = Field(
        None,
        description="Specific attributes to detect. If None, detect all."
    )
    include_bounding_boxes: bool = Field(True)

    class Config:
        use_enum_values = True


class ProtectedAttributeDetectionResponse(BaseModel):
    """Response containing detected protected attributes."""

    detected_attributes: List[ProtectedAttribute]
    image_description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class CounterfactualGenerationRequest(BaseModel):
    """Request to generate counterfactual image by modifying protected attributes."""

    original_image_url: Optional[str] = None
    original_image_base64: Optional[str] = None
    original_text: Optional[str] = None

    # Attribute modifications
    variations: List[CounterfactualVariation]

    # Generation parameters
    generation_tool: ImageGenerationTool
    tool_config: Dict[str, Any] = Field(default_factory=dict)

    # Quality parameters
    preserve_composition: bool = Field(True, description="Keep overall composition")
    preserve_style: bool = Field(True, description="Keep artistic style")
    preserve_background: bool = Field(True, description="Keep background unchanged")

    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


class CounterfactualGenerationResponse(BaseModel):
    """Response containing generated counterfactual image."""

    counterfactual_image_url: Optional[str] = None
    counterfactual_image_base64: Optional[str] = None
    generation_prompt: str
    modifications_applied: List[Dict[str, str]]
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CounterfactualTestCase(BaseModel):
    """Complete test case for causal robustness evaluation."""

    test_id: str
    original_image_url: Optional[str] = None
    original_image_base64: Optional[str] = None
    original_text: Optional[str] = None
    original_label: Optional[str] = None  # Ground truth label (if available)

    # Detected attributes
    protected_attributes: List[ProtectedAttribute]

    # Counterfactuals
    counterfactuals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of counterfactual variations"
    )

    # Evaluation results (populated after testing)
    original_prediction: Optional[Dict[str, Any]] = None
    counterfactual_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    consistency_score: Optional[float] = None
    spurious_correlation_detected: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True


# Predefined counterfactual variation templates

ETHNICITY_VARIATIONS = {
    "african": ["european", "asian", "middle_eastern", "latino"],
    "european": ["african", "asian", "middle_eastern", "latino"],
    "asian": ["african", "european", "middle_eastern", "latino"],
    "middle_eastern": ["african", "european", "asian", "latino"],
    "latino": ["african", "european", "asian", "middle_eastern"],
}

GENDER_VARIATIONS = {
    "male": ["female", "non-binary"],
    "female": ["male", "non-binary"],
    "non-binary": ["male", "female"],
}

AGE_VARIATIONS = {
    "child": ["teenager", "young_adult", "middle_aged", "elderly"],
    "teenager": ["child", "young_adult", "middle_aged", "elderly"],
    "young_adult": ["child", "teenager", "middle_aged", "elderly"],
    "middle_aged": ["child", "teenager", "young_adult", "elderly"],
    "elderly": ["child", "teenager", "young_adult", "middle_aged"],
}

SKIN_TONE_VARIATIONS = {
    "very_light": ["light", "medium", "dark", "very_dark"],
    "light": ["very_light", "medium", "dark", "very_dark"],
    "medium": ["very_light", "light", "dark", "very_dark"],
    "dark": ["very_light", "light", "medium", "very_dark"],
    "very_dark": ["very_light", "light", "medium", "dark"],
}

RELIGIOUS_SYMBOL_VARIATIONS = {
    "hijab": ["christian_cross", "jewish_kippah", "sikh_turban", "no_religious_symbol"],
    "christian_cross": ["hijab", "jewish_kippah", "sikh_turban", "no_religious_symbol"],
    "jewish_kippah": ["hijab", "christian_cross", "sikh_turban", "no_religious_symbol"],
    "sikh_turban": ["hijab", "christian_cross", "jewish_kippah", "no_religious_symbol"],
}


def get_standard_variations(
    attribute_type: ProtectedAttributeType,
    current_value: str
) -> List[str]:
    """Get standard counterfactual variations for a protected attribute.

    Args:
        attribute_type: Type of protected attribute
        current_value: Current value of the attribute

    Returns:
        List of alternative values to test
    """
    variation_map = {
        ProtectedAttributeType.ETHNICITY: ETHNICITY_VARIATIONS,
        ProtectedAttributeType.GENDER: GENDER_VARIATIONS,
        ProtectedAttributeType.AGE: AGE_VARIATIONS,
        ProtectedAttributeType.SKIN_TONE: SKIN_TONE_VARIATIONS,
        ProtectedAttributeType.RELIGIOUS_SYMBOLS: RELIGIOUS_SYMBOL_VARIATIONS,
    }

    variations = variation_map.get(attribute_type, {})
    return variations.get(current_value.lower(), [])


# Prompt templates for image generation

SWAP_PROMPT_TEMPLATE = """Generate an image that is identical to the original image, except:
- Change the {attribute_type} from "{original}" to "{target}"
- Preserve all other elements: {preserve}
- Maintain the same composition, style, and background
- Keep the same pose, expression, and context

Original description: {description}

Modified instruction: Replace {original} with {target} while keeping everything else exactly the same.
"""

REMOVE_PROMPT_TEMPLATE = """Generate an image that is identical to the original image, except:
- Remove the {attribute_type} "{original}"
- Preserve all other elements: {preserve}
- Maintain the same composition, style, and background
- Keep the overall scene intact

Original description: {description}

Modified instruction: Remove {original} while keeping everything else exactly the same.
"""

NEUTRALIZE_PROMPT_TEMPLATE = """Generate an image that is identical to the original image, except:
- Make the {attribute_type} neutral/ambiguous (currently "{original}")
- Preserve all other elements: {preserve}
- Maintain the same composition, style, and background

Original description: {description}

Modified instruction: Make {attribute_type} ambiguous while keeping everything else exactly the same.
"""


def create_counterfactual_variation(
    attribute: ProtectedAttribute,
    target_value: str,
    strategy: ModificationStrategy = ModificationStrategy.SWAP,
    preserve_elements: Optional[List[str]] = None
) -> CounterfactualVariation:
    """Create a counterfactual variation configuration.

    Args:
        attribute: The protected attribute to modify
        target_value: What to change it to
        strategy: How to modify it
        preserve_elements: Elements to keep unchanged

    Returns:
        CounterfactualVariation configuration
    """
    if preserve_elements is None:
        preserve_elements = ["text", "background", "composition", "style"]

    prompt_templates = {
        ModificationStrategy.SWAP: SWAP_PROMPT_TEMPLATE,
        ModificationStrategy.REMOVE: REMOVE_PROMPT_TEMPLATE,
        ModificationStrategy.NEUTRALIZE: NEUTRALIZE_PROMPT_TEMPLATE,
    }

    template = prompt_templates.get(strategy, SWAP_PROMPT_TEMPLATE)

    return CounterfactualVariation(
        original_attribute=attribute,
        target_value=target_value,
        modification_strategy=strategy,
        prompt_template=template,
        preserve_elements=preserve_elements
    )
