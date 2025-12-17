"""Protected Attribute Detector using Vision-Language Models.

Identifies protected attributes (ethnicity, gender, age, religious symbols, etc.)
in images for automated counterfactual testing.
"""

import base64
import json
from typing import List, Dict, Any, Optional
from io import BytesIO

from ..agents import BaseAgent
from .protected_attribute_schema import (
    ProtectedAttribute,
    ProtectedAttributeType,
    ProtectedAttributeDetectionRequest,
    ProtectedAttributeDetectionResponse,
)


# Detection prompt template
ATTRIBUTE_DETECTION_PROMPT = """Analyze this image and identify any PROTECTED ATTRIBUTES that are visible.

Protected attributes include:
1. Demographics: ethnicity, gender, age, religion
2. Visual markers: skin tone, clothing style, religious symbols, cultural symbols
3. Contextual indicators: socioeconomic indicators, ability status, body type
4. Objects/background: cultural objects, geographic indicators

For each protected attribute you detect:
- Specify the attribute type (e.g., "ethnicity", "gender", "age")
- Describe the specific value (e.g., "female", "elderly", "hijab")
- Provide a confidence score (0.0-1.0)
- If possible, describe the location in the image

IMPORTANT:
- Be objective and descriptive, not evaluative
- Focus on visible attributes only
- Err on the side of caution for ambiguous attributes
- Consider cultural sensitivity

Also provide:
- A brief overall description of the image
- Any text visible in the image

Respond in JSON format:
{{
  "image_description": "Brief description of the image",
  "visible_text": "Any text in the image",
  "protected_attributes": [
    {{
      "attribute_type": "type",
      "value": "specific value",
      "confidence": 0.0-1.0,
      "location": "description of where in image",
      "description": "brief context"
    }}
  ]
}}

If there is also text content associated with this image (like a meme caption), it will be provided below:
{text_content}
"""


class ProtectedAttributeDetector:
    """Detects protected attributes in images using VLMs."""

    def __init__(
        self,
        vlm_agent: BaseAgent,
        default_confidence_threshold: float = 0.5
    ):
        """Initialize the detector.

        Args:
            vlm_agent: Vision-language model agent for analysis
            default_confidence_threshold: Minimum confidence for detection
        """
        self.vlm_agent = vlm_agent
        self.default_confidence_threshold = default_confidence_threshold

    def detect(
        self,
        request: ProtectedAttributeDetectionRequest
    ) -> ProtectedAttributeDetectionResponse:
        """Detect protected attributes in an image.

        Args:
            request: Detection request with image and parameters

        Returns:
            Detection response with identified attributes
        """
        # Prepare detection prompt
        text_content_str = request.text_content or "No text content provided."
        prompt = ATTRIBUTE_DETECTION_PROMPT.format(text_content=text_content_str)

        # Prepare context with image
        context = {}
        if request.image_url:
            context["image_url"] = request.image_url
        elif request.image_base64:
            context["image_base64"] = request.image_base64
        else:
            raise ValueError("Either image_url or image_base64 must be provided")

        # Call VLM
        response = self.vlm_agent.generate(prompt, context=context)

        # Parse response
        detected_attrs = self._parse_detection_response(
            response,
            confidence_threshold=request.confidence_threshold,
            requested_types=request.attribute_types
        )

        return detected_attrs

    def _parse_detection_response(
        self,
        vlm_response: Dict[str, Any],
        confidence_threshold: float,
        requested_types: Optional[List[ProtectedAttributeType]]
    ) -> ProtectedAttributeDetectionResponse:
        """Parse VLM response into structured detection result.

        Args:
            vlm_response: Raw response from VLM
            confidence_threshold: Minimum confidence to include
            requested_types: Specific types to filter for

        Returns:
            Structured detection response
        """
        # Extract text from response
        response_text = vlm_response.get("text", "")

        # Try to extract JSON from response
        try:
            # Find JSON block in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
            else:
                # Fallback: parse entire response
                parsed = json.loads(response_text)

        except json.JSONDecodeError:
            # Fallback: empty detection if parsing fails
            return ProtectedAttributeDetectionResponse(
                detected_attributes=[],
                image_description="Failed to parse VLM response",
                metadata={"raw_response": response_text, "error": "JSON parse failed"}
            )

        # Extract image description
        image_description = parsed.get("image_description", "")

        # Parse detected attributes
        detected_attributes = []
        for attr_data in parsed.get("protected_attributes", []):
            # Check confidence threshold
            confidence = float(attr_data.get("confidence", 0.0))
            if confidence < confidence_threshold:
                continue

            # Parse attribute type
            attr_type_str = attr_data.get("attribute_type", "").lower()
            try:
                attr_type = ProtectedAttributeType(attr_type_str)
            except ValueError:
                # Try to map common variants
                type_mapping = {
                    "race": ProtectedAttributeType.ETHNICITY,
                    "skin_color": ProtectedAttributeType.SKIN_TONE,
                    "sex": ProtectedAttributeType.GENDER,
                    "clothing": ProtectedAttributeType.CLOTHING_STYLE,
                }
                attr_type = type_mapping.get(attr_type_str)
                if not attr_type:
                    continue  # Skip unknown types

            # Check if this type was requested
            if requested_types and attr_type not in requested_types:
                continue

            # Create ProtectedAttribute object
            attribute = ProtectedAttribute(
                attribute_type=attr_type,
                value=attr_data.get("value", ""),
                confidence=confidence,
                description=attr_data.get("description")
            )

            detected_attributes.append(attribute)

        return ProtectedAttributeDetectionResponse(
            detected_attributes=detected_attributes,
            image_description=image_description,
            metadata={
                "visible_text": parsed.get("visible_text", ""),
                "raw_response": response_text
            }
        )

    def detect_from_url(
        self,
        image_url: str,
        text_content: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> ProtectedAttributeDetectionResponse:
        """Convenience method to detect from image URL.

        Args:
            image_url: URL to the image
            text_content: Associated text (e.g., meme caption)
            confidence_threshold: Override default threshold

        Returns:
            Detection response
        """
        threshold = confidence_threshold or self.default_confidence_threshold

        request = ProtectedAttributeDetectionRequest(
            image_url=image_url,
            text_content=text_content,
            confidence_threshold=threshold
        )

        return self.detect(request)

    def detect_from_base64(
        self,
        image_base64: str,
        text_content: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> ProtectedAttributeDetectionResponse:
        """Convenience method to detect from base64 image.

        Args:
            image_base64: Base64 encoded image
            text_content: Associated text (e.g., meme caption)
            confidence_threshold: Override default threshold

        Returns:
            Detection response
        """
        threshold = confidence_threshold or self.default_confidence_threshold

        request = ProtectedAttributeDetectionRequest(
            image_base64=image_base64,
            text_content=text_content,
            confidence_threshold=threshold
        )

        return self.detect(request)


def create_detector_from_config(config: Dict[str, Any]) -> ProtectedAttributeDetector:
    """Factory function to create detector from configuration.

    Args:
        config: Configuration dict with agent and threshold settings

    Returns:
        Configured ProtectedAttributeDetector

    Example config:
        {
            "vlm_model": "gpt-4-vision-preview",
            "vlm_provider": "openai",
            "confidence_threshold": 0.6,
            "api_key_env": "OPENAI_API_KEY"
        }
    """
    from ..agents import OpenAIAgent, AgentConfig

    # Create VLM agent based on config
    agent_config = AgentConfig(
        name="AttributeDetector",
        model=config.get("vlm_model", "gpt-4-vision-preview"),
        system_prompt="You are an expert at analyzing images objectively and identifying protected attributes for fairness testing.",
        temperature=0.0,  # Deterministic for detection
    )

    if config.get("vlm_provider") == "openai":
        vlm_agent = OpenAIAgent(agent_config)
    else:
        raise ValueError(f"Unsupported VLM provider: {config.get('vlm_provider')}")

    return ProtectedAttributeDetector(
        vlm_agent=vlm_agent,
        default_confidence_threshold=config.get("confidence_threshold", 0.5)
    )
