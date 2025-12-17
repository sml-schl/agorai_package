"""Counterfactual Image Generator using Image Generation APIs.

Generates counterfactual images by modifying protected attributes
for causal robustness testing.
"""

import base64
import json
import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from .protected_attribute_schema import (
    CounterfactualGenerationRequest,
    CounterfactualGenerationResponse,
    ImageGenerationTool,
    ModificationStrategy,
)


class ImageGenerationAPI(ABC):
    """Abstract base class for image generation APIs."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        base_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate or modify an image.

        Args:
            prompt: Generation/modification prompt
            base_image: Optional base image (URL or base64) for editing
            **kwargs: Additional API-specific parameters

        Returns:
            Dict with generated image URL/base64 and metadata
        """
        pass


class DallE3API(ImageGenerationAPI):
    """OpenAI DALL-E 3 API wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize DALL-E 3 API.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        """
        import openai
        import os

        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        prompt: str,
        base_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using DALL-E 3.

        Note: DALL-E 3 doesn't support image editing, only generation.
        For counterfactuals, we rely on detailed prompts.
        """
        response = self.client.images.generate(
            model=kwargs.get("model", "dall-e-3"),
            prompt=prompt,
            size=kwargs.get("size", "1024x1024"),
            quality=kwargs.get("quality", "standard"),
            n=1,
        )

        return {
            "image_url": response.data[0].url,
            "revised_prompt": response.data[0].revised_prompt,
        }


class StableDiffusionAPI(ImageGenerationAPI):
    """Stable Diffusion API wrapper (via Replicate or HuggingFace)."""

    def __init__(
        self,
        provider: str = "replicate",
        api_key: Optional[str] = None
    ):
        """Initialize Stable Diffusion API.

        Args:
            provider: "replicate" or "huggingface"
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        base_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate or edit image using Stable Diffusion."""
        if self.provider == "replicate":
            return self._generate_replicate(prompt, base_image, **kwargs)
        elif self.provider == "huggingface":
            return self._generate_huggingface(prompt, base_image, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_replicate(
        self,
        prompt: str,
        base_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Replicate API."""
        import replicate
        import os

        # Use img2img if base image provided, else text2img
        if base_image:
            model = "stability-ai/stable-diffusion-img2img:15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d"
            output = replicate.run(
                model,
                input={
                    "image": base_image,
                    "prompt": prompt,
                    "strength": kwargs.get("strength", 0.7),  # How much to modify
                    "guidance_scale": kwargs.get("guidance_scale", 7.5),
                }
            )
        else:
            model = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
            output = replicate.run(
                model,
                input={
                    "prompt": prompt,
                    "width": kwargs.get("width", 1024),
                    "height": kwargs.get("height", 1024),
                    "guidance_scale": kwargs.get("guidance_scale", 7.5),
                }
            )

        # Replicate returns list of URLs
        image_url = output[0] if isinstance(output, list) else output

        return {
            "image_url": image_url,
            "model": model,
        }

    def _generate_huggingface(
        self,
        prompt: str,
        base_image: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using HuggingFace Inference API."""
        import requests
        import os

        api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {self.api_key or os.getenv('HUGGINGFACE_API_KEY')}"}

        payload = {"inputs": prompt}

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

        # Response is the image bytes
        image_base64 = base64.b64encode(response.content).decode("utf-8")

        return {
            "image_base64": image_base64,
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
        }


class CounterfactualGenerator:
    """Generates counterfactual images by modifying protected attributes."""

    def __init__(
        self,
        image_gen_api: ImageGenerationAPI,
        default_tool: ImageGenerationTool = ImageGenerationTool.DALL_E_3
    ):
        """Initialize the generator.

        Args:
            image_gen_api: Image generation API instance
            default_tool: Default tool to use if not specified
        """
        self.image_gen_api = image_gen_api
        self.default_tool = default_tool

    def generate(
        self,
        request: CounterfactualGenerationRequest
    ) -> CounterfactualGenerationResponse:
        """Generate counterfactual image.

        Args:
            request: Generation request with modifications

        Returns:
            Generation response with counterfactual image
        """
        # Build comprehensive modification prompt
        modification_prompt = self._build_modification_prompt(request)

        # Generate counterfactual
        gen_result = self.image_gen_api.generate(
            prompt=modification_prompt,
            base_image=request.original_image_url or request.original_image_base64,
            **request.tool_config
        )

        # Build response
        modifications_applied = [
            {
                "attribute_type": var.original_attribute.attribute_type.value,
                "from": var.original_attribute.value,
                "to": var.target_value,
                "strategy": var.modification_strategy.value,
            }
            for var in request.variations
        ]

        return CounterfactualGenerationResponse(
            counterfactual_image_url=gen_result.get("image_url"),
            counterfactual_image_base64=gen_result.get("image_base64"),
            generation_prompt=modification_prompt,
            modifications_applied=modifications_applied,
            metadata=gen_result
        )

    def _build_modification_prompt(
        self,
        request: CounterfactualGenerationRequest
    ) -> str:
        """Build comprehensive prompt for image modification.

        Args:
            request: Generation request

        Returns:
            Formatted prompt string
        """
        # Start with base instruction
        prompt_parts = [
            "Generate an image with the following modifications:",
            ""
        ]

        # Add each variation
        for var in request.variations:
            var_prompt = var.generate_prompt(
                original_description=request.tool_config.get("original_description", "")
            )
            prompt_parts.append(var_prompt)
            prompt_parts.append("")

        # Add preservation instructions
        if request.preserve_composition:
            prompt_parts.append("IMPORTANT: Maintain the exact same composition and layout.")

        if request.preserve_style:
            prompt_parts.append("IMPORTANT: Maintain the exact same artistic style.")

        if request.preserve_background:
            prompt_parts.append("IMPORTANT: Keep the background completely unchanged.")

        # Add original text if provided
        if request.original_text:
            prompt_parts.append(f"\nOriginal text in image: '{request.original_text}'")
            prompt_parts.append("Keep this text exactly as is in the generated image.")

        return "\n".join(prompt_parts)


def create_generator_from_config(
    config: Dict[str, Any]
) -> CounterfactualGenerator:
    """Factory function to create generator from configuration.

    Args:
        config: Configuration dict

    Returns:
        Configured CounterfactualGenerator

    Example config:
        {
            "tool": "dall-e-3",  # or "stable-diffusion-xl", "replicate"
            "api_key_env": "OPENAI_API_KEY",
            "provider": "openai",  # for SD: "replicate" or "huggingface"
            "default_params": {
                "quality": "standard",
                "size": "1024x1024"
            }
        }
    """
    tool = config.get("tool", "dall-e-3")

    # Create appropriate API wrapper
    if tool == "dall-e-3":
        api = DallE3API()
        tool_enum = ImageGenerationTool.DALL_E_3

    elif tool in ["stable-diffusion-xl", "replicate"]:
        provider = config.get("provider", "replicate")
        api = StableDiffusionAPI(provider=provider)
        tool_enum = ImageGenerationTool.STABLE_DIFFUSION_XL

    else:
        raise ValueError(f"Unsupported tool: {tool}")

    return CounterfactualGenerator(
        image_gen_api=api,
        default_tool=tool_enum
    )


def generate_all_variations(
    generator: CounterfactualGenerator,
    original_image_url: str,
    detected_attributes: List[Any],
    original_text: Optional[str] = None,
    max_variations_per_attribute: int = 3
) -> List[CounterfactualGenerationResponse]:
    """Generate counterfactuals for all detected attributes.

    Args:
        generator: CounterfactualGenerator instance
        original_image_url: URL to original image
        detected_attributes: List of detected ProtectedAttribute objects
        original_text: Original text in image
        max_variations_per_attribute: Max variations to generate per attribute

    Returns:
        List of generated counterfactuals
    """
    from .protected_attribute_schema import (
        get_standard_variations,
        create_counterfactual_variation,
        ModificationStrategy
    )

    counterfactuals = []

    for attribute in detected_attributes:
        # Get standard variations for this attribute
        target_values = get_standard_variations(
            attribute.attribute_type,
            attribute.value
        )

        # Limit variations
        target_values = target_values[:max_variations_per_attribute]

        for target_value in target_values:
            # Create variation config
            variation = create_counterfactual_variation(
                attribute=attribute,
                target_value=target_value,
                strategy=ModificationStrategy.SWAP,
                preserve_elements=["text", "background", "composition"]
            )

            # Create generation request
            request = CounterfactualGenerationRequest(
                original_image_url=original_image_url,
                original_text=original_text,
                variations=[variation],
                generation_tool=generator.default_tool,
                preserve_composition=True,
                preserve_style=True,
                preserve_background=True,
            )

            # Generate counterfactual
            try:
                result = generator.generate(request)
                counterfactuals.append(result)

                # Rate limiting delay
                time.sleep(1)

            except Exception as e:
                print(f"Failed to generate counterfactual for {attribute.attribute_type.value} → {target_value}: {e}")
                continue

    return counterfactuals
